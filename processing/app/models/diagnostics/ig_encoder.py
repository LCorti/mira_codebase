from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")

JOB_ID = ""
JOB_QUEUE = None


class CrossEncoderIGWrapper(torch.nn.Module):
    """
    Wrapper for cross-encoder models (encoder-only for re-ranking) for Integrated Gradients attribution.
    Maps input embeddings (query + document) to relevance score.

    Supports separate attribution for query tokens and document tokens.
    Works with embeddings (not input IDs) to enable gradient computation.
    """

    def __init__(
        self, model, tokenizer, query_text, document_text, ablate_target="query"
    ):
        """
        Args:
            model: The cross-encoder model
            tokenizer: The tokenizer
            query_text: The query string
            document_text: The document string
            ablate_target: 'query' or 'document' - which part to attribute
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.query_text = query_text
        self.document_text = document_text
        self.ablate_target = ablate_target
        self.device = next(model.parameters()).device

        # Get embedding layer for converting IDs to embeddings
        self.embedding_layer = self._get_embedding_layer(model)

        # Tokenize query and document separately to track token boundaries
        self.query_encoding = tokenizer(query_text, add_special_tokens=False)
        self.doc_encoding = tokenizer(document_text, add_special_tokens=False)

        # Create full encoding with special tokens
        # Format: [CLS] query_tokens [SEP] doc_tokens [SEP]
        self.full_encoding = tokenizer(
            query_text,
            document_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Identify token boundaries by finding separator tokens
        full_input_ids = self.full_encoding.input_ids[0].tolist()
        query_tokens = self.query_encoding.input_ids
        doc_tokens = self.doc_encoding.input_ids

        # Find separator token ID (usually [SEP])
        sep_token_id = tokenizer.sep_token_id
        cls_token_id = tokenizer.cls_token_id

        # Find positions of separator tokens
        sep_positions = [
            i for i, token_id in enumerate(full_input_ids) if token_id == sep_token_id
        ]

        if len(sep_positions) < 1:
            # Fallback to manual calculation if no separator found
            self.query_start_idx = 1 if cls_token_id is not None else 0
            self.query_end_idx = self.query_start_idx + len(query_tokens)
            self.doc_start_idx = self.query_end_idx + 1
            self.doc_end_idx = self.doc_start_idx + len(doc_tokens)
        else:
            # Use separator positions to determine boundaries
            self.query_start_idx = 1 if cls_token_id is not None else 0
            self.query_end_idx = sep_positions[0]
            self.doc_start_idx = sep_positions[0] + 1
            self.doc_end_idx = (
                sep_positions[1] if len(sep_positions) > 1 else len(full_input_ids) - 1
            )

        # Create full embedding template (will be used in forward pass)
        self.full_input_embeddings = self.embedding_layer(
            self.full_encoding.input_ids.to(self.device)
        )

    def _get_embedding_layer(self, model):
        """
        Get embedding layer for any AutoModelForSequenceClassification.

        Works for BERT, RoBERTa, ELECTRA, DeBERTa, etc.
        """
        try:
            # Generic approach (works for most models)
            return model.get_input_embeddings()
        except Exception:
            # Fallback to common patterns
            if hasattr(model, "bert"):
                return model.bert.embeddings.word_embeddings
            elif hasattr(model, "roberta"):
                return model.roberta.embeddings.word_embeddings
            elif hasattr(model, "electra"):
                return model.electra.embeddings.word_embeddings
            elif hasattr(model, "deberta"):
                return model.deberta.embeddings.word_embeddings
            else:
                logger.error(f"{JOB_ID}: IG - Could not find embedding layer for model")
                raise ValueError(
                    f"{JOB_ID}: IG - Could not find embedding layer for model"
                )

    def construct_baseline_input_ids(self):
        """
        Construct baseline input IDs filled with padding tokens.

        Returns:
            Baseline tensor with same shape as input, filled with pad_token_id
        """
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            logger.warning(
                f"{JOB_ID}: IG - No pad_token_id found in tokenizer. Using 0 as baseline."
            )
            pad_token_id = 0

        if self.ablate_target == "query":
            baseline_ids = torch.full(
                (len(self.query_encoding.input_ids),),
                fill_value=pad_token_id,
                dtype=torch.long,
            )
        elif self.ablate_target == "document":
            baseline_ids = torch.full(
                (len(self.doc_encoding.input_ids),),
                fill_value=pad_token_id,
                dtype=torch.long,
            )
        else:
            raise ValueError(
                f"{JOB_ID}: IG - Invalid ablate_target: {self.ablate_target}"
            )

        return baseline_ids.unsqueeze(0)  # Add batch dimension

    def forward(self, input_embeddings):
        """
        Forward pass for Integrated Gradients attribution.

        Args:
            input_embeddings: Embedding vectors (NOT input_ids) for query or document tokens
                Shape: (batch_size, seq_len, hidden_dim)

        Returns:
            Relevance scores: (batch_size,) - scalar per example
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device

        if self.ablate_target == "query":
            # Reconstruct full embeddings with interpolated query embeddings

            # Start with the full embedding template
            full_embeddings = self.full_input_embeddings.repeat(batch_size, 1, 1).to(
                device
            )

            # Replace query embeddings with the interpolated versions
            query_len = self.query_end_idx - self.query_start_idx
            if input_embeddings.shape[1] == query_len:
                full_embeddings[:, self.query_start_idx : self.query_end_idx, :] = (
                    input_embeddings
                )
            else:
                logger.warning(
                    f"{JOB_ID}: IG - Query embedding length mismatch: expected {query_len}, got {input_embeddings.shape[1]}"
                )
                # Pad or truncate as needed
                actual_len = min(query_len, input_embeddings.shape[1])
                full_embeddings[
                    :, self.query_start_idx : self.query_start_idx + actual_len, :
                ] = input_embeddings[:, :actual_len, :]

        elif self.ablate_target == "document":
            # Reconstruct full embeddings with interpolated document embeddings
            full_embeddings = self.full_input_embeddings.repeat(batch_size, 1, 1).to(
                device
            )

            # Replace document embeddings with the interpolated versions
            doc_len = self.doc_end_idx - self.doc_start_idx
            if input_embeddings.shape[1] == doc_len:
                full_embeddings[:, self.doc_start_idx : self.doc_end_idx, :] = (
                    input_embeddings
                )
            else:
                logger.warning(
                    f"{JOB_ID}: IG - Document embedding length mismatch: expected {doc_len}, got {input_embeddings.shape[1]}"
                )
                actual_len = min(doc_len, input_embeddings.shape[1])
                full_embeddings[
                    :, self.doc_start_idx : self.doc_start_idx + actual_len, :
                ] = input_embeddings[:, :actual_len, :]
        else:
            raise ValueError(
                f"{JOB_ID}: IG - Invalid ablate_target: {self.ablate_target}"
            )

        # Create attention mask (all ones for valid positions)
        attention_mask = torch.ones(
            batch_size, full_embeddings.shape[1], dtype=torch.long, device=device
        )

        # Run model forward pass with embeddings
        try:
            outputs = self.model(
                inputs_embeds=full_embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            )
        except Exception as e:
            logger.error(f"{JOB_ID}: IG - Model forward pass failed: {e}")
            raise ValueError(f"{JOB_ID}: IG - Model forward pass failed: {e}")

        # Extract raw logits for relevance score
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            # Binary classification or single score
            if logits.shape[-1] > 1:
                # Use logits for positive class
                scores = logits[:, 1]
            else:
                # Single score: flatten to (batch_size,)
                scores = logits.view(-1)
        else:
            raise ValueError(f"{JOB_ID}: IG - Model output does not contain logits")

        return scores  # Shape: (batch_size,)


class IGEncoderDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the Integrated Gradients diagnostic for encoder-only models (cross-encoders).
        """
        super().__init__(
            name="integrated_gradients_encoder",
            description="Diagnostic for evaluating Integrated Gradients attributions on encoder-only re-ranking models.",
        )

    def _preprocess_model_parameters(self, parameters):
        """
        Preprocess model parameters to convert string representations to actual PyTorch types.
        """
        processed = parameters.copy()

        # Remove invalid parameters
        invalid_params = ["architecture"]
        for param in invalid_params:
            processed.pop(param, None)

        # Handle torch_dtype conversion
        if "torch_dtype" in processed:
            dtype_value = processed["torch_dtype"]
            if isinstance(dtype_value, str):
                dtype_map = {
                    "float16": torch.float16,
                    "torch.float16": torch.float16,
                    "float32": torch.float32,
                    "torch.float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                    "torch.bfloat16": torch.bfloat16,
                    "float": torch.float32,
                    "half": torch.float16,
                    "auto": "auto",
                }
                processed["torch_dtype"] = dtype_map.get(dtype_value, torch.float32)

        # Handle device_map - ensure it's properly formatted
        if (
            "device_map" in processed
            and isinstance(processed["device_map"], str)
            and processed["device_map"].lower() in ["auto", "balanced", "sequential"]
        ):
            processed["device_map"] = processed["device_map"].lower()

        return processed

    def load_model_and_tokenizer(
        self,
        model_name,
        model_parameters: dict | None = None,
        tokenizer_parameters: dict | None = None,
    ):
        """
        Load encoder-only model and tokenizer for cross-encoder re-ranking.
        """

        if not model_parameters:
            model_parameters = {}
        if not tokenizer_parameters:
            tokenizer_parameters = {}

        processed_model_params = self._preprocess_model_parameters(model_parameters)
        processed_tokenizer_params = tokenizer_parameters.copy()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, **processed_tokenizer_params
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **processed_model_params
            )
        except Exception as e:
            logger.error(
                f"{JOB_ID}: IG - Failed to load model or tokenizer for {model_name}: {e}"
            )
            raise ValueError(
                f"{JOB_ID}: IG - Failed to load model or tokenizer for {model_name}: {e}"
            )

        return model, tokenizer

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> list[dict]:
        """
        Run Integrated Gradients attribution for encoder-only cross-encoder re-rankers.

        :param results: List of objects per prediction containing 'input' (query), 'documents', etc.
        :param parameters: Additional parameters for the diagnostic computation.
        :param kwargs: Additional keyword arguments, including 'job_id' and 'job_queue'.
        """

        global JOB_ID, JOB_QUEUE
        JOB_ID = kwargs.get("job_id", "")
        JOB_QUEUE = kwargs.get("job_queue")

        if not parameters:
            parameters = {}

        if not parameters.get("model_name"):
            logger.error(
                f"{JOB_ID}: IG - Model name must be specified in parameters with key 'model_name'."
            )
            raise ValueError(
                f"{JOB_ID}: IG - Model name must be specified in parameters with key 'model_name'."
            )
        if not parameters.get("dataset_name"):
            logger.error(f"{JOB_ID}: IG - Parameter 'dataset_name' is required.")
            raise ValueError(f"{JOB_ID}: IG - Parameter 'dataset_name' is required.")
        if not parameters.get("dataset"):
            logger.error(f"{JOB_ID}: IG - Parameter 'dataset' is required.")
            raise ValueError(f"{JOB_ID}: IG - Parameter 'dataset' is required.")

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        model_name = parameters.get("model_name")
        model, tokenizer = self.load_model_and_tokenizer(
            model_name,
            model_parameters=parameters.get("model_parameters", {}),
            tokenizer_parameters=parameters.get("tokenizer_parameters", {}),
        )

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)
        model.eval()

        # Get parameters
        num_samples = parameters.get("dataset", {}).get("num_samples", None)
        num_samples = (
            int(num_samples)
            if num_samples is not None
            and isinstance(num_samples, (int, str))
            and str(num_samples).isdigit()
            else None
        )

        # RAG-specific: how many documents per query to analyze
        num_docs_to_analyze = parameters.get(
            "num_docs_to_analyze", 1e3
        )  # Default: analyze top-1000 document

        # IG configuration
        n_steps = parameters.get("ig_n_steps", 50)
        method = parameters.get("ig_method", "gausslegendre")
        internal_batch_size = parameters.get("ig_internal_batch_size", None)
        convergence_delta_threshold = parameters.get(
            "ig_convergence_delta_threshold", 0.05
        )
        normalize_attributions = parameters.get("ig_normalize_attributions", False)

        # Validate method parameter
        valid_methods = [
            "riemann_left",
            "riemann_right",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]
        if method not in valid_methods:
            logger.warning(
                f"{JOB_ID}: IG - Invalid method '{method}'. Using 'gausslegendre' as default."
            )
            method = "gausslegendre"

        logger.info(
            f"{JOB_ID}: IG - Configuration: n_steps={n_steps}, method={method}, normalize={normalize_attributions}"
        )

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        # Use diagnostic_sample if provided
        diagnostic_sample = parameters.get("diagnostic_sample", {})
        if diagnostic_sample and diagnostic_sample.get("input") is not None:
            results_to_process = [
                r for r in results if r["input"] == diagnostic_sample["input"]
            ]
        else:
            results_to_process = results

        attribution_results = []

        for i, result_item in enumerate(results_to_process):
            if num_samples is not None and i >= num_samples:
                break

            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            query = result_item.get("input", "")
            documents = result_item.get("documents", [])

            if query == "":
                logger.warning(
                    f"{JOB_ID}: IG - Empty query for result item {i + 1}, skipping attribution"
                )
                continue

            if not documents:
                logger.warning(
                    f"{JOB_ID}: IG - No documents found for query {i + 1}, skipping attribution"
                )
                continue

            logger.info(
                f"{JOB_ID}: IG - Processing query {i + 1}/{len(results_to_process)}"
            )

            # Analyze top-k documents for this query
            docs_to_analyze = documents[: min(num_docs_to_analyze, len(documents))]

            query_doc_attributions = []

            # Process each document for the current query
            for doc_idx, doc_item in enumerate(docs_to_analyze):
                self._check_job_cancellation(JOB_ID, JOB_QUEUE)

                document_text = doc_item.get("doc_text", "")
                rerank_score = doc_item.get("rerank_score", 0.0)
                doc_id = doc_item.get("doc_id", f"doc_{doc_idx}")

                logger.info(
                    f"{JOB_ID}: IG - Analyzing document {doc_idx + 1}/{len(docs_to_analyze)} (score: {rerank_score:.4f})"
                )

                # Step 1: Compute query token attributions
                query_wrapper = CrossEncoderIGWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    query_text=query,
                    document_text=document_text,
                    ablate_target="query",
                )

                ig_query = IntegratedGradients(query_wrapper)

                # Extract query tokens and construct input/baseline embeddings
                query_input_ids = (
                    torch.tensor(query_wrapper.query_encoding.input_ids)
                    .unsqueeze(0)
                    .to(model.device)
                )
                query_baseline_ids = query_wrapper.construct_baseline_input_ids().to(
                    model.device
                )
                query_tokens = [
                    tokenizer.decode([t])
                    for t in query_wrapper.query_encoding.input_ids
                ]

                # Convert to embeddings
                query_input_embeddings = query_wrapper.embedding_layer(query_input_ids)
                query_baseline_embeddings = query_wrapper.embedding_layer(
                    query_baseline_ids
                )

                try:
                    query_attributions, query_delta = ig_query.attribute(
                        inputs=query_input_embeddings,
                        baselines=query_baseline_embeddings,
                        n_steps=n_steps,
                        method=method,
                        internal_batch_size=internal_batch_size,
                        return_convergence_delta=True,
                    )

                    # Sum attributions across embedding dimension (hidden_dim)
                    # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len)
                    query_attr_summed = query_attributions.sum(dim=-1)

                    # Optional normalization
                    if normalize_attributions:
                        query_attr_summed = query_attr_summed / torch.norm(
                            query_attr_summed
                        )

                    query_attr_values = query_attr_summed[0].detach().cpu().numpy()
                    query_delta_value = query_delta.item()

                    # Validate convergence
                    if abs(query_delta_value) > convergence_delta_threshold:
                        logger.warning(
                            f"{JOB_ID}: IG - High convergence delta for query ({query_delta_value:.4f}). "
                            f"Consider increasing ig_n_steps (current: {n_steps})"
                        )

                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: IG - Failed to compute query attributions: {e}"
                    )
                    raise ValueError(
                        f"{JOB_ID}: IG - Failed to compute query attributions: {e}"
                    )

                # Step 2: Compute document token attributions
                doc_wrapper = CrossEncoderIGWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    query_text=query,
                    document_text=document_text,
                    ablate_target="document",
                )

                ig_doc = IntegratedGradients(doc_wrapper)

                # Extract document tokens and construct input/baseline embeddings
                doc_input_ids = (
                    torch.tensor(doc_wrapper.doc_encoding.input_ids)
                    .unsqueeze(0)
                    .to(model.device)
                )
                doc_baseline_ids = doc_wrapper.construct_baseline_input_ids().to(
                    model.device
                )
                doc_tokens = [
                    tokenizer.decode([t]) for t in doc_wrapper.doc_encoding.input_ids
                ]

                # Convert to embeddings
                doc_input_embeddings = doc_wrapper.embedding_layer(doc_input_ids)
                doc_baseline_embeddings = doc_wrapper.embedding_layer(doc_baseline_ids)

                try:
                    doc_attributions, doc_delta = ig_doc.attribute(
                        inputs=doc_input_embeddings,
                        baselines=doc_baseline_embeddings,
                        n_steps=n_steps,
                        method=method,
                        internal_batch_size=internal_batch_size,
                        return_convergence_delta=True,
                    )

                    # Sum attributions across embedding dimension
                    doc_attr_summed = doc_attributions.sum(dim=-1)

                    # Optional normalization
                    if normalize_attributions:
                        doc_attr_summed = doc_attr_summed / torch.norm(doc_attr_summed)

                    doc_attr_values = doc_attr_summed[0].detach().cpu().numpy()
                    doc_delta_value = doc_delta.item()

                    # Validate convergence
                    if abs(doc_delta_value) > convergence_delta_threshold:
                        logger.warning(
                            f"{JOB_ID}: IG - High convergence delta for document ({doc_delta_value:.4f}). "
                            f"Consider increasing ig_n_steps (current: {n_steps})"
                        )

                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: IG - Failed to compute document attributions: {e}"
                    )
                    raise ValueError(
                        f"{JOB_ID}: IG - Failed to compute document attributions: {e}"
                    )

                # Clean token representations (remove special characters from tokenization)
                query_tokens_clean = [
                    token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()
                    for token in query_tokens
                ]
                doc_tokens_clean = [
                    token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()
                    for token in doc_tokens
                ]

                # Store attributions for this query-document pair
                query_doc_attributions.append(
                    {
                        "document_id": doc_id,
                        "document_text": document_text,
                        "rerank_score": round(float(rerank_score), 4)
                        if not np.isnan(rerank_score)
                        else 0.0,
                        "query_attributions": [
                            [
                                token,
                                round(
                                    float(
                                        np.asarray(attr_val, dtype=np.float16).item()
                                    ),
                                    4,
                                ),
                            ]
                            for token, attr_val in zip(
                                query_tokens_clean, query_attr_values
                            )
                        ],
                        "document_attributions": [
                            [
                                token,
                                round(
                                    float(
                                        np.asarray(attr_val, dtype=np.float16).item()
                                    ),
                                    4,
                                ),
                            ]
                            for token, attr_val in zip(
                                doc_tokens_clean, doc_attr_values
                            )
                        ],
                        "query_convergence_delta": round(float(query_delta_value), 4),
                        "document_convergence_delta": round(float(doc_delta_value), 4),
                    }
                )

            attribution_results.append(
                {
                    "query": query,
                    "query_id": result_item.get("query_id", str(i)),
                    "num_documents_analyzed": len(query_doc_attributions),
                    "attributions": query_doc_attributions,
                }
            )

        return attribution_results
