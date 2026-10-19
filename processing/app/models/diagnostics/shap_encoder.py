from typing import Any

import numpy as np
import torch
from captum.attr import ShapleyValueSampling
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")

JOB_ID = ""
JOB_QUEUE = None


class CrossEncoderAttributionWrapper(torch.nn.Module):
    """
    Wrapper for cross-encoder models (encoder-only for re-ranking) for attribution.
    Maps input tokens (query + document) to relevance score.

    Supports separate attribution for query tokens and document tokens.
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
            ablate_target: 'query' or 'document' - which part to ablate
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.query_text = query_text
        self.document_text = document_text
        self.ablate_target = ablate_target
        self.device = next(model.parameters()).device

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
            max_length=512,  # Match Alibaba model's max length
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

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for attribution.

        When ablate_target='query': input_ids contains query tokens (which will be ablated)
        When ablate_target='document': input_ids contains document tokens (which will be ablated)

        Returns the relevance score for the query-document pair.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if self.ablate_target == "query":
            # Reconstruct full input with ablated query tokens

            # Start with the full encoding template
            full_input_ids = self.full_encoding.input_ids.repeat(batch_size, 1).to(
                device
            )

            # Replace query tokens with the ablated versions
            query_len = self.query_end_idx - self.query_start_idx
            if input_ids.shape[1] == query_len:
                full_input_ids[:, self.query_start_idx : self.query_end_idx] = input_ids
            else:
                logger.warning(
                    f"{JOB_ID}: SHAP - Query token length mismatch: expected {query_len}, got {input_ids.shape[1]}"
                )
                # Pad or truncate as needed
                actual_len = min(query_len, input_ids.shape[1])
                full_input_ids[
                    :, self.query_start_idx : self.query_start_idx + actual_len
                ] = input_ids[:, :actual_len]

            full_attention_mask = self.full_encoding.attention_mask.repeat(
                batch_size, 1
            ).to(device)

        elif self.ablate_target == "document":
            # Reconstruct full input with ablated document tokens
            full_input_ids = self.full_encoding.input_ids.repeat(batch_size, 1).to(
                device
            )

            # Replace document tokens with the ablated versions
            doc_len = self.doc_end_idx - self.doc_start_idx
            if input_ids.shape[1] == doc_len:
                full_input_ids[:, self.doc_start_idx : self.doc_end_idx] = input_ids
            else:
                logger.warning(
                    f"{JOB_ID}: SHAP - Document token length mismatch: expected {doc_len}, got {input_ids.shape[1]}"
                )
                actual_len = min(doc_len, input_ids.shape[1])
                full_input_ids[
                    :, self.doc_start_idx : self.doc_start_idx + actual_len
                ] = input_ids[:, :actual_len]

            full_attention_mask = self.full_encoding.attention_mask.repeat(
                batch_size, 1
            ).to(device)
        else:
            raise ValueError(
                f"{JOB_ID}: SHAP - Invalid ablate_target: {self.ablate_target}"
            )

        # Run model forward pass
        try:
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                return_dict=True,
            )
        except Exception as e:
            logger.error(f"{JOB_ID}: SHAP - Model forward pass failed: {e}")
            raise ValueError(f"{JOB_ID}: SHAP - Model forward pass failed: {e}")

        # Extract raw logits for relevance score
        # Alibaba model: tensor([ 2.1387,  2.4609, -1.6729])
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            # Single score output, shape is (batch_size, 1)
            # Binary, shape is (batch_size, 2)
            if logits.shape[-1] > 1:
                # Use logits for positive class
                scores = logits[:, 1]
            else:
                # Single score: flatten to (batch_size,)
                scores = logits.view(-1)
        else:
            raise ValueError(f"{JOB_ID}: SHAP - Model output does not contain logits")

        return scores.unsqueeze(-1)  # Shape: (batch_size, 1)


class ShapEncoderDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the SHAP diagnostic for encoder-only models (cross-encoders).
        """
        super().__init__(
            name="shap_encoder",
            description="Diagnostic for evaluating SHAP attributions on encoder-only re-ranking models.",
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
                f"{JOB_ID}: SHAP - Failed to load model or tokenizer for {model_name}: {e}"
            )
            raise ValueError(
                f"{JOB_ID}: SHAP - Failed to load model or tokenizer for {model_name}: {e}"
            )

        return model, tokenizer

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> list[dict]:
        """
        Run SHAP attribution for encoder-only cross-encoder re-rankers.

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
                f"{JOB_ID}: SHAP - Model name must be specified in parameters with key 'model_name'."
            )
            raise ValueError(
                f"{JOB_ID}: SHAP - Model name must be specified in parameters with key 'model_name'."
            )
        if not parameters.get("dataset_name"):
            logger.error(f"{JOB_ID}: SHAP - Parameter 'dataset_name' is required.")
            raise ValueError(f"{JOB_ID}: SHAP - Parameter 'dataset_name' is required.")
        if not parameters.get("dataset"):
            logger.error(f"{JOB_ID}: SHAP - Parameter 'dataset' is required.")
            raise ValueError(f"{JOB_ID}: SHAP - Parameter 'dataset' is required.")

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

        # SHAP configuration
        n_samples = parameters.get("shap_n_samples", 25)
        perturbations_per_eval = parameters.get("shap_perturbations_per_eval", 1)

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
                    f"{JOB_ID}: SHAP - Empty query for result item {i + 1}, skipping attribution"
                )
                continue

            if not documents:
                logger.warning(
                    f"{JOB_ID}: SHAP - No documents found for query {i + 1}, skipping attribution"
                )
                continue

            logger.info(
                f"{JOB_ID}: SHAP - Processing query {i + 1}/{len(results_to_process)}"
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
                    f"{JOB_ID}: SHAP - Analyzing document {doc_idx + 1}/{len(docs_to_analyze)} (score: {rerank_score:.4f})"
                )

                # Step 1: Compute query token attributions
                query_wrapper = CrossEncoderAttributionWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    query_text=query,
                    document_text=document_text,
                    ablate_target="query",
                )

                shapley_query = ShapleyValueSampling(query_wrapper)

                # Extract just the query tokens for attribution
                query_input_ids = (
                    torch.tensor(query_wrapper.query_encoding.input_ids)
                    .unsqueeze(0)
                    .to(model.device)
                )
                query_tokens = [
                    tokenizer.decode([t])
                    for t in query_wrapper.query_encoding.input_ids
                ]

                try:
                    query_attributions = shapley_query.attribute(
                        query_input_ids,
                        n_samples=n_samples,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=True,
                    )
                    query_attr_values = query_attributions[0].detach().cpu().numpy()
                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: SHAP - Failed to compute query attributions: {e}"
                    )
                    raise ValueError(
                        f"{JOB_ID}: SHAP - Failed to compute query attributions: {e}"
                    )

                # Step 2: Compute document token attributions
                doc_wrapper = CrossEncoderAttributionWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    query_text=query,
                    document_text=document_text,
                    ablate_target="document",
                )

                shapley_doc = ShapleyValueSampling(doc_wrapper)

                # Extract just the document tokens for attribution
                doc_input_ids = (
                    torch.tensor(doc_wrapper.doc_encoding.input_ids)
                    .unsqueeze(0)
                    .to(model.device)
                )
                doc_tokens = [
                    tokenizer.decode([t]) for t in doc_wrapper.doc_encoding.input_ids
                ]

                try:
                    doc_attributions = shapley_doc.attribute(
                        doc_input_ids,
                        n_samples=n_samples,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=True,
                    )
                    doc_attr_values = doc_attributions[0].detach().cpu().numpy()
                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: SHAP - Failed to compute document attributions: {e}"
                    )
                    raise ValueError(
                        f"{JOB_ID}: SHAP - Failed to compute document attributions: {e}"
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

                # Ensure attribution arrays are 1D (defensive normalization)
                query_attr_1d = query_attr_values
                if len(query_attr_values.shape) > 1:
                    query_attr_1d = query_attr_values.squeeze()

                doc_attr_1d = doc_attr_values
                if len(doc_attr_values.shape) > 1:
                    doc_attr_1d = doc_attr_values.squeeze()

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
                                query_tokens_clean, query_attr_1d
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
                            for token, attr_val in zip(doc_tokens_clean, doc_attr_1d)
                        ],
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
