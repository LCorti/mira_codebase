import gc

import bm25s
import numpy as np
import Stemmer
import torch

from app.config.registry import (
    get_dataset_loader,
    get_diagnostic_registry,
    get_metric_registry,
    get_model_loader,
)
from app.services.logging_service import LoggingService
from app.services.task_service import TaskService
from app.services.tensor_parser import tensor_to_string

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")

AVAILABLE_TASK_TYPES = TaskService().get_available_tasks()

JOB_ID = None


def _check_job_cancellation(job_id: str, job_queue):
    """
    Helper function to check if a job has been cancelled.

    Args:
        job_id: The ID of the job to check
        job_queue: The job queue instance to check cancellation status

    Raises:
        Exception: If the job has been cancelled
    """
    if job_queue and job_id and job_queue.is_job_cancelled(job_id):
        from app.models.job import JobCancelledException

        logger.info(f"{job_id}: Job execution cancelled during inference/analysis")
        raise JobCancelledException(f"Job {job_id} was cancelled during execution")


def run_inference(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
) -> tuple[list, list]:
    """
    Run inference on the dataset using the model and tokenizer.
    Automatically selects the appropriate inference function based on model type.
    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.
        job_id: The ID of the job (for logging and cancellation)
        job_queue: The job queue instance (for cancellation checking)
    Returns:
        results: A list of results from the inference.
        logits: A list of logits from the inference (if applicable).
    """
    # Set global job id
    global JOB_ID
    JOB_ID = job_id

    # Check for cancellation before starting
    _check_job_cancellation(job_id, job_queue)

    if next(model.parameters()).device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Get model name from parameters or try to infer it
    model_loader = get_model_loader()
    model_name = model_loader.get_model_name()
    if not model_name:
        # Try to get model name from the model's config
        if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            model_name = model.config.name_or_path
        else:
            logger.warning(f"{JOB_ID}: Could not determine model name.")
            raise ValueError("Model name is required for inference.")

    # Determine model type and route to appropriate function
    model_type = model_loader.get_model_type()
    logger.info(f"{JOB_ID}: Detected model type: {model_type} for model: {model_name}")

    # Check for cancellation before calling specific inference function
    _check_job_cancellation(job_id, job_queue)

    try:
        if model_type == "encoder-only":
            return run_inference_encoder(
                model, tokenizer, dataset, parameters, job_id, job_queue
            )
        elif model_type == "encoder-decoder":
            return run_inference_encoder_decoder(
                model, tokenizer, dataset, parameters, job_id, job_queue
            )
        elif model_type == "decoder-only":
            return run_inference_decoder(
                model, tokenizer, dataset, parameters, job_id, job_queue
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"{JOB_ID}: Error during inference for model '{model_name}': {e}")
        raise


def run_inference_encoder(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run inference on encoder-only models for various tasks.
    Args:
        model: The encoder-only model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.

    Returns:
        results: A list of results from the inference.
    """
    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    # Get the task type from parameters
    task = parameters.get("task_type")
    if task == "text-retrieval":  # text-retrieval (Retrieval-Augmented Generation)
        return run_inference_embedding(
            model, tokenizer, dataset, parameters, job_id, job_queue
        )
    else:
        raise ValueError(
            f"Unsupported encoder task: {task}. Supported tasks: {AVAILABLE_TASK_TYPES['encoder-only']}"
        )


def run_inference_embedding(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run text-retrieval inference: BM25 sparse retrieval + dense cross-encoder re-ranking.
    Args:
        model: The cross-encoder model for re-ranking.
        tokenizer: The tokenizer for the cross-encoder.
        dataset: The dataset containing queries and document corpus.
        parameters: Additional dictionary of parameters for text-retrieval inference.
    Returns:
        results: A list of results with retrieved and re-ranked documents.
    """
    dataset_parameters = parameters.get("dataset", {})
    split = dataset_parameters.get("split")
    query_column = dataset_parameters.get("input_column")
    query_id_column = dataset_parameters.get("query_id_column", "_id")
    num_samples = dataset_parameters.get("num_samples", None)
    num_samples = (
        int(num_samples)
        if num_samples is not None
        and isinstance(num_samples, (int, str))
        and str(num_samples).isdigit()
        else None
    )

    # text-retrieval-specific parameters
    rag_parameters = parameters.get("task_parameters", {})
    k_sparse = rag_parameters.get("k_sparse", 10)  # Top-k for sparse retrieval
    k_dense = rag_parameters.get("k_dense", 10)  # Top-k for final ranking

    batch_size = rag_parameters.get("batch_size", 1)  # Batch size for reranking
    logger.info(
        f"{JOB_ID}: text-retrieval inference parameters: k_sparse={k_sparse}, k_dense={k_dense}, batch_size={batch_size}"
    )

    stopwords = rag_parameters.get(
        "stopword_language", None
    )  # Optional stopwords for BM25s (e.g., "english")
    stemming_language = rag_parameters.get(
        "stemming_language", None
    )  # Optional stemming language for BM25s (e.g., "english")

    # Qrels parameters
    qrels_name = rag_parameters.get("qrels_name")
    qrels_split = rag_parameters.get("qrels_split", "train")
    qrels_query_id_column = rag_parameters.get("qrels_query_id_column", "query-id")
    qrels_corpus_id_column = rag_parameters.get("qrels_corpus_id_column", "corpus-id")
    qrels_score_column = rag_parameters.get(
        "qrels_score_column", ""
    )  # Optional score column

    tokenizer_parameters = parameters.get("tokenizer", {}).get("inference", {})
    model_max_length = getattr(tokenizer, "model_max_length", 16384)
    if model_max_length > 1e6:  # Some tokenizers have unrealistic defaults
        model_max_length = 16384

    tokenizer_parameters = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": tokenizer_parameters.get("max_length", model_max_length),
        "padding": True,
        **tokenizer_parameters,
    }
    # == == Temporary patch == ==
    tokenizer_parameters["max_length"] = 4096

    _check_job_cancellation(job_id, job_queue)

    # Get corpus and queries
    try:
        corpus_name = rag_parameters.get("corpus_name")
        corpus_split = rag_parameters.get("corpus_split", "train")
        corpus_id_column = rag_parameters.get("corpus_id_column", "_id")
        corpus_text_column = rag_parameters.get("corpus_text_column", "text")

        try:
            corpus_dataset = get_dataset_loader().load(
                dataset_name=corpus_name, config=corpus_split, cache=False
            )
        except Exception as e:
            logger.error(
                f"{JOB_ID}: Error loading corpus dataset '{corpus_name}' with split '{corpus_split}': {e}"
            )
            raise

        corpus_dataset = corpus_dataset[corpus_split]
        corpus_doc_ids = corpus_dataset[corpus_id_column]  # Document IDs (strings)
        corpus = corpus_dataset[corpus_text_column]
        queries = dataset[split][query_column]

        # Get query IDs if available
        query_ids = None
        if query_id_column in dataset[split].column_names:
            query_ids = dataset[split][query_id_column]
        elif qrels_name:
            logger.error(
                f"{JOB_ID}: Query ID column '{query_id_column}' not found in dataset, but qrels dataset '{qrels_name}' is specified. Query IDs are required for matching ground truth labels."
            )
            raise ValueError(
                f"Query ID column '{query_id_column}' not found in dataset split '{split}', but qrels are provided. Cannot match ground truth without query IDs."
            )

        # Convert to regular Python lists to release dataset memory immediately
        corpus = list(corpus)
        corpus_doc_ids = [str(doc_id) for doc_id in corpus_doc_ids]  # Ensure string IDs
        queries = list(queries)
        if query_ids is not None:
            query_ids = [str(qid) for qid in query_ids]  # Ensure string IDs

        # Delete dataset objects immediately after extraction
        del corpus_dataset
        gc.collect()

    except Exception as e:
        logger.error(f"{JOB_ID}: Error loading corpus '{corpus_name}': {e}")
        raise

    # Load qrels if provided
    qrels_map = {}
    if qrels_name:
        try:
            from collections import defaultdict

            logger.info(f"{JOB_ID}: Loading qrels dataset '{qrels_name}'...")
            qrels_dataset = get_dataset_loader().load(
                dataset_name=qrels_name, cache=False
            )[qrels_split]

            # Extract columns once for efficiency
            qrels_query_ids = [str(qid) for qid in qrels_dataset[qrels_query_id_column]]
            qrels_corpus_ids = [
                str(cid) for cid in qrels_dataset[qrels_corpus_id_column]
            ]
            qrels_scores = (
                qrels_dataset[qrels_score_column]
                if qrels_score_column
                and qrels_score_column in qrels_dataset.column_names
                else [1] * len(qrels_query_ids)
            )

            # Build mapping: query_id -> dict of {doc_id: score}
            qrels_map = defaultdict(dict)
            for qrel_qid, qrel_cid, qrel_score in zip(
                qrels_query_ids, qrels_corpus_ids, qrels_scores
            ):
                qrels_map[qrel_qid][qrel_cid] = qrel_score

            qrels_map = dict(qrels_map)  # Convert back to regular dict
            logger.info(
                f"{JOB_ID}: Loaded qrels for {len(qrels_map)} queries ({len(qrels_query_ids)} total relevance judgments)"
            )

            # Delete temporary lists and dataset immediately
            del qrels_dataset, qrels_query_ids, qrels_corpus_ids, qrels_scores
            gc.collect()

        except Exception as e:
            logger.error(f"{JOB_ID}: Error loading qrels '{qrels_name}': {e}")
            raise
    else:
        logger.info(
            f"{JOB_ID}: No qrels dataset specified for text-retrieval inference."
        )

    if k_sparse > len(corpus):
        logger.warning(
            f"{JOB_ID}: k_sparse ({k_sparse}) is greater than corpus size ({len(corpus)}). Adjusting k_sparse to corpus size."
        )
        k_sparse = len(corpus)

    if k_dense > len(corpus):
        logger.warning(
            f"{JOB_ID}: k_dense ({k_dense}) is greater than corpus size ({len(corpus)}). Adjusting k_dense to corpus size."
        )
        k_dense = len(corpus)

    if k_dense > k_sparse:
        logger.error(
            f"{JOB_ID}: k_dense ({k_dense}) cannot be greater than k_sparse ({k_sparse}). Dense re-ranking operates on sparse retrieval results."
        )
        raise ValueError(
            f"k_dense ({k_dense}) must be less than or equal to k_sparse ({k_sparse})"
        )

    # Check for cancellation before indexing corpus
    _check_job_cancellation(job_id, job_queue)

    try:
        stemmer = Stemmer.Stemmer(stemming_language) if stemming_language else None

        logger.info(f"{JOB_ID}: Tokenizing corpus ({len(corpus)} documents)...")

        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer)

        logger.info(f"{JOB_ID}: Corpus tokenization complete. Building BM25 index...")
    except Exception as e:
        logger.error(f"{JOB_ID}: Error during corpus tokenization for BM25s: {e}")
        raise

    try:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)  # Update to load if exists
        logger.info(f"{JOB_ID}: BM25 index built successfully")

        del corpus_tokens
        gc.collect()
    except Exception as e:
        logger.error(f"{JOB_ID}: Error during BM25s indexing: {e}")
        raise

    # Create lookup dictionaries for corpus
    # Map BM25 index (0-based integer) -> corpus text
    corpus_lookup = {i: corpus[i] for i in range(len(corpus))}
    # Map BM25 index (0-based integer) -> document ID (string)
    corpus_doc_ids_lookup = {i: corpus_doc_ids[i] for i in range(len(corpus_doc_ids))}

    # Free the large lists immediately
    del corpus, corpus_doc_ids
    gc.collect()

    model.eval()
    device = next(model.parameters()).device

    results = []
    logits = []
    for i, query in enumerate(queries):
        _check_job_cancellation(job_id, job_queue)

        if num_samples is not None and i >= num_samples:
            break
        logger.info(
            f"{job_id}: Processing sample {i + 1}/{num_samples if num_samples is not None else len(queries)}"
        )

        # Get query ID for this query
        query_id = query_ids[i] if query_ids is not None else str(i)

        # Step 1: BM25 sparse retrieval
        try:
            tokenized_query = bm25s.tokenize(
                query, stopwords=stopwords, stemmer=stemmer
            )
            result = retriever.retrieve(query_tokens=tokenized_query, k=k_sparse)
        except Exception as e:
            logger.error(
                f"{JOB_ID}: Error during BM25s retrieval for query {i + 1}: {e}"
            )
            raise

        # Check if any results were found
        if result.documents.size == 0 or len(result.documents[0]) == 0:
            logger.warning(f"{JOB_ID}: No candidates found for query {i + 1}")
            results.append(
                {
                    "input": query,
                    "query_id": query_id,
                    "documents": [],
                    "num_candidates": 0,
                    "prediction": [],
                }
            )
            continue

        logger.info(
            f"{JOB_ID}: Retrieved {result.documents.size} candidates from BM25 for query {i + 1}"
        )

        # Extract the sparse candidates
        sparse_candidates = []
        for j in range(len(result.documents[0])):
            bm25_doc_idx = int(result.documents[0][j])
            bm25_score = result.scores[0][j]
            doc_text = corpus_lookup[bm25_doc_idx]

            doc_id = corpus_doc_ids_lookup[bm25_doc_idx]  # Get string document ID
            sparse_candidates.append((doc_id, doc_text, float(bm25_score)))

        # Step 2: Dense re-ranking with cross-encoder
        pairs = [[query, doc_text] for _, doc_text, _ in sparse_candidates]
        rerank_scores = []

        for batch_start in range(0, len(pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(pairs))
            batch_pairs = pairs[batch_start:batch_end]

            _check_job_cancellation(
                job_id, job_queue
            )  # Check for cancellation before each batch

            logger.info(
                f"{job_id}: Re-ranking batch {batch_start // batch_size + 1}/{((len(pairs) - 1) // batch_size) + 1} for query {i + 1}"
            )

            with torch.no_grad():
                try:
                    inputs = tokenizer(batch_pairs, **tokenizer_parameters)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: Error during tokenization for re-ranking batch starting at index {batch_start}: {e}"
                    )
                    raise

                try:
                    outputs = model(**inputs)
                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: Error during model forward pass for re-ranking batch starting at index {batch_start}: {e}"
                    )
                    raise

                # Extract scores based on model output format
                if (
                    hasattr(outputs, "logits") and outputs.logits.shape[-1] > 1
                ):  # Binary classification: use softmax and take positive class [-0.5, 2.1] -> [0.07, 0.93]
                    batch_scores = (
                        torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                    )
                    logits_strs = [tensor_to_string(logit) for logit in outputs.logits]
                    logits.extend(logits_strs)
                else:  # Single score: use sigmoid [[2.1], [-0.8], [1.5]] -> [0.89, 0.31, 0.82]
                    batch_scores = (
                        torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
                    )
                    logits_strs = [tensor_to_string(logit) for logit in outputs.logits]
                    logits.extend(logits_strs)

                # Clean up batch tensors immediately
                del outputs, inputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Store scores for this batch
            for j, score in enumerate(batch_scores):
                global_idx = batch_start + j
                doc_id, doc_text, bm25_score = sparse_candidates[global_idx]
                rerank_scores.append(
                    {
                        "doc_id": doc_id,  # Use string document ID
                        "doc_text": doc_text,
                        "bm25_score": bm25_score,
                        "rerank_score": float(score),
                    }
                )

        # Sort by re-ranking score and take top-k
        reranked_docs = sorted(
            rerank_scores, key=lambda x: x["rerank_score"], reverse=True
        )[:k_dense]

        # Prepare result
        result_item = {
            "input": query,
            "query_id": query_id,
            "documents": reranked_docs,
            "num_candidates": len(sparse_candidates),
            "prediction": [doc["rerank_score"] for doc in reranked_docs],
        }

        # Add ground truth if qrels are available
        if qrels_map and query_id in qrels_map:
            relevant_docs_scores = qrels_map[query_id]
            # Create relevance labels with scores for the reranked documents
            result_item["reference"] = [
                relevant_docs_scores.get(doc["doc_id"], 0) for doc in reranked_docs
            ]
        else:
            if qrels_map:
                logger.warning(
                    f"{JOB_ID}: No relevance judgments found for query ID '{query_id}' in qrels. Available qrel query IDs (first 10): {list(qrels_map.keys())[:10]}"
                )
            else:
                logger.info(
                    f"{JOB_ID}: No qrels dataset specified for query ID '{query_id}'."
                )

        results.append(result_item)

        # Aggressive memory cleanup after each query
        del rerank_scores, reranked_docs, pairs, sparse_candidates
        if qrels_map and query_id in qrels_map:
            del relevant_docs_scores
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, logits


def run_inference_encoder_decoder(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run inference on encoder-decoder models (like T5, BART, FLAN-T5).
    Args:
        model: The encoder-decoder model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.
    Returns:
        results: A list of results from the inference.
    """

    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    task = parameters.get("task_type")
    if task == "summarization":  # Summarization
        return run_inference_summarization(
            model, tokenizer, dataset, parameters, job_id, job_queue
        )
    else:
        raise ValueError(
            f"Unsupported encoder task: {task}. Supported tasks: {AVAILABLE_TASK_TYPES['encoder-decoder']}"
        )


def run_inference_summarization(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run inference on summarization tasks.
    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.
    Returns:
        results: A list of results from the inference.
    """
    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    dataset_parameters = parameters.get("dataset", {})
    split = dataset_parameters.get("split", "test")
    input_column = dataset_parameters.get("input_column")
    reference_column = dataset_parameters.get("reference_column", "reference")
    num_samples = dataset_parameters.get("num_samples", None)
    num_samples = (
        int(num_samples)
        if num_samples is not None
        and isinstance(num_samples, (int, str))
        and str(num_samples).isdigit()
        else None
    )

    if not input_column:
        raise ValueError("Input column must be specified in dataset parameters.")

    model_parameters = parameters.get("model", {}).get("inference", {})

    tokenizer_parameters = parameters.get("tokenizer", {}).get("inference", {})
    model_max_length = getattr(tokenizer, "model_max_length", 16384)
    if model_max_length > 1e6:  # Some tokenizers have unrealistic defaults
        model_max_length = 16384

    tokenizer_parameters = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": tokenizer_parameters.get("max_length", model_max_length),
        "padding": True,
        **tokenizer_parameters,
    }

    # == == Temporary patch == ==
    tokenizer_parameters["max_length"] = 4096

    task_parameters = parameters.get("task_parameters", {})
    system_message = task_parameters.get("system_message", "")

    device = next(model.parameters()).device

    results = []
    logits = []
    for i, sample in enumerate(dataset[split]):
        _check_job_cancellation(job_id, job_queue)

        if num_samples is not None and i >= num_samples:
            break
        logger.info(
            f"{job_id}: Processing sample {i + 1}/{num_samples if num_samples is not None else len(dataset[split])}"
        )

        prompt = sample.get(input_column, "")
        if system_message:
            prompt = (
                system_message.replace("<input>", prompt)
                if system_message and "<input>" in system_message
                else f"{system_message} {prompt}"
            )  # Check for placeholder & replace, otherwise append

        # Encode input
        inputs = tokenizer(prompt, **tokenizer_parameters)

        input_ids = inputs.input_ids.to(device)
        attention_mask = (
            inputs.attention_mask.to(device)
            if hasattr(inputs, "attention_mask")
            else None
        )
        if attention_mask is not None:
            model_parameters["attention_mask"] = attention_mask
        if "return_dict_in_generate" not in model_parameters:
            model_parameters["return_dict_in_generate"] = True
        if "output_scores" not in model_parameters:
            model_parameters["output_scores"] = True

        with torch.no_grad():
            generate_kwargs = {"input_ids": input_ids, **model_parameters}
            outputs = model.generate(**generate_kwargs)

        # Decode the generated tokens
        predictions = []
        for output_ids in outputs.sequences:
            prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
            predictions.append(prediction)

        if hasattr(outputs, "scores") and outputs.scores:
            logit_str = tensor_to_string(outputs.scores[0])  # shape: [1, vocab]
            logits.append(logit_str)

        reference = sample.get(reference_column, "")
        results.append(
            {
                "input": sample.get(input_column, ""),
                "prompt": prompt,
                "prediction": predictions,
                "reference": [reference] if isinstance(reference, str) else reference,
            }
        )

        # Aggressive memory cleanup after each sample
        del inputs, input_ids, outputs, predictions
        if attention_mask is not None:
            del attention_mask
        if "reference" in locals():
            del reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, logits


def run_inference_decoder(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run inference on decoder-only models (like GPT, LLaMA, DeepSeek).
    Args:
        model: The decoder-only model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.
    Returns:
        results: A list of results from the inference.
    """

    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    task = parameters.get("task_type")
    if task == "text-simplification":  # Simplification
        return run_inference_simplification(
            model, tokenizer, dataset, parameters, job_id, job_queue
        )
    else:
        raise ValueError(
            f"Unsupported encoder task: {task}. Supported tasks: {AVAILABLE_TASK_TYPES['decoder-only']}"
        )


def run_inference_simplification(
    model, tokenizer, dataset, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run inference on text simplification tasks.
    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for encoding inputs.
        dataset: The dataset to run inference on.
        parameters: Additional dictionary of parameters for inference.
    Returns:
        results: A list of results from the inference.
    """
    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    dataset_parameters = parameters.get("dataset", {})
    split = dataset_parameters.get("split", "test")
    input_column = dataset_parameters.get("input_column")
    reference_column = dataset_parameters.get("reference_column", "reference")
    num_samples = dataset_parameters.get("num_samples", None)
    num_samples = (
        int(num_samples)
        if num_samples is not None
        and isinstance(num_samples, (int, str))
        and str(num_samples).isdigit()
        else None
    )

    model_parameters = parameters.get("model", {}).get("inference", {})
    if "return_dict_in_generate" not in model_parameters:
        model_parameters["return_dict_in_generate"] = True
    if "output_scores" not in model_parameters:
        model_parameters["output_scores"] = True

    tokenizer_parameters = parameters.get("tokenizer", {}).get("inference", {})
    model_max_length = getattr(tokenizer, "model_max_length", 16384)
    if model_max_length > 1e6:  # Some tokenizers have unrealistic defaults
        model_max_length = 16384

    tokenizer_parameters = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": tokenizer_parameters.get("max_length", model_max_length),
        "padding": True,
        **tokenizer_parameters,
    }

    # == == Temporary patch == ==
    tokenizer_parameters["max_length"] = 4096

    task_parameters = parameters.get("task_parameters", {})
    system_message = task_parameters.get("system_message")
    if not system_message:
        logger.error("System message must be provided for text simplification tasks.")
        raise ValueError(
            "System message must be provided for text simplification tasks."
        )

    results = []
    logits = []
    for i, sample in enumerate(dataset[split]):
        _check_job_cancellation(job_id, job_queue)

        if num_samples is not None and i >= num_samples:
            break
        logger.info(
            f"{job_id}: Processing sample {i + 1}/{num_samples if num_samples is not None else len(dataset[split])}"
        )

        prompt = sample.get(input_column, "")
        if system_message:
            prompt = (
                system_message.replace("<input>", prompt)
                if system_message and "<input>" in system_message
                else f"{system_message} {prompt}"
            )  # Check for placeholder & replace, otherwise append
        messages = [{"role": "user", "content": f"{prompt}"}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode input
        device = next(model.parameters()).device
        inputs = tokenizer([text], **tokenizer_parameters).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **model_parameters)

        decoded_outputs = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)
        ]
        responses = tokenizer.batch_decode(decoded_outputs, skip_special_tokens=True)

        if hasattr(outputs, "scores") and outputs.scores:
            logits_str = tensor_to_string(outputs.scores[0])  # shape: [1, vocab]
            logits.append(logits_str)

        reference = sample.get(reference_column, "")
        results.append(
            {
                "input": sample.get(input_column, ""),
                "prompt": prompt,
                "prediction": responses,
                "reference": [reference] if isinstance(reference, str) else reference,
            }
        )

        # Aggressive memory cleanup after each sample
        del inputs, outputs, decoded_outputs, responses, text, messages
        if "reference" in locals():
            del reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, logits


def __compute_metric_summary(metric_key: str, values: list) -> dict:
    """
    Compute appropriate summary statistics based on metric type.
    Provides more meaningful summaries for different types of metrics.
    """
    if not values:
        return {}

    values = np.array(values)
    n_values = len(values)

    # Determine metric type and compute appropriate summary
    ranking_metrics = {
        "ndcg",
        "map",
        "mean_average_precision",
        "mrr",
        "average_precision",
    }
    text_gen_metrics = {"bleu", "rouge", "meteor", "bertscore"}
    classification_metrics = {"accuracy", "precision", "recall", "f1"}

    # Extract base metric name (remove @k suffixes)
    base_metric = metric_key.split("_at_")[0].split("@")[0].lower()

    def safe_float(x):
        if np.isnan(x):
            return "nan"
        if np.isinf(x):
            return "inf"
        if np.isneginf(x):
            return "-inf"
        return float(x)

    # Common basic statistics
    summary = {
        "mean": safe_float(np.mean(values)),
        "std": safe_float(np.std(values)),
        "count": n_values,
    }

    if base_metric in ranking_metrics:
        # Ranking metrics: Focus on distribution and performance tiers
        summary.update(
            {
                "median": safe_float(np.median(values)),
                "q25": safe_float(np.percentile(values, 25)),
                "q75": safe_float(np.percentile(values, 75)),
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                # Distribution shape
                "skewness": safe_float(__compute_skewness(values))
                if n_values >= 3
                else None,
            }
        )
    elif base_metric in text_gen_metrics:
        # Text generation metrics: Focus on consistency and outliers
        summary.update(
            {
                "median": safe_float(np.median(values)),
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                "coefficient_of_variation": safe_float(np.std(values) / np.mean(values))
                if np.mean(values) > 0
                else "inf",
                # Outlier detection
                "outlier_count": int(__count_outliers(values)),
                "outlier_rate": safe_float(__count_outliers(values) / n_values),
            }
        )
    elif base_metric in classification_metrics:
        # Classification metrics: Focus on error analysis
        summary.update(
            {
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                "median": safe_float(np.median(values)),
                # Error analysis for bounded metrics [0,1]
                "perfect_performance_count": int(np.sum(values == 1.0)),
                "perfect_performance_rate": safe_float(np.mean(values == 1.0)),
                "zero_performance_count": int(np.sum(values == 0.0)),
                "zero_performance_rate": safe_float(np.mean(values == 0.0)),
            }
        )
    else:
        # Generic metrics: Basic statistical summary
        summary.update(
            {
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                "median": safe_float(np.median(values)),
            }
        )

    return summary


def __compute_skewness(values: np.ndarray) -> float:
    """Compute skewness (measure of asymmetry) of the distribution."""
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        return 0.0
    return np.mean(((values - mean_val) / std_val) ** 3)


def __count_outliers(values: np.ndarray, method: str = "iqr") -> int:
    """Count outliers using IQR method."""
    if method == "iqr":
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        return np.sum((values < lower_bound) | (values > upper_bound))
    return 0


def run_analysis(
    metrics: list, result, parameters: dict, job_id: str = "", job_queue=None
):
    """
    Run analysis on the result using the specified metrics.
    Args:
        metrics: List of metric names to compute.
        dataset: The dataset to use for metric computation.
        result: The result object containing predictions.
        parameters: Additional dictionary of parameters for analysis.
    Returns:
        result: The result object with computed metrics added.
    """
    # Set global job id
    global JOB_ID
    JOB_ID = job_id

    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    # Check if result object contains results
    if not result.results:
        logger.warning(f"{JOB_ID}: Skipping analysis - no results found")
        return result

    # Metric parameters
    metric_parameters = parameters.get("metrics", {})

    # Track which metric produced which keys for summary attribution
    metric_key_sources = {}
    failed_metrics = set()  # Track metrics that failed

    for _, metric_name in enumerate(metrics):
        _check_job_cancellation(job_id, job_queue)

        metric = get_metric_registry().get_or_create_metric(metric_name)
        logger.info(f"{JOB_ID}: Computing metric: {metric_name}")

        c_parameters = metric_parameters.get(metric_name, {})
        try:
            computed_metrics = []
            for j, result_item in enumerate(result.results):
                # Check for cancellation every 100 result items to avoid overhead
                if j % 100 == 0:
                    _check_job_cancellation(job_id, job_queue)

                # Check if prediction data is available
                if "prediction" not in result_item:
                    logger.warning(
                        f"{JOB_ID}: Skipping metric computation for {metric_name} - no prediction data available"
                    )
                    continue

                # Check if reference data is available
                if "reference" not in result_item:
                    logger.warning(
                        f"{JOB_ID}: Skipping metric computation for {metric_name} - no reference data available"
                    )
                    continue

                # Check if metric requires cut-off
                if metric_name in ["accuracy", "precision", "recall", "f1"]:
                    cutoff = parameters.get("task_parameters", {}).get(
                        "metric_cutoff", 0.8
                    )
                    binary_predictions = []
                    for p in result_item["prediction"]:
                        if not isinstance(p, (float, int)):
                            continue  # Skip non-numeric predictions
                        binary_predictions.append(1 if p >= cutoff else 0)
                    if binary_predictions:  # Only update if we have valid predictions
                        result_item["prediction"] = binary_predictions

                # Check if metric requires logits
                if metric_name in ["median_rank_correct_word"]:
                    logit_id = result_item.get("logit_id", None)
                    from app.controllers.logit_controller import LogitController

                    c_parameters["logits"] = (
                        [
                            LogitController()
                            .get_logit_by_id(logit_id)
                            .get("logit_str", "")
                        ]
                        if logit_id
                        else []
                    )
                    c_parameters["model_name"] = (
                        result.method().test().project().model_name
                    )
                    c_parameters["tokenizer_parameters"] = parameters.get(
                        "tokenizer", {}
                    ).get("from_pretrained", {})

                if metric_name.lower() in ["sari"]:
                    if result_item["reference"] and not isinstance(
                        result_item["reference"][0], list
                    ):
                        result_item["reference"] = [result_item["reference"]]

                    # SARI metric expects dataset access
                    input_text = result_item.get("input", "")
                    if not input_text:
                        logger.error(
                            f"{JOB_ID}: Skipping SARI metric for item {j} - no input text found"
                        )
                        failed_metrics.add(metric_name)
                        continue

                    c_parameters["dataset"] = result.method().dataset_name
                    c_parameters["split"] = parameters.get("dataset", {}).get(
                        "split", "train"
                    )
                    c_parameters["input_column"] = parameters.get("dataset", {}).get(
                        "input_column", "input"
                    )
                    c_parameters["sources_column"] = (
                        parameters.get("metrics", {})
                        .get(metric_name.lower(), {})
                        .get("sources_column", "source")
                    )
                    c_parameters["inputs"] = [input_text]

                # Check if either prediction or reference are a float, int, or string; if so, put them in an list
                if isinstance(result_item["prediction"], (float, int, str)):
                    result_item["prediction"] = [result_item["prediction"]]
                if isinstance(result_item["reference"], (float, int, str)):
                    result_item["reference"] = [result_item["reference"]]

                try:
                    computed_metric = metric.compute(
                        predictions=result_item["prediction"],
                        references=result_item["reference"],
                        **c_parameters,
                    )

                    # Check if metric returned error information (e.g., readability metric)
                    if "errors" in computed_metric:
                        error_info = computed_metric["errors"]
                        failed_sub_metrics = error_info.get("failed_metrics", [])
                        error_details = error_info.get("error_details", {})

                        # Add failed sub-metrics to failed_metrics list with parent metric name
                        for sub_metric in failed_sub_metrics:
                            failed_metric_key = f"{metric_name}.{sub_metric}"
                            failed_metrics.add(failed_metric_key)

                            # Extract specific error message for this sub-metric
                            specific_error = "Unknown error"

                            # Try the current item's text key first
                            text_key = f"text_{j}"
                            if (
                                text_key in error_details
                                and sub_metric in error_details[text_key]
                            ):
                                specific_error = error_details[text_key][sub_metric]
                            # If not found, try text_0 (for single-item metric calls)
                            elif (
                                "text_0" in error_details
                                and sub_metric in error_details["text_0"]
                            ):
                                specific_error = error_details["text_0"][sub_metric]

                            logger.warning(
                                f"{JOB_ID}: Sub-metric {sub_metric} of {metric_name} failed for item {j}: {specific_error}"
                            )

                        # Remove error information before processing results
                        computed_metric = {
                            k: v for k, v in computed_metric.items() if k != "errors"
                        }

                except Exception as e:
                    logger.error(
                        f"{JOB_ID}: Error computing {metric_name} metric for item {j}: {e}"
                    )
                    failed_metrics.add(metric_name)
                    continue

                computed_metrics.append(computed_metric)

            merged_metrics = {}
            if not computed_metrics:
                continue

            # Get the metric name (outer key)
            metric_name_key = next(iter(computed_metrics[0].keys()))

            # Unwrap the outer layer to get the actual metric values
            metric_values = [metric[metric_name_key] for metric in computed_metrics]

            # Check if we have nested structure (dict) or flat structure (primitive/list)
            if metric_values and isinstance(metric_values[0], dict):
                for inner_key in metric_values[0]:
                    merged_metrics[inner_key] = [
                        val[inner_key] for val in metric_values
                    ]
            else:
                merged_metrics[metric_name_key] = metric_values

            parsed_metrics = metric.parse_output(merged_metrics, len(result.results))
        except Exception as e:
            logger.error(f"{JOB_ID}: Error computing {metric_name} metric: {e}")
            continue

        for item, item_metrics in zip(result.results, parsed_metrics):
            if "metrics" not in item:
                item["metrics"] = {}

            for key, value in item_metrics.items():
                item["metrics"][key] = value
                # Track which metric produced this key
                metric_key_sources[key] = metric_name

        logger.info(f"{JOB_ID}: Finished computing metric: {metric_name}")

    logger.info(f"{JOB_ID}: Finished computing metrics; generating summary statistics.")

    # Create summary statistics from all computed metrics
    summary = {}
    if result.results and any("metrics" in item for item in result.results):
        metric_keys = set()
        for item in result.results:
            if "metrics" in item:
                metric_keys.update(item["metrics"].keys())

        # Compute summary statistics for each metric
        for metric_key in metric_keys:
            values = []
            for item in result.results:
                if "metrics" in item and metric_key in item["metrics"]:
                    value = item["metrics"][metric_key]
                    # Only include numeric values for summary statistics
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, list):
                        values.extend([v for v in value if isinstance(v, (int, float))])

            if values:
                metric_summary = __compute_metric_summary(metric_key, values)
                # Add source metric attribution
                if metric_key in metric_key_sources:
                    metric_summary["source_metric"] = metric_key_sources[metric_key]
                summary[metric_key] = metric_summary
    result.summary = summary

    # Update job status to WARNING if any metrics failed
    if failed_metrics:
        from app.controllers.job_controller import JobController

        failed_metrics_str = ", ".join(failed_metrics)
        warning_message = f"Job completed in TIME_PLACEHOLDER seconds with warnings: {failed_metrics_str} metric(s) failed. DIAGNOSTICS_PLACEHOLDER. Check logs for details."
        JobController.update_job_status(
            job_id=job_id, status="WARNING", status_message=warning_message
        )

    return result


def run_diagnostics(
    diagnostics: list,
    result,
    parameters: dict,
    job_id: str = "",
    job_queue=None,
    model_name: str = "",
):
    """
    Run diagnostics on the result using the specified diagnostics.
    Args:
        diagnostics: List of diagnostic names to compute.
        result: The result object containing predictions.
        parameters: Additional dictionary of parameters for diagnostics.
    Returns:
        result: The result object with computed diagnostics added.
    """
    # Set global job id
    global JOB_ID
    JOB_ID = job_id

    # Check for cancellation at the start
    _check_job_cancellation(job_id, job_queue)

    # Check if result object contains results
    passed_results = result.results if result and hasattr(result, "results") else None

    # Diagnostic parameters
    d_parameters = parameters.get("diagnostics", {})

    # Model parameters
    model_parameters = parameters.get("model", {}).get("from_pretrained", {})
    tokenizer_parameters = parameters.get("tokenizer", {}).get("from_pretrained", {})

    # Dataset parameters
    dataset_parameters = parameters.get("dataset", {})

    # Check diagnostic_only parameters
    diagnostic_only = parameters.get("task_parameters", {}).get(
        "diagnostic_only", False
    )
    diagnostic_type = parameters.get("task_parameters", {}).get(
        "diagnostic_type", "lxt"
    )
    diagnostic_sample = (
        parameters.get("task_parameters", {})
        .get("diagnostic_sample", {})
        .get("sample", None)
    )

    if diagnostic_only:
        diagnostics = [diagnostic_type]

    failed_diagnostics = set()  # Track diagnostics that failed

    for diagnostic_name in diagnostics:
        _check_job_cancellation(job_id, job_queue)

        diagnostic = get_diagnostic_registry().get_diagnostic(diagnostic_name)
        c_parameters = d_parameters.get(diagnostic_name, {})
        c_parameters["model_name"] = model_name
        c_parameters["model_parameters"] = model_parameters
        c_parameters["tokenizer_parameters"] = tokenizer_parameters
        c_parameters["dataset_name"] = result.method().dataset_name
        c_parameters["dataset"] = dataset_parameters
        c_parameters["diagnostic_sample"] = diagnostic_sample
        c_parameters["task_parameters"] = parameters.get("task_parameters", {})

        try:
            logger.info(f"{JOB_ID}: Executing {diagnostic_name} diagnostic")
            computed_diagnostic = diagnostic.compute(
                results=passed_results,
                parameters=c_parameters,
                job_id=job_id,
                job_queue=job_queue,
            )
        except Exception as e:
            logger.error(f"{JOB_ID}: Error executing {diagnostic_name} diagnostic: {e}")
            failed_diagnostics.add(diagnostic_name)
            continue

        result.diagnostics[diagnostic_name] = computed_diagnostic

        logger.info(f"{JOB_ID}: Finished {diagnostic_name} diagnostic.")

    # Update job status to WARNING if any diagnostics failed
    if failed_diagnostics:
        from app.controllers.job_controller import JobController

        failed_diagnostics_str = ", ".join(failed_diagnostics)

        # Check if there's already a WARNING status with metrics failures
        current_job = JobController.get_job_by_id(job_id, include=set())
        current_status = current_job.get("status") if current_job else None
        current_message = current_job.get("status_message", "") if current_job else ""

        if current_status == "WARNING" and "DIAGNOSTICS_PLACEHOLDER" in current_message:
            warning_message = current_message.replace(
                "DIAGNOSTICS_PLACEHOLDER",
                f"{failed_diagnostics_str} diagnostic(s) failed",
            )
        else:
            warning_message = f"Job completed in TIME_PLACEHOLDER seconds with warnings: {failed_diagnostics_str} diagnostic(s) failed. Check logs for details."

        JobController.update_job_status(
            job_id=job_id, status="WARNING", status_message=warning_message
        )

    return result
