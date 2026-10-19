from typing import Any

import torch

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService
from app.services.tensor_parser import string_to_tensor

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class MRCWMetric(Metric):
    def __init__(self):
        """
        Initialize the Median Rank of the Correct Word metric.
        """
        super().__init__(
            name="median_rank_correct_word",
            description="Computes the median position of the true word when all vocabulary candidates are ranked by the model’s predicted probability.",
        )
        # Note: We don't use evaluate.load() here since MRCW is not a standard metric
        self.metric = None

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        logits: list[str],
        **kwargs: Any,
    ) -> dict:
        """
        Compute the Median Rank of the Correct Word metric for the given predictions and references.

        :param predictions: List of input text, where each separate text snippet is one list entry.
        :type predictions: List[str]
        :param references: List of reference text, where each separate text snippet is one list entry.
        :type references: List[str]
        :param logits: The raw output logits from the model (encoded as string).
        :type logits: List[str]
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        :type kwargs: Any
        """

        # Validate inputs
        if not predictions:
            raise ValueError("Predictions must not be empty.")
        if not references:
            raise ValueError("References must not be empty.")
        if not logits:
            raise ValueError("Logits must not be empty.")
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have the same length; got {len(predictions)} and {len(references)}."
            )
        if len(predictions) != len(logits):
            raise ValueError(
                f"Predictions and logits must have the same length; got {len(predictions)} and {len(logits)}."
            )

        # Try to get tokenizer from kwargs or registry
        from app.config.registry import get_model_loader

        try:
            _, tokenizer = get_model_loader().get_model_tokenizer()
        except Exception:
            logger.warning(
                "Could not get tokenizer from model registry; trying to get from parameters."
            )
            try:
                _, tokenizer = get_model_loader().load(
                    model_name=kwargs.get("model_name", ""),
                    tokenizer_parameters=kwargs.get("tokenizer_parameters", {}),
                    cache=False,
                )
            except Exception:
                tokenizer = None
                logger.error("Could not load tokenizer from parameters.")

        if tokenizer is None:
            logger.error("Tokenizer is not available for MRCW metric computation.")
            raise ValueError(
                "Tokenizer must be provided either as parameter or available in model registry for MRCW metric computation."
            )

        all_ranks = []

        for _, reference, logits_str in zip(predictions, references, logits):
            if not logits_str:
                logger.warning("Logits string is empty or None; skipping this sample.")
                continue

            logits_tensor = None
            if isinstance(logits_str, str):
                logger.debug("Logits is a string; attempting to parse.")
                logits_tensor = string_to_tensor(logits_str)

            if logits_tensor is None:
                logger.warning(
                    f"Could not parse logits string into tensor: {logits_str[:30]}..."
                )
                continue

            if len(logits_tensor.shape) != 2:
                logger.warning(
                    f"Logits tensor has invalid shape: {logits_tensor.shape}; expected 2D tensor."
                )
                continue

            if isinstance(reference, list):
                logger.debug("Reference is a list; taking the first element.")
                reference = reference[0]

            try:
                gold_first_token_id = tokenizer.encode(
                    " " + reference, add_special_tokens=False
                )[0]
            except Exception:
                logger.warning(f"Could not encode reference text: {reference}")
                continue

            # Rank = 1 + #tokens with strictly higher logit than gold
            gold_logit = logits_tensor[0, gold_first_token_id]
            generate_rank = (
                1 + torch.count_nonzero(logits_tensor[0] > gold_logit).item()
            )

            all_ranks.append(generate_rank)

        return {"median_rank_correct_word": all_ranks}
