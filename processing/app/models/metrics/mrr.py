from typing import Any

import numpy as np

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class MRRMetric(Metric):
    def __init__(self):
        """
        Initialize the MRRMetric.
        Mean Reciprocal Rank (MRR) measures the rank of the first relevant item in the ranking.
        MRR@k evaluates only the top k results. The reciprocal rank is 1/rank for the first relevant item.
        """
        super().__init__(
            name="mrr",
            description="Computes the Mean Reciprocal Rank (MRR) metric for ranking tasks.",
        )
        # Note: We don't use evaluate.load() here since MRR is not available in HuggingFace evaluate
        self.metric = None

    def compute(
        self,
        predictions: list[list[float]] | list[float],
        references: list[list[int | bool]] | list[int | bool],
        k: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute the Mean Reciprocal Rank (MRR) metric.

        For single query:
        - predictions should be a 1D array of relevance scores (higher = more relevant)
        - references should be a 1D array of binary relevance (1=relevant, 0=not relevant)
        - Example: predictions = [0.9, 0.8, 0.7], references = [0, 1, 0] → RR = 1/2 = 0.5

        For multiple queries:
        - predictions should be a 2D array where each row is the scores for one query's results
        - references should be a 2D array where each row is the binary relevance for one query
        - Example: predictions = [[0.9, 0.8], [0.7, 0.6]], references = [[0, 1], [1, 0]]
        - Query 1: First relevant at rank 2 → RR = 1/2 = 0.5
        - Query 2: First relevant at rank 1 → RR = 1/1 = 1.0
        - MRR = (0.5 + 1.0) / 2 = 0.75

        :param predictions: Prediction scores for ranking (will be sorted in descending order).
        :type predictions: Union[List[List[float]], List[float]]
        :param references: Binary relevance indicators (1=relevant, 0=not relevant).
        :type references: Union[List[List[Union[int, bool]]], List[Union[int, bool]]]
        :param k: Evaluate MRR at rank k (MRR@k). If None, uses all items.
        :type k: Optional[int], optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed MRR score.
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")

        # Convert to numpy arrays
        y_score = np.array(predictions)
        y_true = np.array(references)

        # Handle 1D case (single query)
        if y_score.ndim == 1:
            if y_true.ndim != 1:
                logger.error("For single query, references should be 1D")
                raise ValueError("For single query, references should be 1D")
            if len(y_score) != len(y_true):
                logger.error("Predictions and references must have the same length.")
                raise ValueError(
                    "Predictions and references must have the same length."
                )

            # Convert to 2D for consistent processing
            y_score = y_score.reshape(1, -1)
            y_true = y_true.reshape(1, -1)

        elif y_score.ndim == 2:
            # Multi-query case
            if y_true.ndim != 2:
                logger.error("For multiple queries, references should be 2D")
                raise ValueError("For multiple queries, references should be 2D")
            if y_score.shape != y_true.shape:
                logger.error("Predictions and references must have the same shape.")
                raise ValueError("Predictions and references must have the same shape.")

        else:
            raise ValueError(
                "Predictions must be 1D (single query) or 2D (multiple queries)"
            )

        # Validate k parameter
        if k is not None:
            if k <= 0:
                logger.error("k must be a positive integer")
                raise ValueError("k must be a positive integer")
            if k > y_score.shape[1]:
                k = y_score.shape[1]  # Adjust k to max available results
                logger.warning(
                    f"k is larger than the number of items, adjusting k to length of predictions: {k}"
                )

        # Validate that references are binary
        unique_values = np.unique(y_true)
        if not all(val in [0, 1, True, False] for val in unique_values):
            logger.error(
                f"References must be binary (0/1 or True/False). Found values: {unique_values}"
            )
            raise ValueError(
                f"References must be binary (0/1 or True/False). Found values: {unique_values}"
            )

        # Convert boolean to int if needed
        y_true = y_true.astype(int)

        try:
            reciprocal_ranks = []

            for query_idx in range(y_score.shape[0]):
                query_scores = y_score[query_idx]
                query_relevance = y_true[query_idx]

                # Sort by scores in descending order (highest scores first)
                sorted_indices = np.argsort(query_scores)[::-1]

                # Apply k cutoff if specified
                if k is not None:
                    sorted_indices = sorted_indices[:k]

                # Find the rank of the first relevant item (1-indexed)
                reciprocal_rank = 0.0
                for rank, doc_idx in enumerate(sorted_indices, 1):
                    if query_relevance[doc_idx] == 1:
                        reciprocal_rank = 1.0 / rank
                        break

                reciprocal_ranks.append(reciprocal_rank)

            # Compute MRR as the mean of reciprocal ranks
            mrr_value = np.mean(reciprocal_ranks)

            # Create appropriate key name
            if k is not None:
                key_name = f"mrr_at_{k}"
            else:
                key_name = "mrr"

            return {key_name: float(mrr_value)}

        except Exception as e:
            logger.error(f"Failed to compute MRR metric: {e!s}")
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
