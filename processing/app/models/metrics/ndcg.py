from typing import Any

import numpy as np
from sklearn.metrics import ndcg_score

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class NDCGMetric(Metric):
    def __init__(self):
        """
        Initialize the NDCGMetric.
        Normalized Discounted Cumulative Gain (NDCG) measures the quality of ranking by considering
        both the relevance of items and their positions in the ranking. It supports graded relevance scores.
        """
        super().__init__(
            name="ndcg",
            description="Computes the Normalized Discounted Cumulative Gain (NDCG) metric for ranking tasks.",
        )
        # Note: We don't use evaluate.load() here since NDCG is not available in HuggingFace evaluate
        self.metric = None

    def compute(
        self,
        predictions: list[list[float]] | list[float],
        references: list[list[int | float]] | list[int | float],
        k: int | None = None,
        sample_weight: list[float] | None = None,
        ignore_ties: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute the NDCG metric using scikit-learn's implementation.

        For single query:
        - predictions should be a 1D array of relevance scores (higher = more relevant)
        - references should be a 1D array of true relevance grades (can be binary or graded)
        - Example: predictions = [0.9, 0.8, 0.7], references = [2, 0, 1] (graded: 2=highly relevant, 1=relevant, 0=not relevant)

        For multiple queries:
        - predictions should be a 2D array where each row is the scores for one query's results
        - references should be a 2D array where each row is the relevance grades for one query
        - Example: predictions = [[0.9, 0.8], [0.7, 0.6]], references = [[2, 0], [1, 1]]

        :param predictions: Prediction scores for ranking.
        :type predictions: Union[List[List[float]], List[float]]
        :param references: True relevance grades (can be binary 0/1 or graded 0/1/2/3...).
        :type references: Union[List[List[Union[int, float]]], List[Union[int, float]]]
        :param k: Evaluate NDCG at rank k (NDCG@k). If None, uses all items.
        :type k: Optional[int], optional
        :param sample_weight: Sample weights. Defaults to None.
        :type sample_weight: Optional[List[float]], optional
        :param ignore_ties: Whether to ignore ties in ranking. Defaults to False.
        :type ignore_ties: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed NDCG score.
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")

        # Convert to numpy arrays
        y_score = np.array(predictions)
        y_true = np.array(references)

        if y_score.ndim == 1:  # Handle 1D case (single query)
            if y_true.ndim != 1:
                logger.error("For single query, references should be 1D")
                raise ValueError("For single query, references should be 1D")
            if len(y_score) != len(y_true):
                logger.error("Predictions and references must have the same length.")
                raise ValueError(
                    "Predictions and references must have the same length."
                )

            # Convert to 2D for sklearn (expects 2D input)
            y_score = y_score.reshape(1, -1)
            y_true = y_true.reshape(1, -1)
        elif y_score.ndim == 2:  # Multi-query case
            if y_true.ndim != 2:
                logger.error("For multiple queries, references should be 2D")
                raise ValueError("For multiple queries, references should be 2D")
            if y_score.shape != y_true.shape:
                logger.error("Predictions and references must have the same shape.")
                raise ValueError("Predictions and references must have the same shape.")
        else:
            logger.error(
                "Predictions must be 1D (single query) or 2D (multiple queries)"
            )
            raise ValueError(
                "Predictions must be 1D (single query) or 2D (multiple queries)"
            )

        # Validate sample_weight if provided
        if sample_weight is not None:
            if len(sample_weight) != y_score.shape[0]:
                raise ValueError(
                    "Sample weights must have the same length as the number of queries."
                )
            if any(w < 0 for w in sample_weight):
                raise ValueError("Sample weights must be non-negative.")
            sample_weight = np.array(sample_weight)

        # Validate k parameter
        if k is not None:
            if k <= 0:
                raise ValueError("k must be a positive integer")
            if k > y_score.shape[1]:
                k = y_score.shape[1]  # Adjust k to max available items
                logger.warning(
                    f"k is larger than the number of items, adjusting k to length of predictions: {k}"
                )

        try:
            # Compute NDCG using scikit-learn
            ndcg_value = ndcg_score(
                y_true=y_true,
                y_score=y_score,
                k=k,
                sample_weight=sample_weight,
                ignore_ties=ignore_ties,
            )

            # Create appropriate key name
            if k is not None:
                key_name = f"ndcg_at_{k}"
            else:
                key_name = "ndcg"

            return {key_name: float(ndcg_value)}

        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
