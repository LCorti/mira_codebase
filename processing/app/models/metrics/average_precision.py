from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class AveragePrecisionMetric(Metric):
    def __init__(self):
        """
        Initialize the AveragePrecisionMetric.
        Average Precision (AP) summarizes a precision-recall curve as the weighted mean of precisions
        achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
        """
        super().__init__(
            name="average_precision",
            description="Computes the Average Precision (AP) metric for binary classification or ranking tasks.",
        )
        # Note: We don't use evaluate.load() here since AP is not available in HuggingFace evaluate
        self.metric = None

    def compute(
        self,
        predictions: list[float | int],
        references: list[int | bool],
        pos_label: int | str = 1,
        sample_weight: list[float] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute the Average Precision metric using scikit-learn's implementation.

        :param predictions: Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
        :type predictions: list[Union[float, int]]
        :param references: True binary labels or binary label indicators.
        :type references: list[Union[int, bool]]
        :param pos_label: The label of the positive class. Defaults to 1.
        :type pos_label: Union[int, str], optional
        :param sample_weight: Sample weights. Defaults to None.
        :type sample_weight: Optional[list[float]], optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed Average Precision score.
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            logger.error("Predictions and references must have the same length.")
            raise ValueError("Predictions and references must have the same length.")

        # Convert to numpy arrays for scikit-learn
        y_true = np.array(references)
        y_score = np.array(predictions)

        # Validate that references are binary
        unique_labels = np.unique(y_true)
        if len(unique_labels) > 2:
            logger.error(
                f"Average Precision is designed for binary classification. Found {len(unique_labels)} unique labels: {unique_labels}"
            )
            raise ValueError(
                f"Average Precision is designed for binary classification. Found {len(unique_labels)} unique labels: {unique_labels}"
            )

        # Handle pos_label - convert references to binary if needed
        if pos_label != 1 and pos_label in unique_labels:
            y_true = (y_true == pos_label).astype(int)
        elif pos_label not in [0, 1] and pos_label not in unique_labels:
            raise ValueError(f"pos_label {pos_label} is not present in references")

        # Validate sample_weight if provided
        if sample_weight is not None:
            if len(sample_weight) != len(predictions):
                logger.error(
                    "Sample weights must have the same length as predictions and references."
                )
                raise ValueError(
                    "Sample weights must have the same length as predictions and references."
                )
            if any(w < 0 for w in sample_weight):
                logger.error("Sample weights must be non-negative.")
                raise ValueError("Sample weights must be non-negative.")
            sample_weight = np.array(sample_weight)

        try:
            # Compute Average Precision using scikit-learn
            ap_score = average_precision_score(
                y_true=y_true,
                y_score=y_score,
                pos_label=1,  # We've already handled pos_label conversion above
                sample_weight=sample_weight,
            )

            return {"average_precision": float(ap_score)}

        except Exception as e:
            logger.error(f"Error computing Average Precision: {e!s}")
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
