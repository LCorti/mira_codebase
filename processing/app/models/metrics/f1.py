from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class F1Metric(Metric):
    def __init__(self):
        """
        Initialize the F1Metric.
        """
        super().__init__(
            name="f1", description="Computes the F1 score metric for model predictions."
        )
        self.metric = evaluate.load("f1")

    def compute(
        self,
        predictions: list[str | int],
        references: list[str | int],
        labels: list[int] | None = None,
        pos_label: int = 1,
        average: str = "weighted",
        sample_weight: list[float] | None = None,
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the F1 score metric with native parameter support.
        See more details about F1 at: https://huggingface.co/spaces/evaluate-metric/f1

        :param predictions: Predicted labels.
        :type predictions: list[Union[str, int]]
        :param references: Ground truth labels.
        :type references: list[Union[str, int]]
        :param labels: The set of labels to include when average is not set to 'binary'. Defaults to None.
        :type labels: Optional[list[int]], optional
        :param pos_label: The class to be considered the positive class, in the case where average is set to binary. Defaults to 1.
        :type pos_label: int, optional
        :param average: This parameter is required for multiclass/multilabel targets. Defaults to 'weighted'.
        :type average: str, optional
        :param sample_weight: Sample weights. Defaults to None.
        :type sample_weight: Optional[list[float]], optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed F1 score metric values.
        :rtype: Union[dict[str, float], dict[str, list[float]]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Validate average parameter
        valid_averages = ["binary", "micro", "macro", "weighted", "samples", None]
        if average not in valid_averages:
            raise ValueError(
                f"Invalid average: {average}. Valid averages are: {valid_averages}"
            )

        # Validate sample_weight if provided
        if sample_weight is not None:
            if len(sample_weight) != len(predictions):
                raise ValueError(
                    "Sample weights must have the same length as predictions and references."
                )
            if any(w < 0 for w in sample_weight):
                raise ValueError("Sample weights must be non-negative.")

        try:
            # Prepare arguments for the metric computation
            compute_args = {
                "predictions": predictions,
                "references": references,
                "pos_label": pos_label,
                "average": average,
                **kwargs,
            }

            # Add optional parameters only if provided
            if labels is not None:
                compute_args["labels"] = labels
            if sample_weight is not None:
                compute_args["sample_weight"] = sample_weight

            result = self.metric.compute(**compute_args)
            return {"f1": result}
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
