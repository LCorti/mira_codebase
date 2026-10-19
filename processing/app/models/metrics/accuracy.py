from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__(
            name="accuracy", description="Computes the accuracy of model predictions."
        )
        self.metric = evaluate.load("accuracy")

    def compute(
        self,
        predictions: list[str | int],
        references: list[str | int],
        normalize: bool = True,
        sample_weight: list[float] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute the accuracy metric with native parameter support.

        :param predictions: The predicted labels.
        :type predictions: List[Union[str, int]]
        :param references: The true labels.
        :type references: List[Union[str, int]]
        :param normalize: If True, return the fraction of correctly classified samples. If False, return the number of correctly classified samples. Defaults to True.
        :type normalize: bool, optional
        :param sample_weight: Sample weights for weighted accuracy calculation. Defaults to None.
        :type sample_weight: Optional[List[float]], optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed accuracy metric value (always aggregated).
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            logger.error("Predictions and references must have the same length.")
            raise ValueError("Predictions and references must have the same length.")

        # Check if predictions or references are string, if so, perform exact match
        if any(isinstance(p, str) for p in predictions) and any(
            isinstance(r, str) for r in references
        ):
            # Perform exact match for string predictions and references

            predictions = [p.lower() for p in predictions]
            references = [r.lower() for r in references]

            matches = [pred == ref for pred, ref in zip(predictions, references)]
            return {"accuracy": sum(matches) / len(matches) if matches else 0.0}
        else:
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

            # Prepare arguments for the metric computation
            compute_args = {
                "predictions": predictions,
                "references": references,
                "normalize": normalize,
                **kwargs,
            }

            # Add sample_weight only if provided
            if sample_weight is not None:
                compute_args["sample_weight"] = sample_weight

            result = {"accuracy": -1.0}  # Default value in case of failure
            try:
                result = self.metric.compute(**compute_args)
            except Exception as e:
                logger.error(f"Failed to compute {self.name} metric: {e!s}")
                raise RuntimeError(
                    f"Failed to compute {self.name} metric: {e!s}"
                ) from e

            return result
