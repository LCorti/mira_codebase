from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric


class CustomMetric(Metric):
    def __init__(
        self, metric_name: str, description: str | None = None, **metric_kwargs
    ):
        """
        Initialize a custom metric by loading any Hugging Face evaluate metric.

        :param metric_name: The name of the Hugging Face metric to load (e.g., "bleu", "meteor", "bertscore")
        :param description: Optional custom description. If None, uses a default description.
        :param metric_kwargs: Additional keyword arguments to pass to evaluate.load()
        :raises RuntimeError: If the metric cannot be loaded.

        :Example:
        >>> from app.models.metrics.custom_metric import CustomMetric
        >>> bleu_metric = CustomMetric("bleu", "BLEU score for machine translation")
        >>> blue_score = bleu_metric.compute(predictions=["hello world"], references=["hello world"])
        >>> MetricRegistry.register("custom_bleu", bleu_metric)
        """
        # Use provided description or create a default one
        if description is None:
            description = (
                f"Custom {metric_name} metric loaded from Hugging Face evaluate"
            )

        super().__init__(name=metric_name, description=description)

        try:
            self.metric = evaluate.load(metric_name, **metric_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Hugging Face metric '{metric_name}': {e!s}"
            ) from e

    def get_metric_info(self) -> dict[str, Any]:
        """
        Get information about the loaded metric.

        :return: Dictionary containing metric information
        """
        try:
            return {
                "name": self.name,
                "description": self.description,
                "features": getattr(self.metric, "features", None),
                "inputs_description": getattr(self.metric, "inputs_description", None),
                "citation": getattr(self.metric, "citation", None),
                "homepage": getattr(self.metric, "homepage", None),
            }
        except Exception:
            return {"name": self.name, "description": self.description}
