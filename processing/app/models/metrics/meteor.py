from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric


class MeteorMetric(Metric):
    def __init__(self):
        """
        Initialize the MeteorMetric.
        """
        super().__init__(
            name="meteor",
            description="Computes the METEOR metric for model predictions.",
        )
        self.metric = evaluate.load("meteor")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        alpha: float = 0.9,
        beta: int = 3,
        gamma: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the METEOR metric with native parameter support.
        See more details about METEOR at: https://huggingface.co/spaces/evaluate-metric/meteor

        :param predictions: List of predictions to score. Each prediction should be a string with tokens separated by spaces.
        :type predictions: List[str]
        :param references: List of reference for each prediction or a list of several references per prediction.
        :type references: Union[List[str], List[List[str]]]
        :param alpha: Parameter for controlling relative weights of precision and recall. Defaults to 0.9.
        :type alpha: float, optional
        :param beta: Parameter for controlling shape of penalty as a function of fragmentation. Defaults to 3.
        :type beta: int, optional
        :param gamma: The relative weight assigned to fragmentation penalty. Defaults to 0.5.
        :type gamma: float, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed METEOR metric value.
        :rtype: Union[dict[str, float], dict[str, List[float]]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Validate alpha parameter
        if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
            raise ValueError("alpha must be a float between 0 and 1.")

        # Validate beta parameter
        if not isinstance(beta, (int, float)) or beta < 0:
            raise ValueError("beta must be a non-negative number.")

        # Validate gamma parameter
        if not isinstance(gamma, (int, float)) or gamma < 0 or gamma > 1:
            raise ValueError("gamma must be a float between 0 and 1.")

        try:
            return self.metric.compute(
                predictions=predictions,
                references=references,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
