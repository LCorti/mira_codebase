from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric


class RougeMetric(Metric):
    def __init__(self):
        """
        Initialize the RougeMetric with specified ROUGE types.
        """
        super().__init__(
            name="rouge", description="Computes the ROUGE metric for model predictions."
        )
        self.metric = evaluate.load("rouge")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        rouge_types: list[str] | None = None,
        use_aggregator: bool = True,
        use_stemmer: bool = False,
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the ROUGE metric with native parameter support.
        See more details about ROUGE at: https://huggingface.co/spaces/evaluate-metric/rouge

        :param predictions: List of predictions to score. Each prediction should be a string with tokens separated by spaces.
        :type predictions: list[str]
        :param references: List of reference for each prediction or a list of several references per prediction.
        :type references: Union[list[str], list[list[str]]]
        :param rouge_types: A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
        :type rouge_types: list[str], optional
        :param use_aggregator: If True, returns aggregates. Defaults to True.
        :type use_aggregator: bool, optional
        :param use_stemmer: If True, uses Porter stemmer to strip word suffixes. Defaults to False.
        :type use_stemmer: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed ROUGE metric values. Aggregated if use_aggregator=True, per-sample if use_aggregator=False.
        :rtype: Union[dict[str, float], dict[str, list[float]]]
        """
        # Set default rouge_types if not provided
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Validate rouge_types
        valid_rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        invalid_types = [t for t in rouge_types if t not in valid_rouge_types]
        if invalid_types:
            raise ValueError(
                f"Invalid rouge types: {invalid_types}. Valid types are: {valid_rouge_types}"
            )

        try:
            result = self.metric.compute(
                predictions=predictions,
                references=references,
                rouge_types=rouge_types,
                use_aggregator=use_aggregator,
                use_stemmer=use_stemmer,
                **kwargs,
            )
            return {"rouge": result}
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
