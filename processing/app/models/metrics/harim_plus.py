from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class HarimPlusMetric(Metric):
    def __init__(self):
        """
        Initialize the HarimPlusMetric.

        :param pretrained_name: Summarization model checkpoint or path, loaded by transformers.AutoModelForSeq2SeqLM.from_pretrained(). Defaults to "facebook/bart-large-cnn".
        :type pretrained_name: str
        :param tokenizer: Tokenizer function compatible with transformers.PreTrainedTokenizer. Use when your tokenizer cannot be loaded by from_pretrained.
        :type tokenizer: Optional tokenizer object
        """
        super().__init__(
            name="harim_plus",
            description="HaRiM+ is a reference-less evaluation metric for summarization which leverages the power of summarization model.",
        )
        self.metric = evaluate.load("NCSOFT/harim_plus")

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        use_aggregator: bool = False,
        bsz: int = 32,
        return_details: bool = False,
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the HaRiM+ metric with native parameter support.

        Note: For HaRiM+, predictions are summaries and references are the source articles.
        Higher scores indicate better quality.

        :param predictions: List of generated summaries to evaluate.
        :type predictions: List[str]
        :param references: List of source articles that were summarized.
        :type references: List[str]
        :param use_aggregator: If True, average of the scores are returned. Defaults to False.
        :type use_aggregator: bool, optional
        :param bsz: Batch size for harim to iterate through the given pairs. Defaults to 32.
        :type bsz: int, optional
        :param return_details: Whether to show more than harim+ score (returns logppl, harim term). Defaults to False.
        :type return_details: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed HaRiM+ metric values. Contains 'harim+' key with scores.
        :rtype: dict[str, float] OR dict[str, List[float]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Validate that all predictions and references are strings
        if not all(isinstance(pred, str) for pred in predictions):
            raise ValueError("All predictions must be strings (summaries).")
        if not all(isinstance(ref, str) for ref in references):
            raise ValueError("All references must be strings (source articles).")

        try:
            result = self.metric.compute(
                predictions=predictions,
                references=references,
                use_aggregator=use_aggregator,
                bsz=bsz,
                return_details=return_details,
            )

            if isinstance(result, list):
                return {"harim+": result}

            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
