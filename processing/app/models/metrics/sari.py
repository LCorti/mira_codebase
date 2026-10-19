from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class SariMetric(Metric):
    def __init__(self):
        """
        Initialize the SariMetric
        """
        super().__init__(
            name="sari", description="Computes the SARI metric for model predictions."
        )
        self.metric = evaluate.load("sari")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[int] | list[list[str]],
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the SARI metric with native parameter support.
        See more details about SARI at: https://huggingface.co/spaces/evaluate-metric/sari

        :param predictions: List of predictions to score. Each prediction should be a string with tokens separated by spaces.
        :type predictions: list[str]
        :param references: List of reference for each prediction or a list of several references per prediction.
        :type references: list[list[str]]
        :param kwargs: Additional keyword arguments for metric computation. Expected keys:
            - sources_column: Column name in the dataset that contains the source texts
            - dataset: Name of the HuggingFace dataset
            - split: Split of the dataset (e.g., 'train', 'test', 'validation')
            - input_column: Column name that contains the input texts for matching
            - inputs: List of input texts to match against the dataset
        :type kwargs: Any

        :return: Computed SARI metric values.
        :rtype: Union[dict[str, float], dict[str, list[float]]]
        """
        # Extract sources_column from kwargs
        sources_column = kwargs.get("sources_column", "")

        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            logger.error("Predictions and references length mismatch.")
            raise ValueError("Predictions and references must have the same length.")
        if not sources_column:
            logger.error(
                "Sources column must be provided in kwargs for loading sources."
            )
            raise ValueError("Sources column must be provided.")
        if (
            not kwargs.get("dataset")
            or not kwargs.get("split")
            or not kwargs.get("input_column")
        ):
            logger.error(
                "Dataset name, split, and input column must be provided in kwargs for loading sources."
            )
            raise ValueError(
                "Dataset name, split, and input column must be provided in kwargs for loading sources."
            )
        if not kwargs.get("inputs"):
            logger.error(
                "Input texts must be provided in kwargs to match sources with predictions."
            )
            raise ValueError(
                "Input texts must be provided in kwargs to match sources with predictions."
            )

        # Load sources from the dataset
        try:
            from app.config.registry import get_dataset_loader

            dataset_loader = get_dataset_loader()
            dataset = dataset_loader.load(
                dataset_name=kwargs.get("dataset", ""), cache=False
            )
            data = dataset[kwargs.get("split", "train")]

            # Get the input column and inputs to match
            input_column = kwargs.get("input_column")
            inputs = kwargs.get("inputs", [])

            # Build a mapping from input text to source text
            input_to_source = {}
            for idx in range(len(data)):
                input_text = data[input_column][idx]
                source_text = data[sources_column][idx]
                input_to_source[input_text] = source_text

            # Match sources with predictions based on the provided inputs
            sources = []
            for input_text in inputs:
                if input_text not in input_to_source:
                    logger.error(f"Input text not found in dataset: {input_text}")
                    raise ValueError(
                        f"Input text not found in dataset column '{input_column}'. This usually means the input text format doesn't match the dataset entries."
                    )
                sources.append(input_to_source[input_text])

        except Exception as e:
            raise RuntimeError(
                f"Failed to load sources from dataset '{kwargs.get('dataset')}' split '{kwargs.get('split')}' column '{sources_column}': {e!s}"
            ) from e

        # Validate sources length matches predictions and references
        if len(sources) != len(predictions):
            logger.error(
                f"Sources length ({len(sources)}) does not match predictions length ({len(predictions)})."
            )
            raise ValueError(
                f"Sources length ({len(sources)}) must match predictions length ({len(predictions)})."
            )

        # Validate sources is not empty and contains valid data
        if not sources or any(not s or not s.strip() for s in sources):
            logger.error(
                f"Sources list is empty or contains empty strings. Inputs provided: {len(inputs)}, Dataset has {len(data)} entries. Matched sources: {[s[:50] if s else '<empty>' for s in sources]}"
            )
            raise ValueError(
                f"Sources list is empty or contains empty strings. Inputs provided: {len(inputs)}, Dataset has {len(data)} entries. Matched sources: {[s[:50] if s else '<empty>' for s in sources]}"
            )

        try:
            return self.metric.compute(
                sources=sources,
                predictions=predictions,
                references=references,
            )
        except Exception as e:
            logger.error(f"Error computing SARI metric: {e!s}")
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
