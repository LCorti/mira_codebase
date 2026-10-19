from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")

MODEL_NAME = "nightdessert/WeCheck"


class WeCheckMetric(Metric):
    def __init__(self):
        """
        Initialize the WeCheckMetric for factual consistency evaluation.

        This metric uses the WeCheck model to verify factual consistency between
        a premise (original text) and hypothesis (generated summary/simplification).
        WeCheck is a lighter and more efficient alternative to TrueTeacher.
        """
        super().__init__(
            name="wecheck",
            description="Evaluates factual consistency between premise and hypothesis using WeCheck model.",
        )
        self.model = None
        self.tokenizer = None

    def compute(self, predictions: list[str], references: list[str], **kwargs: Any):
        """
        Compute factual consistency scores using WeCheck model.

        :param predictions: List of hypotheses (generated summaries/simplifications)
        :type predictions: list[str]
        :param references: List of premises (original texts)
        :type references: list[str]
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Dictionary containing per-sample factual consistency scores and binary results
        :rtype: dict[str, list[float]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            logger.error(f"Error loading WeCheck tokenizer: {e!s}")
            raise RuntimeError(f"Failed to load WeCheck tokenizer: {e!s}") from e

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        except Exception as e:
            logger.error(f"Error loading WeCheck model: {e!s}")
            raise RuntimeError(f"Failed to load WeCheck model: {e!s}") from e

        scores = []
        binary_results = []

        try:
            for premise, hypothesis in zip(references, predictions):
                logger.info(
                    f"Evaluating premise: {premise}, type {type(premise)} | hypothesis: {hypothesis}, type {type(hypothesis)}"
                )
                if isinstance(premise, list):
                    premise = premise[0]
                    logger.warning(
                        "Premise is a list; using the first element for WeCheck evaluation."
                    )

                if isinstance(hypothesis, list):
                    hypothesis = hypothesis[0]
                    logger.warning(
                        "Hypothesis is a list; using the first element for WeCheck evaluation."
                    )

                inputs = self.tokenizer(
                    premise,
                    hypothesis,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )

                device = next(self.model.parameters()).device
                input_ids = inputs["input_ids"].to(device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs["logits"][:, 0]
                    consistency_score = torch.sigmoid(logits).item()

                scores.append(consistency_score)

                # Binary result: 1 if score > 0.5, 0 otherwise
                binary_result = 1.0 if consistency_score > 0.5 else 0.0
                binary_results.append(binary_result)

            return {"factual_consistency": scores, "binary_consistency": binary_results}

        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
