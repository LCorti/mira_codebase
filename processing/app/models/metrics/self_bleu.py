import itertools
from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric


class SelfBleuMetric(Metric):
    def __init__(self):
        """
        Initialize the SelfBleuMetric.
        Self-BLEU measures the diversity of generated text by computing BLEU scores
        between generated texts within the same set. Lower scores indicate higher diversity.
        """
        super().__init__(
            name="self_bleu",
            description="Computes the Self-BLEU metric to measure diversity of generated text. Lower scores indicate higher diversity.",
        )
        self.metric = evaluate.load("bleu")

    def compute(
        self,
        predictions: list[str],
        # Not used in Self-BLEU, but kept for interface consistency
        references: list[str] | None = None,
        tokenizer=None,
        max_order: int = 4,
        smooth: bool = False,
        **kwargs: Any,
    ):
        """
        Compute the Self-BLEU metric for measuring diversity of generated text.

        Self-BLEU compares each prediction against all other predictions in the same set,
        then averages the scores. Lower Self-BLEU indicates higher diversity (less repetition).

        :param predictions: list of generated texts to measure diversity for.
        :type predictions: list[str]
        :param references: Not used in Self-BLEU computation, kept for interface consistency.
        :type references: list[str], optional
        :param tokenizer: Approach used for tokenizing predictions. The default tokenizer is tokenizer_13a.
        :type tokenizer: callable, optional
        :param max_order: Maximum n-gram order to use when computing BLEU score. Defaults to 4.
        :type max_order: int, optional
        :param smooth: Whether or not to apply Lin et al. 2004 smoothing. Defaults to False.
        :type smooth: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed Self-BLEU metric values including self_bleu score and individual scores.
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions:
            raise ValueError("Predictions must not be empty.")
        if len(predictions) < 2:
            raise ValueError(
                "Self-BLEU requires at least 2 predictions to compute diversity."
            )

        # Validate max_order parameter
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError("max_order must be a positive integer.")

        # Validate smooth parameter
        if not isinstance(smooth, bool):
            raise TypeError("`smooth` must be a boolean value.")

        try:
            # Compute Self-BLEU scores
            all_bleu_scores = []

            # For each prediction, compute its BLEU score against all other predictions
            for i, current_prediction in enumerate(predictions):
                # Get all other predictions as references
                other_predictions = [
                    pred for j, pred in enumerate(predictions) if j != i
                ]

                if not other_predictions:
                    continue

                # Prepare arguments for the metric computation
                compute_args = {
                    "predictions": [current_prediction],  # Single prediction
                    "references": [
                        other_predictions
                    ],  # All other predictions as references
                    "max_order": max_order,
                    "smooth": smooth,
                    **kwargs,
                }

                # Add tokenizer only if provided
                if tokenizer is not None:
                    compute_args["tokenizer"] = tokenizer

                # Compute BLEU score for this prediction against others
                result = self.metric.compute(**compute_args)
                all_bleu_scores.append(result["bleu"])

            # Calculate average Self-BLEU score
            if all_bleu_scores:
                avg_self_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
            else:
                avg_self_bleu = 0.0

            return {
                "self_bleu": {
                    "self_bleu": avg_self_bleu,
                    "individual_scores": all_bleu_scores,
                    "num_comparisons": len(all_bleu_scores),
                    "diversity_interpretation": self._interpret_diversity(
                        avg_self_bleu
                    ),
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e

    def _interpret_diversity(self, self_bleu_score: float) -> str:
        """
        Provide a human-readable interpretation of the Self-BLEU score.

        :param self_bleu_score: The computed Self-BLEU score.
        :type self_bleu_score: float

        :return: Human-readable interpretation of diversity.
        :rtype: str
        """
        if self_bleu_score < 0.1:
            return "Very high diversity (very low repetition)"
        elif self_bleu_score < 0.3:
            return "High diversity (low repetition)"
        elif self_bleu_score < 0.5:
            return "Moderate diversity"
        elif self_bleu_score < 0.7:
            return "Low diversity (high repetition)"
        else:
            return "Very low diversity (very high repetition)"

    def compute_pairwise(
        self,
        predictions: list[str],
        tokenizer=None,
        max_order: int = 4,
        smooth: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Compute pairwise BLEU scores between all predictions for detailed analysis.

        :param predictions: List of generated texts.
        :type predictions: list[str]
        :param tokenizer: Tokenizer to use for BLEU computation.
        :type tokenizer: callable, optional
        :param max_order: Maximum n-gram order. Defaults to 4.
        :type max_order: int, optional
        :param smooth: Whether to apply smoothing. Defaults to False.
        :type smooth: bool, optional
        :param kwargs: Additional arguments.
        :type kwargs: Any

        :return: Dictionary containing pairwise scores and statistics.
        :rtype: dict[str, Any]
        """
        if len(predictions) < 2:
            raise ValueError("Pairwise computation requires at least 2 predictions.")

        pairwise_scores = []
        pairs = list(itertools.combinations(range(len(predictions)), 2))

        for i, j in pairs:
            # Compute BLEU score between prediction i and prediction j
            compute_args = {
                "predictions": [predictions[i]],
                "references": [[predictions[j]]],
                "max_order": max_order,
                "smooth": smooth,
                **kwargs,
            }

            if tokenizer is not None:
                compute_args["tokenizer"] = tokenizer

            result = self.metric.compute(**compute_args)
            pairwise_scores.append({"pair": (i, j), "bleu": result["bleu"]})

        # Calculate statistics
        bleu_values = [score["bleu"] for score in pairwise_scores]

        return {
            "pairwise_scores": pairwise_scores,
            "mean_pairwise_bleu": sum(bleu_values) / len(bleu_values),
            "min_pairwise_bleu": min(bleu_values),
            "max_pairwise_bleu": max(bleu_values),
            "num_pairs": len(pairs),
        }
