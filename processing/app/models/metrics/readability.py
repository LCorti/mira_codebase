from typing import Any, ClassVar

from readability import Readability

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class ReadabilityMetric(Metric):
    """
    A wrapper class for py-readability-metrics library that provides native support
    for various readability formulas including Flesch-Kincaid, Gunning Fog, ARI,
    Dale-Chall, SMOG, and more.

    This metric computes readability scores for text, which can be useful for
    assessing the complexity and grade level of generated text.
    """

    # List of available readability metrics
    AVAILABLE_METRICS: ClassVar[list[str]] = [
        "flesch_kincaid",
        "flesch",
        "gunning_fog",
        "coleman_liau",
        "dale_chall",
        "ari",
        "linsear_write",
        "smog",
        "spache",
    ]

    def __init__(self):
        """
        Initialize the ReadabilityMetric.

        The specific metrics to compute are specified in the compute() method.
        """
        super().__init__(
            name="readability",
            description="Computes readability metrics for model predictions using py-readability-metrics.",
        )

        # We don't use a single metric object like other metrics, but handle computation directly
        self.metric = self  # Set to self to pass the base class check

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[int] | None = None,
        metrics: list[str] | None = None,
        smog_all_sentences: bool = False,
        return_individual_scores: bool = True,
        **kwargs: Any,
    ):
        """
        Compute readability metrics for the given predictions.

        Note: The references parameter is ignored for readability metrics as they are
        computed solely based on the text properties of predictions.

        :param predictions: List of text predictions to analyze for readability.
        :type predictions: List[str]
        :param references: Not used for readability metrics (included for compatibility).
        :type references: Union[List[str], List[int]], optional
        :param metrics: List of readability metrics to compute. If None, all available metrics will be computed. Available metrics: 'flesch_kincaid', 'flesch', 'gunning_fog', 'coleman_liau', 'dale_chall', 'ari', 'linsear_write', 'smog', 'spache'
        :type metrics: Optional[List[str]]
        :param smog_all_sentences: For SMOG metric, whether to use all sentences instead of sampling 30.
        :type smog_all_sentences: bool
        :param return_individual_scores: Whether to return individual scores per prediction or aggregated scores.
        :type return_individual_scores: bool
        :param kwargs: Additional keyword arguments (ignored).
        :type kwargs: Any

        :return: Dictionary containing computed readability metrics.
        :rtype: Union[Dict[str, float], Dict[str, List[float]]]
        """
        if not predictions:
            raise ValueError("Predictions must not be empty.")

        # Set default metrics if none specified
        if metrics is None:
            selected_metrics = self.AVAILABLE_METRICS
        else:
            selected_metrics = metrics

        # Validate metrics
        invalid_metrics = [
            m for m in selected_metrics if m not in self.AVAILABLE_METRICS
        ]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Available metrics: {self.AVAILABLE_METRICS}"
            )

        # Validate that predictions contain sufficient text
        for i, text in enumerate(predictions):
            if not isinstance(text, str):
                raise TypeError(
                    f"Prediction at index {i} must be a string, got {type(text)}"
                )
            if len(text.strip()) == 0:
                raise ValueError(
                    f"Prediction at index {i} is empty or contains only whitespace"
                )

        # Check for minimum word count (py-readability-metrics requires >= 100 words)
        for i, text in enumerate(predictions):
            word_count = len(text.split())
            if word_count < 100:
                logger.warning(
                    f"Warning: Prediction at index {i} has only {word_count} words. py-readability-metrics recommends at least 100 words for accurate results."
                )

        try:
            all_scores = {}
            all_errors = {}  # Track errors per text
            failed_metrics_set = set()  # Track which metrics failed across all texts

            if return_individual_scores:
                # Initialize lists for each metric
                for metric_name in selected_metrics:
                    all_scores[metric_name] = []

            # Compute readability for each prediction
            individual_results = []
            for idx, text in enumerate(predictions):
                try:
                    r = Readability(text)
                    text_scores, text_errors = self.__compute_single_text(
                        r, selected_metrics, smog_all_sentences
                    )
                    individual_results.append(text_scores)

                    # Track errors for this text
                    if text_errors:
                        all_errors[f"text_{idx}"] = text_errors
                        failed_metrics_set.update(text_errors.keys())

                    if return_individual_scores:
                        for metric_name in selected_metrics:
                            # Only append if metric was successfully computed
                            if metric_name in text_scores:
                                all_scores[metric_name].append(text_scores[metric_name])

                except Exception as e:
                    logger.error(
                        f"Error: Failed to compute readability for text {idx}: {e!s}"
                    )
                    all_errors[f"text_{idx}"] = {"error": str(e)}

            # Prepare return value
            result = {"readability": all_scores if return_individual_scores else {}}

            if not return_individual_scores:
                # Return aggregated scores (averages) only for successful metrics
                aggregated_scores = {}
                for metric_name in selected_metrics:
                    if metric_name not in failed_metrics_set:
                        scores = [
                            res[metric_name]
                            for res in individual_results
                            if metric_name in res
                        ]
                        aggregated_scores[metric_name] = (
                            sum(scores) / len(scores) if scores else 0.0
                        )
                result["readability"] = aggregated_scores

            # Include error information if any errors occurred
            if all_errors:
                # Simplify error structure for single-text case
                if len(predictions) == 1 and "text_0" in all_errors:
                    # Single text: flatten the structure
                    result["errors"] = {
                        "failed_metrics": list(failed_metrics_set),
                        "error_details": {"text_0": all_errors["text_0"]},
                    }
                else:
                    # Multiple texts: keep the full structure
                    result["errors"] = {
                        "failed_metrics": list(failed_metrics_set),
                        "error_details": all_errors,
                    }

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to compute readability metrics: {e!s}") from e

    def __compute_single_text(
        self,
        readability_obj: Readability,
        selected_metrics: list[str],
        smog_all_sentences: bool = False,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """
        Compute readability metrics for a single text using the Readability object.

        :param readability_obj: Initialized Readability object for the text.
        :type readability_obj: Readability
        :param selected_metrics: List of metrics to compute for this text.
        :type selected_metrics: List[str]
        :param smog_all_sentences: Whether to use all sentences for SMOG computation.
        :type smog_all_sentences: bool

        :return: Tuple of (scores dictionary, errors dictionary).
        :rtype: tuple[Dict[str, float], Dict[str, str]]
        """
        scores = {}
        errors = {}

        for metric_name in selected_metrics:
            try:
                if metric_name == "flesch_kincaid":
                    result = readability_obj.flesch_kincaid()
                    scores[metric_name] = result.score

                elif metric_name == "flesch":
                    result = readability_obj.flesch()
                    scores[metric_name] = result.score

                elif metric_name == "gunning_fog":
                    result = readability_obj.gunning_fog()
                    scores[metric_name] = result.score

                elif metric_name == "coleman_liau":
                    result = readability_obj.coleman_liau()
                    scores[metric_name] = result.score

                elif metric_name == "dale_chall":
                    result = readability_obj.dale_chall()
                    scores[metric_name] = result.score

                elif metric_name == "ari":
                    result = readability_obj.ari()
                    scores[metric_name] = result.score

                elif metric_name == "linsear_write":
                    result = readability_obj.linsear_write()
                    scores[metric_name] = result.score

                elif metric_name == "smog":
                    result = readability_obj.smog(all_sentences=smog_all_sentences)
                    scores[metric_name] = result.score

                elif metric_name == "spache":
                    result = readability_obj.spache()
                    scores[metric_name] = result.score

            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e!s}")
                errors[metric_name] = str(e)
                # Don't include failed metric in scores

        return scores, errors

    def get_metric_details(self, metric_name: str) -> dict[str, str]:
        """
        Get detailed information about a specific readability metric.

        :param metric_name: Name of the readability metric.
        :type metric_name: str

        :return: Dictionary containing metric details.
        :rtype: Dict[str, str]
        """
        details = {
            "flesch_kincaid": {
                "name": "Flesch-Kincaid Grade Level",
                "description": "Measures the U.S. grade level required to understand the text. Used by the U.S. Army for technical manuals.",
                "interpretation": "Lower scores indicate easier readability. A score of 12 means high school senior level.",
                "range": "Typically 0-18+ (grade levels)",
            },
            "flesch": {
                "name": "Flesch Reading Ease",
                "description": "Measures how easy a text is to read. Used by the U.S. Department of Defense as standard.",
                "interpretation": "Higher scores indicate easier readability. 90-100: Very Easy, 60-70: Standard, 0-30: Very Difficult.",
                "range": "0-100 (higher is easier)",
            },
            "gunning_fog": {
                "name": "Gunning Fog Index",
                "description": "Estimates years of formal education needed to understand text on first reading.",
                "interpretation": "Lower scores indicate easier readability. A score of 12 requires high school senior level.",
                "range": "Typically 6-17 (years of education)",
            },
            "coleman_liau": {
                "name": "Coleman-Liau Index",
                "description": "Uses characters per word instead of syllables. Often gives lower grades for technical documents.",
                "interpretation": "Lower scores indicate easier readability. Represents U.S. grade level.",
                "range": "Typically 1-16+ (grade levels)",
            },
            "dale_chall": {
                "name": "Dale-Chall Readability",
                "description": "Based on familiar words rather than syllable counts. Very accurate for readability assessment.",
                "interpretation": "Lower scores indicate easier readability. 4.9 or below: easily understood by 4th graders.",
                "range": "Typically 0-10+ (grade levels)",
            },
            "ari": {
                "name": "Automated Readability Index (ARI)",
                "description": "Uses characters per word factor instead of syllables. Widely used on all text types.",
                "interpretation": "Lower scores indicate easier readability. Represents U.S. grade level.",
                "range": "Typically 1-14+ (grade levels)",
            },
            "linsear_write": {
                "name": "Linsear Write",
                "description": "Developed for the U.S. Air Force to calculate readability of technical manuals.",
                "interpretation": "Lower scores indicate easier readability. Represents U.S. grade level.",
                "range": "Typically 1-16+ (grade levels)",
            },
            "smog": {
                "name": "SMOG (Simple Measure of Gobbledygook)",
                "description": "Popular method for health literacy materials. Uses sample of 30 sentences by default.",
                "interpretation": "Lower scores indicate easier readability. Represents U.S. grade level.",
                "range": "Typically 4-18+ (grade levels)",
            },
            "spache": {
                "name": "Spache Readability Formula",
                "description": "Designed for primary-grade reading materials. Best for 3rd grade level and below.",
                "interpretation": "Lower scores indicate easier readability. Optimized for elementary texts.",
                "range": "Typically 1-4+ (grade levels)",
            },
        }

        if metric_name not in details:
            raise ValueError(
                f"Unknown metric: {metric_name}. Available metrics: {list(details.keys())}"
            )

        return details[metric_name]

    def __str__(self):
        return f"Readability Metric: {self.description}"

    def __repr__(self):
        return "ReadabilityMetric()"
