import json
import os

from app.config.paths import CONFIG_DIR
from app.models.metrics.accuracy import AccuracyMetric
from app.models.metrics.average_precision import AveragePrecisionMetric
from app.models.metrics.base_metric import Metric
from app.models.metrics.bertscore import BertScoreMetric
from app.models.metrics.bleu import BleuMetric
from app.models.metrics.compliance_check import ComplianceCheckMetric
from app.models.metrics.custom_metric import CustomMetric
from app.models.metrics.f1 import F1Metric
from app.models.metrics.harim_plus import HarimPlusMetric
from app.models.metrics.mean_average_precision import MeanAveragePrecisionMetric
from app.models.metrics.median_rank_correct_word import MRCWMetric
from app.models.metrics.meteor import MeteorMetric
from app.models.metrics.mrr import MRRMetric
from app.models.metrics.ndcg import NDCGMetric
from app.models.metrics.perplexity import PerplexityMetric
from app.models.metrics.precision import PrecisionMetric
from app.models.metrics.readability import ReadabilityMetric
from app.models.metrics.recall import RecallMetric
from app.models.metrics.rouge import RougeMetric
from app.models.metrics.sari import SariMetric
from app.models.metrics.self_bleu import SelfBleuMetric
from app.models.metrics.wecheck import WeCheckMetric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)

AVAILABLE_METRIC_CLASSES = {
    "AccuracyMetric": AccuracyMetric,
    "RougeMetric": RougeMetric,
    "BleuMetric": BleuMetric,
    "SelfBleuMetric": SelfBleuMetric,
    "MeteorMetric": MeteorMetric,
    "BertScoreMetric": BertScoreMetric,
    "PrecisionMetric": PrecisionMetric,
    "RecallMetric": RecallMetric,
    "F1Metric": F1Metric,
    "AveragePrecisionMetric": AveragePrecisionMetric,
    "MeanAveragePrecisionMetric": MeanAveragePrecisionMetric,
    "NDCGMetric": NDCGMetric,
    "MRRMetric": MRRMetric,
    "CustomMetric": CustomMetric,
    "ReadabilityMetric": ReadabilityMetric,
    "WeCheckMetric": WeCheckMetric,
    "HarimPlusMetric": HarimPlusMetric,
    "SariMetric": SariMetric,
    "PerplexityMetric": PerplexityMetric,
    "MRCWMetric": MRCWMetric,
    "ComplianceCheckMetric": ComplianceCheckMetric,
}


class MetricRegistry:
    """Registry for managing available metrics with metadata and validation."""

    def __init__(self, config_path: str | None = None):
        self._metrics: dict[str, type[Metric]] = {}
        self._metadata: dict[str, dict] = {}
        self._config_path = config_path or CONFIG_DIR.joinpath("metrics.json")

        # Check if config file exists
        if not os.path.exists(self._config_path):
            logger.warning(
                f"Config file not found: {self._config_path}. Creating it now."
            )
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)

            # Write an empty dict to initialize the config file
            with open(self._config_path, "w") as f:
                json.dump({}, f, indent=4)
                logger.debug(
                    f"Created new config file at '{self._config_path}' with empty metrics object."
                )

            self._register_default_metrics(register=True)
            logger.info(
                f"MetricRegistry initialized with {len(self._metrics)} metrics."
            )
            return

        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file {self._config_path}: {e}")
            self._register_default_metrics(register=True)

        if not config or not config.items():
            self._register_default_metrics(register=True)
        else:
            self._load_from_config()

        logger.info(f"MetricRegistry initialized with {len(self._metrics)} metrics.")

    def _load_from_config(self):
        """Load metrics configuration from JSON file."""
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            for name, metric_config in config.items():
                metric_class_name = metric_config.get("class")
                if metric_class_name not in AVAILABLE_METRIC_CLASSES:
                    logger.warning(
                        f"Unknown metric class '{metric_class_name}' for metric '{name}'. Skipping."
                    )
                    continue
                metric_class = AVAILABLE_METRIC_CLASSES[metric_class_name]
                metric_metadata = metric_config.get("metadata", {})
                self.register(name, metric_class, metric_metadata)

            logger.debug(f"Loaded {len(self._metrics)} metrics from config")
        except Exception as e:
            logger.error(f"Failed to load config from {self._config_path}: {e!s}")
            self._register_default_metrics()

    def _register_default_metrics(self, register: bool = False):
        """Fallback: Register default metrics without config."""
        self.register(
            "accuracy",
            AccuracyMetric,
            {
                "name": "accuracy",
                "description": "Computes accuracy metric for retrieval tasks",
                "formula": "(TP + TN) / (TP + TN + FP + FN)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["accuracy"],
                "task_types": ["text-retrieval", "text-generation", "summarization"],
                "sub_task_types": {
                    "text-retrieval": ["document-retrieval"],
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Accuracy",
            },
            register=register,
        )

        self.register(
            "rouge",
            RougeMetric,
            {
                "name": "rouge",
                "description": "Computes ROUGE metric for text generation tasks",
                "formula": "Σ(n-grams in ref ∩ n-grams in pred) / Σ(n-grams in ref)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "multiple_scores",
                "output_metrics": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {"text-generation": [], "summarization": []},
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/W04-1013/",
            },
            register=register,
        )

        self.register(
            "bleu",
            BleuMetric,
            {
                "name": "bleu",
                "description": "Computes BLEU metric for machine translation and text generation tasks",
                "formula": "BP x exp(Σ(w_n x log(p_n))) where p_n is n-gram precision",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "multiple_scores",
                "output_metrics": [
                    "bleu",
                    "precisions",
                    "brevity_penalty",
                    "length_ratio",
                    "translation_length",
                    "reference_length",
                ],
                "task_types": ["translation", "text-generation"],
                "sub_task_types": {
                    "translation": [],
                    "text-generation": ["text-simplification"],
                },
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/P02-1040/",
            },
            register=register,
        )

        # self.register("self_bleu", SelfBleuMetric, {
        #     "name": "self_bleu",
        #     "description": "Computes Self-BLEU metric for measuring diversity of generated text",
        #     "formula": "1/N x Σ(BLEU(sent_i, corpus\\sent_i)) for diversity measurement",
        #     "range": "0-1 (lower is better for diversity)",
        #     "input_types": ["predictions"],
        #     "output_format": "single_score",
        #     "output_metrics": ["self_bleu"],
        #     "task_types": ["text-generation"],
        #     "sub_task_types": {"text-generation": []},
        #     "task_blacklist": [],
        #     "paper_ref": "https://dl.acm.org/doi/10.1145/3209978.3210080"
        # }, register=register)

        self.register(
            "meteor",
            MeteorMetric,
            {
                "name": "meteor",
                "description": "Computes METEOR metric for machine translation evaluation",
                "formula": "(1 - penalty) x harmonic_mean(precision, recall)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["meteor"],
                "task_types": ["translation", "text-generation"],
                "sub_task_types": {
                    "translation": [],
                    "text-generation": [],
                    "summarization": [],
                },
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/W05-0909/",
            },
            register=register,
        )

        self.register(
            "bertscore",
            BertScoreMetric,
            {
                "name": "bertscore",
                "description": "Computes BERTScore metric for text generation evaluation using BERT embeddings",
                "formula": "cosine_similarity(BERT_embeddings(pred), BERT_embeddings(ref))",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "multiple_scores",
                "output_metrics": ["precision", "recall", "f1"],
                "task_types": ["text-generation", "translation", "summarization"],
                "sub_task_types": {
                    "text-generation": [],
                    "translation": [],
                    "summarization": [],
                },
                "task_blacklist": [],
                "paper_ref": "https://arxiv.org/abs/1904.09675",
            },
            register=register,
        )

        self.register(
            "precision",
            PrecisionMetric,
            {
                "name": "precision",
                "description": "Computes precision metric for classification tasks",
                "formula": "TP / (TP + FP)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["precision"],
                "task_types": [
                    "text-classification",
                    "token-classification",
                    "text-retrieval",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "token-classification": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Precision_and_recall",
            },
            register=register,
        )

        self.register(
            "recall",
            RecallMetric,
            {
                "name": "recall",
                "description": "Computes recall metric for classification tasks",
                "formula": "TP / (TP + FN)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["recall"],
                "task_types": [
                    "text-classification",
                    "token-classification",
                    "text-retrieval",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "token-classification": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Precision_and_recall",
            },
            register=register,
        )

        self.register(
            "f1",
            F1Metric,
            {
                "name": "f1",
                "description": "Computes F1 score metric for classification tasks",
                "formula": "2 x (precision x recall) / (precision + recall)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["f1"],
                "task_types": [
                    "text-classification",
                    "token-classification",
                    "text-retrieval",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "token-classification": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/F1_score",
            },
            register=register,
        )

        self.register(
            "average_precision",
            AveragePrecisionMetric,
            {
                "name": "average_precision",
                "description": "Computes Average Precision (AP) metric for binary classification and ranking tasks",
                "formula": "Σ(P(k) x rel(k)) where P(k) is precision at rank k",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["Average Precision"],
                "task_types": [
                    "text-classification",
                    "token-classification",
                    "text-ranking",
                    "text-retrieval",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "token-classification": [],
                    "text-ranking": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Average_precision",
            },
            register=register,
        )

        self.register(
            "mean_average_precision",
            MeanAveragePrecisionMetric,
            {
                "name": "mean_average_precision",
                "description": "Computes Mean Average Precision (MAP) metric for multi-class classification and information retrieval tasks",
                "formula": "(1/Q) x Σ(AP_q) where Q is number of queries and AP_q is average precision for query q",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["Mean Average Precision"],
                "task_types": [
                    "text-classification",
                    "token-classification",
                    "text-ranking",
                    "text-retrieval",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "token-classification": [],
                    "text-ranking": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Mean_average_precision",
            },
            register=register,
        )

        self.register(
            "ndcg",
            NDCGMetric,
            {
                "name": "ndcg",
                "description": "Computes Normalized Discounted Cumulative Gain (NDCG) metric for ranking tasks with graded relevance",
                "formula": "DCG@k / IDCG@k where DCG = Σ(2^rel_i - 1) / log2(i + 1)",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["ndcg"],
                "task_types": ["text-ranking", "text-retrieval"],
                "sub_task_types": {
                    "text-ranking": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Discounted_cumulative_gain",
            },
            register=register,
        )

        self.register(
            "mrr",
            MRRMetric,
            {
                "name": "mrr",
                "description": "Computes Mean Reciprocal Rank (MRR) metric for ranking tasks",
                "formula": "(1/|Q|) x Σ(1/rank_i) where rank_i is the rank of the first relevant item",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["mrr"],
                "task_types": ["text-ranking", "text-retrieval"],
                "sub_task_types": {
                    "text-ranking": [],
                    "text-retrieval": ["document-retrieval"],
                },
                "task_blacklist": [],
                "paper_ref": "https://en.wikipedia.org/wiki/Mean_reciprocal_rank",
            },
            register=register,
        )

        self.register(
            "readability",
            ReadabilityMetric,
            {
                "name": "readability",
                "description": "Computes various readability metrics for text analysis",
                "formula": "Multiple formulas: Flesch-Kincaid, Gunning Fog, Coleman-Liau, etc.",
                "range": "Varies by metric (e.g., Flesch: 0-100, Grade levels: 1-20+)",
                "input_types": ["texts"],
                "output_format": "multiple_scores",
                "output_metrics": [
                    "Flesh-Kincaid Grade Level",
                    "Flesch Reading Ease",
                    "Dale-Chall Readability",
                    "Automated Readability Index",
                    "Coleman-Liau Index",
                    "Gunning Fog Score",
                    "SMOG Index",
                    "Spache Readability",
                    "Linsear Write",
                ],
                "task_types": ["summarization", "text-generation"],
                "sub_task_types": {
                    "summarization": [],
                    "text-generation": ["text-simplification"],
                },
                "task_blacklist": [],
                "paper_ref": "https://py-readability-metrics.readthedocs.io/en/latest/",
            },
            register=register,
        )

        self.register(
            "wecheck",
            WeCheckMetric,
            {
                "name": "wecheck",
                "description": "Evaluates factual consistency between premise and hypothesis using WeCheck model.",
                "formula": "WeCheck score based on transformer model predictions for factual consistency",
                "range": "0-1",
                "input_types": ["predictions", "references"],
                "output_format": "multiple_scores",
                "output_metrics": ["Factual Consistency", "Binary Consistency"],
                "task_types": [
                    "text-classification",
                    "summarization",
                    "text-generation",
                ],
                "sub_task_types": {
                    "text-classification": [],
                    "summarization": [],
                    "text-generation": ["text-simplification"],
                },
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/2023.acl-long.18/",
            },
            register=register,
        )

        self.register(
            "harim_plus",
            HarimPlusMetric,
            {
                "name": "harim_plus",
                "description": "HaRiM+ is a reference-less evaluation metric for summarization quality assessment using a summarization model.",
                "formula": "HaRiM+ score based on neural model assessment of summary quality without references",
                "range": "unbounded (higher is better)",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["HaRiM+ Score"],
                "task_types": ["summarization"],
                "sub_task_types": {"summarization": []},
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/2022.aacl-main.66/",
            },
            register=register,
        )

        self.register(
            "sari",
            SariMetric,
            {
                "name": "sari",
                "description": "Sentence-level evaluation metric for text simplification. It compares the predicted simplified sentences against the reference and the source sentence.",
                "formula": "SARI score based on n-gram precision and recall between simplified and reference sentences.",
                "range": "0-100",
                "input_types": ["predictions", "references"],
                "output_format": "single_score",
                "output_metrics": ["SARI Score"],
                "task_types": ["text-generation"],
                "sub_task_types": {"text-generation": ["text-simplification"]},
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/Q16-1029/",
            },
            register=register,
        )

        self.register(
            "perplexity",
            PerplexityMetric,
            {
                "name": "perplexity",
                "description": "Computes the perplexity metric for model predictions. Perplexity measures how well a probability model predicts a sample.",
                "formula": "e**(sum(losses) / num_tokenized_tokens)",
                "range": "0-∞ (lower is better)",
                "input_types": ["predictions"],
                "output_format": "single_score",
                "task_types": ["text-generation"],
                "sub_task_types": {"text-generation": ["text-simplification"]},
                "task_blacklist": ["summarization"],
                "paper_ref": "https://en.wikipedia.org/wiki/Perplexity",
            },
            register=register,
        )

        self.register(
            "median_rank_correct_word",
            MRCWMetric,
            {
                "name": "median_rank_correct_word",
                "description": "Median Rank of the Correct Word (MRCW) measures how well a model ranks the correct next word in its predictions.",
                "formula": "Median rank position of the correct next word in the predicted logits",
                "range": "1-Vocab Size (lower is better)",
                "input_types": ["predictions", "references", "logits"],
                "output_format": "single_score",
                "output_metrics": ["Median Rank"],
                "task_types": ["median-rank-correct-word"],
                "sub_task_types": {"median-rank-correct-word": []},
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/P16-1144/",
            },
            register=register,
        )

        self.register(
            "compliance_check",
            ComplianceCheckMetric,
            {
                "name": "compliance_check",
                "description": "Compliance Check Metric evaluates the adherence of model predictions to safety guidelines. It classifies outputs as compliant or non-compliant based on predefined rules.",
                "formula": "Compliance label (0: compliant, 1: non-compliant)",
                "range": "0 or 1",
                "input_types": ["predictions", "references"],
                "output_format": "dict",
                "output_metrics": ["Compliance Label"],
                "task_types": ["compliance-check"],
                "sub_task_types": {"compliance-check": []},
                "task_blacklist": [],
                "paper_ref": "https://aclanthology.org/2024.naacl-long.301/",
            },
            register=register,
        )

    def register(
        self,
        name: str,
        metric_class: type[Metric],
        metadata: dict | None = None,
        register: bool = False,
    ):
        """Register a metric with optional metadata."""
        if not issubclass(metric_class, Metric):
            raise TypeError(f"Metric class {metric_class} must inherit from Metric")

        self._metrics[name] = metric_class
        self._metadata[name] = metadata or {}
        if register:
            self.save_metric_to_config(metric_class, metadata)
        logger.debug(f"Registered metric: {name}")

    def get_metric(self, name: str, **kwargs) -> Metric:
        """
        Get a metric instance by name.

        :param name: The name of the metric to retrieve.
        :param kwargs: Additional keyword arguments for metric initialization.
        :return: An instance of the specified metric.
        :raises ValueError: If the metric is not found.
        """
        if name not in self._metrics:
            available = ", ".join(self._metrics.keys())
            raise ValueError(f"Metric '{name}' not found. Available: {available}")

        try:
            return self._metrics[name](**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create metric '{name}': {e!s}") from e

    def get_or_create_metric(self, name: str, **kwargs) -> Metric:
        """
        Get a metric instance by name, or create it if it doesn't exist.

        :param name: The name of the metric to retrieve or create.
        :param kwargs: Additional keyword arguments for metric initialization.
        :return: An instance of the specified metric.
        """
        if name not in self._metrics:
            return self.create_custom_metric(name, **kwargs)

        return self.get_metric(name, **kwargs)

    def list_metrics(self) -> list[str]:
        """
        List all available metric names.

        :return: A list of metric names.
        """
        return list(self._metrics.keys())

    def get_metadata(self, name: str) -> dict:
        """
        Get metadata for a specific metric.

        :param name: The name of the metric to retrieve metadata for.
        :return: Metadata dictionary for the specified metric.
        :raises ValueError: If the metric is not found.
        """
        if name not in self._metadata:
            raise ValueError(f"Metric '{name}' not found")
        return self._metadata[name]

    def get_all_metadata(self) -> dict[str, dict]:
        """
        Get metadata for all registered metrics.

        :return: A dictionary mapping metric names to their metadata.
        """
        return self._metadata.copy()

    def find_metrics_by_task(self, task_types: list[str]) -> list[str]:
        """
        Find metrics suitable for specific task types.

        :param task_types: A list of task types to find metrics for (e.g., ["classification", "text-generation"]).
        :return: A list of metric names that are suitable for any of the specified task types.
        """
        task_types_set = set(task_types)
        suitable_metrics = set()

        for name, metadata in self._metadata.items():
            # Check if any task_type is in the blacklist
            task_blacklist = set(metadata.get("task_blacklist", []))
            if task_types_set & task_blacklist:  # Set intersection
                continue

            main_task_types = set(metadata.get("task_types", []))
            if task_types_set & main_task_types:  # Set intersection
                suitable_metrics.add(name)
                continue

            # Check if any task_type is in any of the sub-task type arrays
            sub_task_types = metadata.get("sub_task_types", {})
            all_sub_tasks = set()
            for subtasks in sub_task_types.values():
                all_sub_tasks.update(subtasks)

            if task_types_set & all_sub_tasks:  # Set intersection
                suitable_metrics.add(name)

        return list(suitable_metrics)

    def create_custom_metric(
        self, metric_name: str, description: str | None = None, **metric_kwargs
    ) -> CustomMetric:
        """
        Get a custom metric instance by loading any Hugging Face evaluate metric.

        :param metric_name: The name of the Hugging Face metric to load
        :param description: Optional custom description for the metric
        :param metric_kwargs: Additional keyword arguments to pass to evaluate.load()
        :return: A CustomMetric instance
        :raises RuntimeError: If the metric cannot be loaded
        """
        try:
            return CustomMetric(metric_name, description, **metric_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create custom metric '{metric_name}': {e!s}"
            ) from e

    def save_metric_to_config(self, metric: type[Metric], metadata: dict | None = None):
        """
        Save a custom metric configuration to the JSON config file for persistence.

        :param metric: The Metric class to save
        :param metadata: Optional metadata
        """
        if not issubclass(metric, Metric):
            raise TypeError(f"Expected Metric class, got {type(metric)}")

        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            # Create a temporary instance to get the name
            temp_instance = metric()
            metric_name = temp_instance.name
            metric_class = metric
            metric_metadata = metadata or {
                "description": temp_instance.description
                or f"Custom {metric_name} metric",
                "input_types": ["N/A"],
                "output_format": "N/A",
                "task_types": ["N/A"],
            }

            config[metric_name] = {
                "name": metric_name,
                "class": metric_class.__name__,
                "metadata": metric_metadata,
            }

            # Save back to file
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=4)

            logger.debug(f"Saved custom metric '{metric_name}' to config")

            # Also register it in memory immediately
            self.register(metric_name, metric_class, metric_metadata)
        except Exception as e:
            logger.error(f"Failed to save custom metric to config: {e!s}")
            raise RuntimeError(f"Failed to save custom metric to config: {e!s}") from e

    def remove_custom_metric_from_config(self, metric_name: str):
        """
        Remove a custom metric from the config file.

        :param name: The name of the metric to remove
        """
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            if not metric_name in config:
                logger.warning(f"Custom metric '{metric_name}' not found in config")
                raise ValueError(f"Custom metric '{metric_name}' not found in config")

            del config[metric_name]

            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=4)

            # Remove from memory too
            if metric_name in self._metrics:
                del self._metrics[metric_name]
            if metric_name in self._metadata:
                del self._metadata[metric_name]

            logger.debug(f"Removed custom metric '{metric_name}' from config")

        except Exception as e:
            logger.error(f"Failed to remove custom metric from config: {e!s}")
            raise RuntimeError(
                f"Failed to remove custom metric from config: {e!s}"
            ) from e

    def close(self):
        """
        Close the MetricRegistry and release any resources.
        """
        self._metrics.clear()
        self._metadata.clear()
        logger.info("MetricRegistry has been closed.")
