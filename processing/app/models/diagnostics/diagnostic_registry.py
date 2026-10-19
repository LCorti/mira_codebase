import json
import os

from app.config.paths import CONFIG_DIR
from app.models.diagnostics.base_diagnostic import Diagnostic
from app.models.diagnostics.hans import HansDiagnostic

# from app.models.diagnostics.cc_shap import CCSHAPDiagnostic
from app.models.diagnostics.holistic_bias import HolisticBiasDiagnostic
from app.models.diagnostics.ig_encoder import IGEncoderDiagnostic
from app.models.diagnostics.lm_diagnostics_accuracy import (
    LMDiagnosticsAccuracyDiagnostic,
)
from app.models.diagnostics.lm_diagnostics_sensitivity import (
    LMDiagnosticsSensitivityDiagnostic,
)
from app.models.diagnostics.lxt import LxtDiagnostic
from app.models.diagnostics.nli_stress_test import NLIStressTestDiagnostic
from app.models.diagnostics.sent_eval import SentEvalDiagnostic
from app.models.diagnostics.shap import ShapDiagnostic
from app.models.diagnostics.shap_encoder import ShapEncoderDiagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)

AVAILABLE_DIAGNOSTIC_CLASSES = {
    "LMDiagnosticsAccuracyDiagnostic": LMDiagnosticsAccuracyDiagnostic,
    "LMDiagnosticsSensitivityDiagnostic": LMDiagnosticsSensitivityDiagnostic,
    "HansDiagnostic": HansDiagnostic,
    "NLIStressTestDiagnostic": NLIStressTestDiagnostic,
    "SentEvalDiagnostic": SentEvalDiagnostic,
    # "CCSHAPDiagnostic": CCSHAPDiagnostic,
    "HolisticBiasDiagnostic": HolisticBiasDiagnostic,
    "LxtDiagnostic": LxtDiagnostic,
    "ShapDiagnostic": ShapDiagnostic,
    "ShapEncoderDiagnostic": ShapEncoderDiagnostic,
    "IGEncoderDiagnostic": IGEncoderDiagnostic,
}


class DiagnosticRegistry:
    """Registry for managing available diagnostics with metadata and validation."""

    def __init__(self, config_path: str | None = None):
        self._diagnostics: dict[str, type[Diagnostic]] = {}
        self._metadata: dict[str, dict] = {}
        self._config_path = config_path or CONFIG_DIR.joinpath("diagnostics.json")

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
                    f"Created new config file at '{self._config_path}' with empty diagnostics object."
                )

            self._register_default_diagnostics(register=True)
            logger.info(
                f"DiagnosticRegistry initialized with {len(self._diagnostics)} diagnostics."
            )
            return

        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file {self._config_path}: {e}")
            self._register_default_diagnostics(register=True)

        if not config or not config.items():
            self._register_default_diagnostics(register=True)
        else:
            self._load_from_config()

        logger.info(
            f"DiagnosticRegistry initialized with {len(self._diagnostics)} diagnostics."
        )

    def _load_from_config(self):
        """Load diagnostics configuration from JSON file."""
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            for name, diagnostic_config in config.items():
                diagnostic_class_name = diagnostic_config.get("class")
                if diagnostic_class_name not in AVAILABLE_DIAGNOSTIC_CLASSES:
                    logger.warning(
                        f"Unknown diagnostic class '{diagnostic_class_name}' for diagnostic '{name}'. Skipping."
                    )
                    continue
                diagnostic_class = AVAILABLE_DIAGNOSTIC_CLASSES[diagnostic_class_name]
                diagnostic_metadata = diagnostic_config.get("metadata", {})
                self.register(name, diagnostic_class, diagnostic_metadata)

            logger.debug(f"Loaded {len(self._diagnostics)} diagnostics from config")
        except Exception as e:
            logger.error(f"Failed to load config from {self._config_path}: {e!s}")
            self._register_default_diagnostics()

    def _register_default_diagnostics(self, register: bool = False):
        """Fallback: Register default diagnostics without config."""
        self.register(
            "lm_diagnostics_accuracy",
            LMDiagnosticsAccuracyDiagnostic,
            {
                "name": "lm_diagnostics_accuracy",
                "description": "Psycholinguistic diagnostic accuracy from LM Diagnostics. Measures how accurately a language model predicts psycholinguistic phenomena.",
                "input_types": ["predictions"],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "lm_diagnostics_sensitivity",
            LMDiagnosticsSensitivityDiagnostic,
            {
                "name": "lm_diagnostics_sensitivity",
                "description": "Psycholinguistic diagnostic sensitivity from LM Diagnostics. Measures how sensitive a language model is to psycholinguistic phenomena.",
                "input_types": ["target_probs"],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "hans",
            HansDiagnostic,
            {
                "name": "hans",
                "description": "Heuristic Analysis for NLI Systems (HANS) diagnostic. Evaluates whether NLI models rely on superficial heuristics.",
                "input_types": ["predictions"],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "nli_stress_test",
            NLIStressTestDiagnostic,
            {
                "name": "nli_stress_test",
                "description": "Stress Test diagnostic for NLI models. Automatically generate stress tests for word overlap, negation, length mismatch, antonym, spelling error and numerical reasoning.",
                "input_types": ["predictions"],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "sent_eval",
            SentEvalDiagnostic,
            {
                "name": "sent_eval",
                "description": "SentEval is a diagnostic suite designed to evaluate the linguistic properties captured by sentence embeddings, helping to assess how well language models understand various aspects of language.",
                "input_types": [],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        # self.register("cc_shap", CCSHAPDiagnostic, {
        #     "name": "cc_shap",
        #     "description": "Diagnostic for evaluating faithfulness or self-consistency of natural language explanations from the CC-SHAP method.",
        #     "input_types": ["predictions"],
        #     "output_format": "report_string",
        #     "task_types": ["text-generation", "summarization"],
        #     "sub_task_types": {"text-generation": ["text-simplification"], "summarization": []}
        # }, register=register)

        self.register(
            "holistic_bias",
            HolisticBiasDiagnostic,
            {
                "name": "holistic_bias",
                "description": "Holistic Bias is a diagnostic dataset designed to identify new biases in language models using a holistic descriptor dataset, helping to evaluate their fairness and bias mitigation.",
                "input_types": ["predictions"],
                "output_format": "report_string",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "lxt",
            LxtDiagnostic,
            {
                "name": "lxt",
                "description": "Executes the Layer-wise Relevance Propagation for Transformers (LXT) diagnostic from Achtibat et al (2024).",
                "input_types": ["input_text"],
                "output_format": "relevance_scores",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "shap",
            ShapDiagnostic,
            {
                "name": "shap",
                "description": "Diagnostic for evaluating SHapley Additive exPlanations (SHAP). SHAP values explain the output of a model by computing the contribution of each feature to the prediction.",
                "input_types": ["input_text", "predictions"],
                "output_format": "attribution_matrices",
                "task_types": ["text-generation", "summarization"],
                "sub_task_types": {
                    "text-generation": ["text-simplification"],
                    "summarization": [],
                },
            },
            register=register,
        )

        self.register(
            "shap_encoder",
            ShapEncoderDiagnostic,
            {
                "name": "shap_encoder",
                "description": "Diagnostic for evaluating SHapley Additive exPlanations (SHAP) using encoder models. SHAP values explain the output of a model by computing the contribution of each feature to the prediction.",
                "input_types": ["input_text", "predictions"],
                "output_format": "attribution_matrices",
                "task_types": ["text-retrieval"],
                "sub_task_types": {"text-retrieval": []},
            },
            register=register,
        )

        self.register(
            "integrated_gradients_encoder",
            IGEncoderDiagnostic,
            {
                "name": "integrated_gradients_encoder",
                "description": "Diagnostic for evaluating Integrated Gradients (IG) using encoder models. IG attributes the prediction of a model to its input features by integrating gradients along the path from a baseline to the input.",
                "input_types": ["input_text", "predictions"],
                "output_format": "attribution_matrices",
                "task_types": ["text-retrieval"],
                "sub_task_types": {"text-retrieval": []},
            },
            register=register,
        )

    def register(
        self,
        name: str,
        diagnostic_class: type[Diagnostic],
        metadata: dict | None = None,
        register: bool = False,
    ):
        """Register a diagnostic with optional metadata."""

        self._diagnostics[name] = diagnostic_class
        self._metadata[name] = metadata or {}
        if register:
            self.save_diagnostic_to_config(diagnostic_class, metadata)
        logger.debug(f"Registered diagnostic: {name}")

    def get_diagnostic(self, name: str, **kwargs) -> Diagnostic:
        """
        Get a diagnostic instance by name.

        :param name: The name of the diagnostic to retrieve.
        :param kwargs: Additional keyword arguments for diagnostic initialization.
        :return: An instance of the specified diagnostic.
        :raises ValueError: If the diagnostic is not found.
        """
        if name not in self._diagnostics:
            available = ", ".join(self._diagnostics.keys())
            raise ValueError(f"Diagnostic '{name}' not found. Available: {available}")

        try:
            return self._diagnostics[name](**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create diagnostic '{name}': {e!s}") from e

    def list_diagnostics(self) -> list[str]:
        """
        List all available diagnostic names.

        :return: A list of diagnostic names.
        """
        return list(self._diagnostics.keys())

    def get_metadata(self, name: str) -> dict:
        """
        Get metadata for a specific diagnostic.

        :param name: The name of the diagnostic to retrieve metadata for.
        :return: Metadata dictionary for the specified diagnostic.
        :raises ValueError: If the diagnostic is not found.
        """
        if name not in self._metadata:
            raise ValueError(f"Diagnostic '{name}' not found")
        return self._metadata[name]

    def get_all_metadata(self) -> dict[str, dict]:
        """
        Get metadata for all registered diagnostics.

        :return: A dictionary mapping diagnostic names to their metadata.
        """
        return self._metadata.copy()

    def save_diagnostic_to_config(
        self, diagnostic: type[Diagnostic], metadata: dict | None = None
    ):
        """
        Save a custom diagnostic configuration to the JSON config file for persistence.

        :param diagnostic: The Diagnostic class to save
        :param metadata: Optional metadata
        """

        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            # Create a temporary instance to get the name
            temp_instance = diagnostic()
            diagnostic_name = temp_instance.name
            diagnostic_class = diagnostic
            diagnostic_metadata = metadata or {
                "description": temp_instance.description
                or f"Custom {diagnostic_name} diagnostic",
                "input_types": ["N/A"],
                "output_format": "N/A",
                "task_types": ["N/A"],
            }

            config[diagnostic_name] = {
                "name": diagnostic_name,
                "class": diagnostic_class.__name__,
                "metadata": diagnostic_metadata,
            }

            # Save back to file
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=4)

            logger.debug(f"Saved custom diagnostic '{diagnostic_name}' to config")

            # Also register it in memory immediately
            self.register(diagnostic_name, diagnostic_class, diagnostic_metadata)
        except Exception as e:
            logger.error(f"Failed to save custom diagnostic to config: {e!s}")
            raise RuntimeError(
                f"Failed to save custom diagnostic to config: {e!s}"
            ) from e

    def remove_custom_diagnostic_from_config(self, diagnostic_name: str):
        """
        Remove a custom diagnostic from the config file.

        :param diagnostic_name: The name of the diagnostic to remove
        """
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)

            if not diagnostic_name in config:
                logger.warning(
                    f"Custom diagnostic '{diagnostic_name}' not found in config"
                )
                raise ValueError(
                    f"Custom diagnostic '{diagnostic_name}' not found in config"
                )

            del config[diagnostic_name]

            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=4)

            # Remove from memory too
            if diagnostic_name in self._diagnostics:
                del self._diagnostics[diagnostic_name]
            if diagnostic_name in self._metadata:
                del self._metadata[diagnostic_name]

            logger.debug(f"Removed custom diagnostic '{diagnostic_name}' from config")

        except Exception as e:
            logger.error(f"Failed to remove custom diagnostic from config: {e!s}")
            raise RuntimeError(
                f"Failed to remove custom diagnostic from config: {e!s}"
            ) from e

    def close(self):
        """
        Close the DiagnosticRegistry and release any resources.
        """
        self._diagnostics.clear()
        self._metadata.clear()
        logger.info("DiagnosticRegistry has been closed.")
