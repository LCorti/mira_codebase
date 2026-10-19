import json
import os
import re

import yaml
from datasets import get_dataset_config_info, load_dataset
from huggingface_hub import hf_hub_download

from app.config.paths import CONFIG_DIR
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)

DEFAULT_DATASETS = {
    "dtk-llms/mira_rag_queries": {
        "name": "dtk-llms/mira_rag_queries",
        "description": "A dataset for evaluating the model's performance on retrieval-augmented generation tasks.",
        "num_samples": 4,
        "splits": ["train"],
        "task_types": ["text-retrieval"],
    },
    "dtk-llms/mira_summarization": {
        "name": "dtk-llms/mira_summarization",
        "description": "A dataset for evaluating the model's performance on summarization tasks.",
        "num_samples": 6,
        "splits": ["train"],
        "task_types": ["summarization", "text-generation"],
    },
    "dtk-llms/mira_simplification": {
        "name": "dtk-llms/mira_simplification",
        "description": "A dataset for evaluating the model's performance on text simplification tasks.",
        "num_samples": 6,
        "splits": ["train"],
        "task_types": ["text-simplification", "text-generation"],
    },
}

DIAGNOSTIC_DATASETS = {
    "dtk-llms/lambada": {
        "name": "dtk-llms/lambada",
        "description": "Evaluating the capabilities of computational models for text understanding by means of a word prediction task.",
        "num_samples": 12000,
        "splits": ["train", "validation", "test"],
        "task_types": [
            "text-generation",
            "text-simplification",
            "summarization",
            "median-rank-correct-word",
        ],
    },
    "dtk-llms/XSTest": {
        "name": "dtk-llms/XSTest",
        "description": "Identifying Exaggerated Safety Behaviors in Large Language Models.",
        "num_samples": 1000,
        "splits": ["train"],
        "task_types": ["compliance-check"],
    },
    "dtk-llms/lm-diagnostics-cprag": {
        "name": "dtk-llms/lm-diagnostics-cprag",
        "description": "Commonsense Reasoning and Pragmatic inference (CPRAG) dataset from the LM Diagnostics paper.",
        "num_samples": 102,
        "splits": ["train", "validation"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/lm-diagnostics-negnat": {
        "name": "dtk-llms/lm-diagnostics-negnat",
        "description": "Negation Natural (NEG-NAT) dataset from the LM Diagnostics paper.",
        "num_samples": 64,
        "splits": ["train", "validation"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/lm-diagnostics-negsimp": {
        "name": "dtk-llms/lm-diagnostics-negsimp",
        "description": "Negation Simple (NEG-SIMP) dataset from the LM Diagnostics paper.",
        "num_samples": 72,
        "splits": ["train", "validation"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/lm-diagnostics-role": {
        "name": "dtk-llms/lm-diagnostics-role",
        "description": "Event Knowledge and Semantic Roles (ROLE) dataset from the LM Diagnostics paper.",
        "num_samples": 88,
        "splits": ["train", "validation"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/hans": {
        "name": "dtk-llms/hans",
        "description": "A dataset for evaluating heuristic use from the HANS method.",
        "num_samples": 60000,
        "splits": ["train", "test"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/NLIStressTest": {
        "name": "dtk-llms/NLIStressTest",
        "description": "A dataset for evaluating NLI stress tests from the 'Stress Test Evaluation for Natural Language Inference' paper.",
        "num_samples": 100000,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/SentEval": {
        "name": "dtk-llms/SentEval",
        "description": "A dataset for evaluating sentence embeddings using SentEval.",
        "num_samples": 0,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/CC-SHAP-comve": {
        "name": "dtk-llms/CC-SHAP-comve",
        "description": "A dataset for evaluating the CC-SHAP diagnostic on the ComVE task.",
        "num_samples": 1000,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/CC-SHAP-CJ": {
        "name": "dtk-llms/CC-SHAP-CJ",
        "description": "A dataset for evaluating the CC-SHAP diagnostic on the Causal Judgment task.",
        "num_samples": 160,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/CC-SHAP-DQA": {
        "name": "dtk-llms/CC-SHAP-DQA",
        "description": "A dataset for evaluating the CC-SHAP diagnostic on the Disambiguation QA task.",
        "num_samples": 228,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/CC-SHAP-LD5": {
        "name": "dtk-llms/CC-SHAP-LD5",
        "description": "A dataset for evaluating the CC-SHAP diagnostic on the Logical Deduction Five Objects task.",
        "num_samples": 220,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/CC-SHAP-esnli": {
        "name": "dtk-llms/CC-SHAP-esnli",
        "description": "A dataset for evaluating the CC-SHAP diagnostic on the e-SNLI task.",
        "num_samples": 9820,
        "splits": ["train"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "dtk-llms/HolisticBias": {
        "name": "dtk-llms/HolisticBias",
        "description": "A dataset for evaluating biases in language models.",
        "num_samples": 500000,
        "splits": ["train", "test"],
        "task_types": ["text-generation", "text-simplification", "summarization"],
    },
    "BeIR/trec-covid": {
        "name": "BeIR/trec-covid",
        "description": "A dataset for evaluating information retrieval models on COVID-19 related queries.",
        "num_samples": 50,
        "splits": ["queries", "corpus"],
        "task_types": ["text-retrieval"],
    },
    "BeIR/scifact": {
        "name": "BeIR/scifact",
        "description": "A dataset for evaluating information retrieval models on scientific claims.",
        "num_samples": 1100,
        "splits": ["queries", "corpus"],
        "task_types": ["text-retrieval"],
    },
    "BeIR/scidocs": {
        "name": "BeIR/scidocs",
        "description": "A dataset for evaluating information retrieval models on scientific documents.",
        "num_samples": 1000,
        "splits": ["queries", "corpus"],
        "task_types": ["text-retrieval"],
    },
}


class DatasetLoaderService:
    def __init__(self, config_path=None):
        """
        Initializes the DatasetLoader with the directory for datasets and the configuration file path.
        Args:
            config_path (str): Path to the configuration file containing dataset information.
        Raises:
            FileNotFoundError: If the specified dataset directory or configuration file does not exist.
        """
        if not config_path:
            self.__config_path = os.path.join(CONFIG_DIR, "datasets.json")
        else:
            self.__config_path = config_path
        self.__dataset = None

        # Check if config file exists
        if not os.path.exists(self.__config_path):
            logger.warning(
                f"Config file not found: {self.__config_path}. Creating it now."
            )
            os.makedirs(os.path.dirname(self.__config_path), exist_ok=True)

            # Write an empty dict to initialize the config file
            with open(self.__config_path, "w") as f:
                json.dump({}, f, indent=4)
                logger.debug(
                    f"Created new config file at '{self.__config_path}' with empty datasets object."
                )

            self._register_default_datasets()
            logger.info("DatasetLoaderService initialized with default datasets.")
            return

        try:
            with open(self.__config_path, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.critical(f"Error loading config file {self.__config_path}: {e}")
            return

        if not config or not config.items():
            self._register_default_datasets()
            logger.debug(
                "DatasetLoaderService initialized with default datasets due to empty config."
            )

        logger.info("DatasetLoaderService initialized.")

    def __open_config(self) -> dict:
        """
        Opens the configuration file and returns its content.
        Returns:
            dict: The content of the configuration file.
        Raises:
            OSError: If there is an error reading the configuration file.
            json.JSONDecodeError: If the configuration file is not a valid JSON.
        """
        try:
            with open(self.__config_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error opening config file: {e}")
            raise

    def _register_default_datasets(self):
        """Register default fallback datasets when config is missing or empty."""
        try:
            for dataset_name, dataset_info in DEFAULT_DATASETS.items():
                self.load(dataset_name=dataset_name, save_parameters=dataset_info)

            logger.debug(f"Default datasets available: {list(DEFAULT_DATASETS.keys())}")

        except Exception as e:
            logger.error(f"Failed to register default datasets: {e!s}")
            raise RuntimeError(f"Failed to register default datasets: {e!s}") from e

    def __save_new_dataset(
        self, dataset_name: str, save_parameters: dict | None = None
    ):
        """
        Saves a new dataset to the configuration file.
        Args:
            dataset_name (str): The name of the dataset.
        Raises:
            OSError: If there is an error writing to the configuration file.
        """

        if not save_parameters:
            save_parameters = {}

        datasets_config = self.__open_config()

        new_entry = {
            "name": dataset_name,
            **save_parameters,
        }

        datasets_config[dataset_name] = new_entry

        try:
            with open(self.__config_path, "w") as f:
                json.dump(datasets_config, f, indent=4)
            logger.debug(f"Dataset '{dataset_name}' saved successfully.")
        except OSError as e:
            logger.error(f"Error saving dataset configuration: {e}")
            raise

    def load(
        self,
        dataset_name,
        config: str = "",
        save_parameters: dict | None = None,
        cache: bool = True,
    ):
        """
        Loads a dataset by its name.
        Args:
            dataset_name (str): The name of the dataset to load.
            config (str): The configuration name for the dataset (optional).
        Returns:
            dict: The loaded dataset.
        Raises:
            ValueError: If the dataset name is empty or if the dataset cannot be found.
            OSError: If there is an error loading the dataset.
            json.JSONDecodeError: If the configuration file is not a valid JSON.
        """

        if not save_parameters:
            save_parameters = {}
        if dataset_name in DIAGNOSTIC_DATASETS:
            cache = False

        datasets_config = self.__open_config()

        # Check if the dataset is in the cache
        cache_flag = False
        if dataset_name in datasets_config:
            cache_flag = True
            logger.debug(f"Loading dataset '{dataset_name}' from cache.")

        if not cache_flag:
            logger.debug(
                f"Dataset '{dataset_name}' not found in cache. Loading from Hugging Face Hub."
            )

        try:
            dataset = load_dataset(dataset_name)
        except Exception:
            logger.warning(
                f"Initial load attempt for dataset '{dataset_name}' failed. Trying with config '{config}'"
            )
            try:
                dataset = load_dataset(dataset_name, config)
            except ValueError as e:
                # Handle case where config name is missing
                if "Config name is missing" in str(e):
                    logger.warning(
                        f"Dataset '{dataset_name}' requires a config. Attempting to load with default config."
                    )
                    try:
                        # Parse available configs from error message
                        match = re.search(
                            r"Please pick one among the available configs: \[(.*?)\]",
                            str(e),
                        )
                        if match:
                            configs = [
                                c.strip().strip("'\"")
                                for c in match.group(1).split(",")
                            ]
                            logger.info(
                                f"Available configs for '{dataset_name}': {configs}"
                            )

                            # If a config was provided and it's in the available configs, use it
                            # This handles BeIR-style datasets where config names match split names
                            if config and config in configs:
                                default_config = config
                                logger.info(
                                    f"Using provided config '{config}' which matches an available config"
                                )
                            else:
                                # Otherwise use the last config (latest version)
                                default_config = configs[-1] if configs else None

                            if default_config:
                                dataset = load_dataset(
                                    dataset_name, str(default_config)
                                )

                                # Update save_parameters to include the config
                                save_parameters["config"] = default_config

                                if cache:
                                    self.__dataset = dataset
                                logger.debug(
                                    f"Dataset '{dataset_name}' with config '{default_config}' loaded successfully."
                                )
                            else:
                                logger.error(
                                    f"Could not determine default config for dataset '{dataset_name}': {e}"
                                )
                                raise
                        else:
                            logger.error(
                                f"Could not parse available configs from error message for dataset '{dataset_name}': {e}"
                            )
                            raise
                    except ValueError as ve:
                        # Re-raise the original ValueError if we can't handle it
                        logger.error(
                            f"Error loading dataset '{dataset_name}' with default config: {ve}"
                        )
                        raise
                    except Exception as config_error:
                        logger.error(
                            f"Unexpected error loading dataset '{dataset_name}' with default config: {config_error}"
                        )
                        raise
                else:
                    logger.error(f"Error loading dataset '{dataset_name}': {e}")
                    raise
            except Exception as e:
                logger.error(f"Error loading dataset '{dataset_name}': {e}")
                raise

        if cache:
            self.__dataset = dataset
        logger.info(f"Dataset '{dataset_name}' loaded successfully.")

        if not cache_flag and cache:
            try:
                task_categories = self.extract_task_categories(dataset_name)

                save_parameters["task_types"] = task_categories
                save_parameters["splits"] = list(dataset.keys())
                save_parameters["num_samples"] = sum(
                    len(dataset[split]) for split in dataset
                )
                self.__save_new_dataset(dataset_name, save_parameters)
                logger.debug(f"Dataset '{dataset_name}' saved to cache.")
            except OSError as e:
                logger.error(f"Error saving dataset '{dataset_name}' to cache: {e}")
                raise

        return dataset

    def get_dataset(self):
        """
        Returns the loaded dataset.
        Returns:
            dict: The loaded dataset.
        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self.__dataset is None:
            logger.error("Dataset is not loaded. Call load() first.")
            raise ValueError("Dataset is not loaded. Call load() first.")

        return self.__dataset

    def extract_task_categories(self, dataset_name: str) -> list:
        """
        Extracts task_categories from the YAML front matter of a README file.
        Args:
            dataset_name (str): The name of the dataset.
        Returns:
            list: List of task categories, empty list if not found.
        """
        try:
            README = hf_hub_download(
                repo_id=dataset_name, filename="README.md", repo_type="dataset"
            )
            with open(README, "r") as f:
                readme_content = f.read()
        except Exception as e:
            logger.error(f"Error reading README.md for dataset '{dataset_name}': {e}")
            return []

        try:
            # Extract YAML front matter between --- delimiters
            yaml_match = re.match(r"^---\s*\n(.*?)\n---", readme_content, re.DOTALL)
            if not yaml_match:
                logger.debug("No YAML front matter found in README")
                return []

            yaml_content = yaml_match.group(1)
            metadata = yaml.safe_load(yaml_content)

            if isinstance(metadata, dict) and "task_categories" in metadata:
                task_categories = metadata["task_categories"]
                if isinstance(task_categories, list):
                    return task_categories
                else:
                    return []
            else:
                return []

        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML front matter: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error extracting task categories: {e}")
            return []

    def close(self):
        """
        Closes the dataset loader service.
        This method can be used to perform any cleanup if necessary.
        """
        self.__dataset = None
        logger.info("DatasetLoaderService has been closed.")

    ### CRUD ###
    def index(self, include_metrics=False, include_diagnostics: bool = False):
        """
        Returns a list of all datasets in the system.
        Args:
            include_metrics (bool): Whether to include dataset metrics in the response.
            include_diagnostics (bool): Whether to include dataset diagnostics in the response.
        Returns:
            list[str]: List of dataset names.
        Raises:
            RuntimeError: If there is an error retrieving datasets.
        """
        datasets = {}
        try:
            datasets = self.__open_config()
            if include_diagnostics:
                datasets.update(DIAGNOSTIC_DATASETS)

            if include_metrics:
                for dataset_name, dataset in datasets.items():
                    dataset["metrics"] = self.get_metrics(dataset_name)
            return datasets
        except Exception as e:
            logger.error(f"Error retrieving datasets: {e}")
            raise RuntimeError(f"Failed to retrieve datasets: {e}")

    def show(
        self,
        dataset_name: str,
        include_metrics: bool = False,
        include_diagnostics: bool = True,
    ) -> dict:
        """
        Returns the details of a specific dataset.
        Args:
            dataset_name (str): The name of the dataset to retrieve.
            include_metrics (bool): Whether to include dataset metrics in the response.
            include_diagnostics (bool): Whether to include dataset diagnostics in the response.
        Returns:
            dict: Details of the specified dataset.
        Raises:
            RuntimeError: If there is an error retrieving the dataset.
        """
        dataset = {dataset_name: {}}
        try:
            datasets_config = self.__open_config()
            if include_diagnostics:
                datasets_config.update(DIAGNOSTIC_DATASETS)

            if dataset_name not in datasets_config:
                # Try to load from Hugging Face and save to config
                try:
                    self.load(dataset_name=dataset_name, cache=True)
                    # Reload config after loading the dataset
                    datasets_config = self.__open_config()
                    if dataset_name in datasets_config:
                        dataset[dataset_name] = datasets_config[dataset_name]
                    else:
                        logger.warning(
                            f"Dataset '{dataset_name}' was loaded but not found in config."
                        )
                        return {}
                except Exception as load_error:
                    logger.warning(
                        f"Failed to load dataset '{dataset_name}' from Hugging Face: {load_error}"
                    )
                    return {}
            else:
                dataset[dataset_name] = datasets_config[dataset_name]

            if include_metrics:
                metrics = self.get_metrics(dataset_name)
                # logger.info(f"Metrics for dataset '{dataset_name}': {metrics}")
                dataset[dataset_name]["metrics"] = metrics

            return dataset
        except Exception as e:
            logger.error(f"Error retrieving dataset '{dataset_name}': {e}")
            raise RuntimeError(f"Failed to retrieve dataset '{dataset_name}': {e}")

    def config(self, dataset_name: str):
        """
        Returns the configuration of a specific dataset.
        Args:
            dataset_name (str): The name of the dataset to retrieve configuration for.
        Returns:
            dict: Configuration of the specified dataset.
        Raises:
            RuntimeError: If there is an error retrieving the dataset configuration.
        """
        try:
            return get_dataset_config_info(dataset_name)
        except Exception as e:
            logger.error(f"Error retrieving config for dataset '{dataset_name}': {e}")
            raise RuntimeError(
                f"Failed to retrieve config for dataset '{dataset_name}': {e}"
            )

    def config_splits(self, dataset_name: str):
        """
        Returns the splits of a specific dataset.
        Args:
            dataset_name (str): The name of the dataset to retrieve splits for.
        Returns:
            list: List of splits in the specified dataset.
        Raises:
            RuntimeError: If there is an error retrieving the dataset splits.
        """
        try:
            dataset = self.show(dataset_name)
            return dataset[dataset_name].get("splits", [])
        except Exception as e:
            logger.error(f"Error retrieving splits for dataset '{dataset_name}': {e}")
            raise RuntimeError(
                f"Failed to retrieve splits for dataset '{dataset_name}': {e}"
            )

    def get_metrics(self, dataset_name: str):
        """
        Returns the metrics for a specific dataset.
        Args:
            dataset_name (str): The name of the dataset to retrieve metrics for.
        Returns:
            dict: Metrics for the specified dataset.
        Raises:
            RuntimeError: If there is an error retrieving the metrics.
        """

        try:
            from app.controllers.metric_controller import MetricController

            dataset = self.show(dataset_name)
            task_types = (
                dataset[dataset_name].get("task_types", [])
                if dataset
                else self.extract_task_categories(dataset_name)
            )
            all_tasks = MetricController.get_all_metrics()

            # If a dataset has no metrics specified, return all available metrics
            if not task_types:
                return all_tasks

            metrics = MetricController.find_metrics_by_task(task_types)
            if not metrics:
                return all_tasks

            return metrics

        except Exception as e:
            logger.error(f"Error retrieving metrics for dataset '{dataset_name}': {e}")
            raise RuntimeError(
                f"Failed to retrieve metrics for dataset '{dataset_name}': {e}"
            )

    def find_datasets_by_tasks(
        self, task_types: list, include_metrics: bool = False
    ) -> dict:
        """
        Finds datasets that are associated with specific task types.
        Args:
            task_types (list): A list of task types to filter datasets by.
            include_metrics (bool): Whether to include metrics in the dataset information.
        Returns:
            list: A list of datasets that match the specified task types.
        """
        try:
            datasets = self.__open_config()

            matching_datasets = {}
            for dataset_name, dataset in datasets.items():
                if not set(dataset.get("task_types", [])) & set(task_types):
                    continue

                if include_metrics:
                    dataset["metrics"] = self.get_metrics(dataset_name)

                matching_datasets[dataset_name] = dataset
            return matching_datasets
        except Exception as e:
            logger.error(f"Error finding datasets by tasks '{task_types}': {e}")
            raise RuntimeError(f"Failed to find datasets by tasks '{task_types}': {e}")
