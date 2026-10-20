import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from app.config.paths import CONFIG_DIR
from app.services.logging_service import LoggingService
from app.services.model_architecture import (
    get_architecture_mappings,
    get_architecture_patterns,
)

# Set up module-specific logger
logger = LoggingService.setup_module_logger(__name__)

DEFAULT_MODELS = [
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    "Qwen/Qwen2-1.5B-Instruct",
]


class ModelLoaderService:
    def __init__(self, config_path=None):
        """
        Initializes the ModelLoader with the directory for models, tokenizers, and the configuration file path.
        Args:
            models_dir (str): Directory where models are stored.
            tokenizer_dir (str): Directory where tokenizers are stored.
            config_path (str): Path to the configuration file containing model information.
        Raises:
            FileNotFoundError: If the specified directories or configuration file do not exist.
        """
        if not config_path:
            self.__config_path = os.path.join(CONFIG_DIR, "models.json")
        else:
            self.__config_path = config_path
        self.__model = None
        self.__model_name = None
        self.__model_type = None
        self.__tokenizer = None
        self.__model_parameters = {}
        self.__tokenizer_parameters = {}

        # Check if config file exists, if not create it
        if not os.path.exists(self.__config_path):
            logger.warning(
                f"Config file '{self.__config_path}' does not exist. Creating it now."
            )
            os.makedirs(os.path.dirname(self.__config_path), exist_ok=True)

            # Write an empty dict to initialize the config file
            with open(self.__config_path, "w") as f:
                json.dump({}, f, indent=4)
                logger.debug(
                    f"Created new config file at '{self.__config_path}' with empty models object."
                )

            self._register_default_models()
            logger.info("ModelLoaderService initialized with default models.")
            return

        try:
            with open(self.__config_path, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.critical(f"Error loading config file {self.__config_path}: {e}")
            return

        if not config or not config.items():
            self._register_default_models()
            logger.debug(
                "ModelLoaderService initialized with default models due to empty config."
            )

        logger.info("ModelLoaderService initialized.")

    def __convert_parameter_types(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Converts string representations of PyTorch types to actual PyTorch objects.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameters that may contain string representations

        Returns:
            Dict[str, Any]: Dictionary with converted parameters
        """
        converted_params = parameters.copy()

        # Define mapping of string representations to actual PyTorch objects
        torch_dtype_mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
            "torch.complex64": torch.complex64,
            "torch.complex128": torch.complex128,
            # Also support without "torch." prefix
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }

        for key, value in converted_params.items():
            if not isinstance(value, str):
                continue

            if (
                key in ["torch_dtype", "dtype"] and value in torch_dtype_mapping
            ):  # Handle torch_dtype parameters
                converted_params[key] = torch_dtype_mapping[value]
                logger.debug(
                    f"Converted parameter '{key}': '{value}' -> {torch_dtype_mapping[value]}"
                )
            elif key == "device":  # Handle device parameters
                try:
                    # Try to create a torch device from the string
                    converted_params[key] = torch.device(value)
                    logger.debug(
                        f"Converted parameter '{key}': '{value}' -> {torch.device(value)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not convert device parameter '{value}': {e}. Keeping as string."
                    )
            elif key in [
                "trust_remote_code",
                "use_cache",
                "output_attentions",
                "output_hidden_states",
            ]:  # Handle boolean-like strings
                if value.lower() in ["true", "false"]:
                    converted_params[key] = value.lower() == "true"
                    logger.debug(
                        f"Converted parameter '{key}': '{value}' -> {value.lower() == 'true'}"
                    )

        return converted_params

    def __open_config(self):
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

    def _register_default_models(self):
        """
        Registers default models in the configuration file.
        Args:
            register (bool): If True, registers default models.
        Raises:
            OSError: If there is an error writing to the configuration file.
        """
        try:
            for model_name in DEFAULT_MODELS:
                model_type = self.__determine_model_type(model_name)
                self.__save_model_to_config(model_name, model_type)
        except Exception as e:
            logger.error(f"Error registering default models: {e}")
            raise OSError(f"Failed to register default models: {e!s}")

    def __save_model_to_config(self, model_name: str, model_type: str | None = None):
        """
        Saves a new model and its tokenizer to the configuration file.
        Args:
            model_name (str): The name of the model.
            model_type (str): The type of the model (encoder-only, encoder-decoder, decoder-only).
        Raises:
            OSError: If there is an error writing to the configuration file.
        """
        config = AutoConfig.from_pretrained(model_name)
        models_config = self.__open_config()

        new_entry = {
            "name": model_name,
            "size": int(
                self.__get_model_total_size(model_name) / (1024**2)
            ),  # Convert to MB
        }
        if model_type:
            new_entry["type"] = model_type
        new_entry = {**new_entry, **config.to_dict()}

        models_config[model_name] = new_entry

        try:
            with open(self.__config_path, "w") as f:
                json.dump(models_config, f, indent=4)
            logger.debug(
                f"Model '{model_name}' saved successfully with type '{model_type}'."
            )
        except OSError as e:
            logger.error(f"Error saving model configuration: {e}")
            raise

    def __construct_auto_model_name(self, architecture_name: str) -> str:
        """
        Constructs AutoModel class name from architecture name using regex patterns.

        :param architecture_name: The architecture name to convert.
        :type architecture_name: str
        :return: The constructed AutoModel class name (e.g., "AutoModelForCausalLM") or None
        :rtype: str
        """
        # Common patterns for converting architecture names to AutoModel names
        patterns = [
            # Match patterns like "Qwen2ForCausalLM" -> "AutoModelForCausalLM"
            (r"^[A-Za-z0-9]+For([A-Za-z]+)$", r"AutoModelFor\1"),
            # Match patterns like "BertForSequenceClassification" -> "AutoModelForSequenceClassification"
            (r"^[A-Za-z0-9]+For([A-Za-z]+Classification)$", r"AutoModelFor\1"),
            # Match patterns like "T5ForConditionalGeneration" -> "AutoModelForSeq2SeqLM"
            (r"^[A-Za-z0-9]+ForConditionalGeneration$", r"AutoModelForSeq2SeqLM"),
            # Match patterns like "GPT2LMHeadModel" -> "AutoModelForCausalLM"
            (r"^[A-Za-z0-9]+LMHeadModel$", r"AutoModelForCausalLM"),
            # Match patterns like "BertModel" -> "AutoModel"
            (r"^[A-Za-z0-9]+Model$", r"AutoModel"),
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, architecture_name)
            if match:
                auto_model_name = re.sub(pattern, replacement, architecture_name)
                logger.debug(
                    f"Constructed auto model name '{auto_model_name}' from architecture '{architecture_name}'"
                )
                return auto_model_name

        logger.debug(
            f"Could not construct auto model name for architecture '{architecture_name}'"
        )
        return None

    def __construct_tokenizer_class_name(self, architecture_name: str) -> str:
        """
        Constructs tokenizer class name from architecture name.

        :param architecture_name: The architecture name to convert.
        :type architecture_name: str
        :return: The constructed tokenizer class name (e.g., "BertTokenizer") or None
        :rtype: str
        """
        # Common patterns for converting architecture names to tokenizer names
        patterns = [
            # Extract base model name patterns
            (r"^([A-Za-z0-9]+)(?:For[A-Za-z]+|Model|LMHeadModel)$", r"\1Tokenizer"),
            # Special cases for models with different tokenizer naming
            (r"^(T5).*$", r"\1Tokenizer"),
            (r"^(BERT|Bert).*$", r"BertTokenizer"),
            (r"^(RoBERTa|Roberta).*$", r"RobertaTokenizer"),
            (r"^(GPT2|Gpt2).*$", r"GPT2Tokenizer"),
            (r"^(DistilBert).*$", r"DistilBertTokenizer"),
            (r"^(Albert).*$", r"AlbertTokenizer"),
            (r"^(Electra).*$", r"ElectraTokenizer"),
            (r"^(Qwen2?).*$", r"Qwen2Tokenizer"),
            (r"^(Llama).*$", r"LlamaTokenizer"),
            (r"^(Mistral).*$", r"LlamaTokenizer"),  # Mistral often uses Llama tokenizer
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, architecture_name, re.IGNORECASE)
            if match:
                tokenizer_name = re.sub(
                    pattern, replacement, architecture_name, flags=re.IGNORECASE
                )
                logger.debug(
                    f"Constructed tokenizer name '{tokenizer_name}' from architecture '{architecture_name}'"
                )
                return tokenizer_name

        # Fallback: try adding "Tokenizer" to base name
        base_name = re.sub(r"(For[A-Za-z]+|Model|LMHeadModel)$", "", architecture_name)
        if base_name != architecture_name:
            tokenizer_name = f"{base_name}Tokenizer"
            logger.debug(
                f"Fallback tokenizer name '{tokenizer_name}' from architecture '{architecture_name}'"
            )
            return tokenizer_name

        logger.debug(
            f"Could not construct tokenizer name for architecture '{architecture_name}'"
        )
        return None

    def __find_auto_class(self, model_name: str):
        """
        Finds the appropriate model class based on the model architecture.

        :param model_name: The name or path of the model.
        :type model_name: str
        :return: Tuple containing the model class and tokenizer class, or AutoModel and AutoTokenizer as fallback.
        :rtype: tuple

        :raises: AttributeError: If the architecture class cannot be found in transformers.
        """
        try:
            config = AutoConfig.from_pretrained(model_name)
            architecture = getattr(config, "architectures", None)

            if not architecture or not isinstance(architecture, (list, tuple)):
                logger.warning(
                    f"Model '{model_name}' does not have a valid architecture defined. Using AutoModel."
                )
                return AutoModel, AutoTokenizer

            architecture_name = architecture[0]
            logger.debug(
                f"Found architecture: {architecture_name} for model '{model_name}'"
            )

            model_class = AutoModel
            tokenizer_class = AutoTokenizer

            # Try to get the architecture class directly from transformers
            try:
                import transformers

                architecture_class = getattr(transformers, architecture_name, None)
                if architecture_class is not None:
                    logger.debug(
                        f"Using architecture class '{architecture_name}' for model '{model_name}'"
                    )
                    model_class = architecture_class
            except AttributeError:
                pass

            # If AutoModel is still the default, try to find a specific class
            if model_class == AutoModel:
                auto_model_name = self.__construct_auto_model_name(architecture_name)
                if auto_model_name:
                    try:
                        auto_model_class = getattr(transformers, auto_model_name, None)
                        if auto_model_class is not None:
                            logger.debug(
                                f"Using constructed AutoModel class '{auto_model_name}' for model '{model_name}'"
                            )
                            model_class = auto_model_class
                    except AttributeError:
                        pass

            # Try to find specific tokenizer class
            tokenizer_class_name = self.__construct_tokenizer_class_name(
                architecture_name
            )
            if tokenizer_class_name:
                try:
                    specific_tokenizer_class = getattr(
                        transformers, tokenizer_class_name, None
                    )
                    if specific_tokenizer_class is not None:
                        logger.debug(
                            f"Using specific tokenizer class '{tokenizer_class_name}' for model '{model_name}'"
                        )
                        tokenizer_class = specific_tokenizer_class
                except AttributeError:
                    logger.warning(
                        f"Specific tokenizer class '{tokenizer_class_name}' not found. Using AutoTokenizer."
                    )

            # Check for tokenizer class config
            if tokenizer_class == AutoTokenizer:
                tokenizer_class_config = getattr(config, "tokenizer_class", None)
                if tokenizer_class_config:
                    try:
                        config_tokenizer_class = getattr(
                            transformers, tokenizer_class_config, None
                        )
                        if config_tokenizer_class is not None:
                            logger.debug(
                                f"Using tokenizer class from config: '{tokenizer_class_config}' for model '{model_name}'"
                            )
                            tokenizer_class = config_tokenizer_class
                    except AttributeError:
                        logger.warning(
                            f"Tokenizer class '{tokenizer_class_config}' from config not found. Using AutoTokenizer."
                        )

            # Log final choices
            if model_class == AutoModel:
                logger.debug(
                    f"No specific model class found for '{architecture_name}', using AutoModel for '{model_name}'"
                )
            if tokenizer_class == AutoTokenizer:
                logger.debug(
                    f"No specific tokenizer class found for '{architecture_name}', using AutoTokenizer for '{model_name}'"
                )

            logger.info(
                f"Model class: {model_class.__name__}, Tokenizer class: {tokenizer_class.__name__} for model '{model_name}'"
            )
            return model_class, tokenizer_class

        except Exception as e:
            logger.warning(
                f"Error finding architecture for model '{model_name}': {e}. Using AutoModel as fallback."
            )
            return AutoModel, AutoTokenizer

    def __load_class(self, class_name: str, is_model: bool = True):
        """
        Dynamically loads a class from the transformers library.

        :param class_name: The name of the class to load.
        :type class_name: str
        :param is_model: If True, loads a model class; otherwise, loads a tokenizer class.
        :type is_model: bool
        :return: The loaded class.
        :rtype: type

        :raises: ImportError: If the class cannot be found in transformers.
        """
        try:
            import transformers

            cls = getattr(transformers, class_name, None)
            if cls is None:
                raise ImportError(f"Class '{class_name}' not found in transformers.")
            return cls
        except ImportError as e:
            logger.error(f"Error loading class '{class_name}': {e}")
            raise

    def __determine_model_type(self, model_name: str) -> str:
        """
        Determine the model type based on the model architecture.
        Args:
            model_name (str): The name or path of the model.
        Returns:
            str: One of 'encoder-only', 'encoder-decoder', or 'decoder-only'
        """
        try:
            config = AutoConfig.from_pretrained(model_name)

            # Check if it's an encoder-decoder model
            if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
                return "encoder-decoder"

            # Get architecture mappings from configuration
            (
                encoder_only_architectures,
                encoder_decoder_architectures,
                decoder_only_architectures,
            ) = get_architecture_mappings()
            decoder_only_patterns, encoder_decoder_patterns = (
                get_architecture_patterns()
            )

            architecture = getattr(config, "architectures", [])
            if architecture:
                arch_name = architecture[0]

                # Check for encoder-only patterns
                if any(
                    enc_arch in arch_name for enc_arch in encoder_only_architectures
                ):
                    return "encoder-only"

                # Check for encoder-decoder patterns
                if any(
                    enc_dec_arch in arch_name
                    for enc_dec_arch in encoder_decoder_architectures
                ):
                    return "encoder-decoder"

                # Check for decoder-only patterns
                if any(
                    dec_arch in arch_name for dec_arch in decoder_only_architectures
                ):
                    return "decoder-only"

                # Check for common decoder-only patterns
                if any(pattern in arch_name for pattern in decoder_only_patterns):
                    return "decoder-only"

                # Check for common encoder-decoder patterns
                if any(pattern in arch_name for pattern in encoder_decoder_patterns):
                    return "encoder-decoder"

                # If no match found, raise an error
                logger.error(
                    f"Unknown model architecture '{arch_name}' for model '{model_name}'."
                )
                raise ValueError(f"Unsupported model architecture: {arch_name}")

            # If no architecture information available, raise an error
            logger.error(f"No architecture information found for model '{model_name}'.")
            raise ValueError(
                f"Cannot determine model type: no architecture information available for {model_name}"
            )

        except ValueError:
            # Re-raise ValueError
            raise
        except Exception as e:
            logger.error(f"Error determining model type for {model_name}: {e}")
            raise ValueError(f"Failed to determine model type for {model_name}: {e!s}")

    def __get_model_total_size(self, model_name: str) -> int | None:
        """Get the total size of a model from Hugging Face Hub API."""
        try:
            url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                total_size = model_info.get("usedStorage")
                logger.info(
                    f"Total size for model '{model_name}': {total_size / (1024**2):.2f} MB"
                )
                return total_size
            return None
        except Exception as e:
            logger.warning(f"Could not fetch model size for {model_name}: {e}")
            return None

    def __log_download_progress(self, model_name: str, stop_event: threading.Event):
        """Background thread to log download progress periodically."""
        cache_dir = Path.home().joinpath(".cache", "huggingface", "hub")
        if not cache_dir.exists():
            logger.warning(
                f"Cache directory '{cache_dir}' does not exist. Download progress logging will not work."
            )
            return

        start_time = time.time()
        total_expected_size = self.__get_model_total_size(model_name)

        while not stop_event.is_set():
            elapsed = int(time.time() - start_time)

            # Check if model files are being downloaded
            model_dirs = list(cache_dir.glob(f"*{model_name.replace('/', '--')}*"))
            if model_dirs:
                current_size = sum(
                    f.stat().st_size
                    for d in model_dirs
                    for f in d.rglob("*")
                    if f.is_file()
                )
                current_mb = current_size / (1024**2)

                if total_expected_size:
                    total_mb = total_expected_size / (1024**2)
                    progress_pct = (current_size / total_expected_size) * 100
                    logger.info(
                        f"Downloading '{model_name}'... {current_mb:.1f}/{total_mb:.1f} MB ({progress_pct:.1f}%) - {elapsed}s elapsed"
                    )
                else:
                    logger.info(
                        f"Downloading '{model_name}'... {current_mb:.1f} MB downloaded - {elapsed}s elapsed"
                    )
            else:
                logger.info(f"Downloading '{model_name}'... ({elapsed}s elapsed)")

            stop_event.wait(10)  # Log every 10 seconds

    def load(
        self,
        model_name: str,
        model_parameters: dict | None = None,
        tokenizer_parameters: dict | None = None,
        cache: bool = True,
    ) -> tuple:
        """
        Loads a model and its tokenizer based on the provided model name.
        Args:
            model_name (str): The name of the model to load.
            model_parameters (dict): Additional parameters for loading the model.
            tokenizer_parameters (dict): Additional parameters for loading the tokenizer.
        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        Raises:
            OSError: If there is an error reading or writing files.
        Example:
            >>> model_loader = ModelLoader()
            >>> model, tokenizer = model_loader.load("google-bert/bert-base-uncased")
        """
        models_config = self.__open_config()

        if not model_parameters:
            model_parameters = {}
        if not tokenizer_parameters:
            tokenizer_parameters = {}

        # Check if the model is in the cache
        cache_flag = False
        if model_name in models_config:
            cache_flag = True
            logger.debug(f"Loading model '{model_name}' from cache.")

        if not cache_flag:
            logger.debug(
                f"Model '{model_name}' not found in cache. Loading from Hugging Face Hub."
            )

        model_architecture = model_parameters.pop("architecture", None)
        tokenizer_architecture = tokenizer_parameters.pop("architecture", None)

        # Convert string parameters to proper PyTorch types
        parsed_model_parameters = self.__convert_parameter_types(model_parameters)
        parsed_tokenizer_parameters = self.__convert_parameter_types(
            tokenizer_parameters
        )

        # Check if model with parameters is already self loaded and if tokenizer parameters are the same
        if (
            self.__model_name == model_name
            and self.__model
            and self.__model_parameters == parsed_model_parameters
            and self.__tokenizer
            and self.__tokenizer_parameters == parsed_tokenizer_parameters
        ):
            logger.debug(
                f"Model '{model_name}' with specified parameters is already loaded."
            )
            return self.__model, self.__tokenizer

        if cache:
            self.__model_parameters = parsed_model_parameters
            self.__tokenizer_parameters = parsed_tokenizer_parameters

        # Get the model and tokenizer architecture class based on the model config if not specified
        found_model_class, found_tokenizer_class = self.__find_auto_class(model_name)
        if not model_architecture:
            logger.debug(
                f"Using architecture '{found_model_class.__name__}' for model '{model_name}'."
            )
            model_class = found_model_class
        else:
            model_class = self.__load_class(model_architecture, is_model=True)

        if not tokenizer_architecture:
            logger.debug(
                f"Using architecture '{found_tokenizer_class.__name__}' for tokenizer of model '{model_name}'."
            )
            tokenizer_class = found_tokenizer_class
        else:
            tokenizer_class = self.__load_class(tokenizer_architecture, is_model=False)

        # Start progress monitoring if not in cache
        progress_stop_event = None
        progress_thread = None
        if not cache_flag:
            progress_stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=self.__log_download_progress,
                args=(model_name, progress_stop_event),
            )
            progress_thread.daemon = True
            progress_thread.start()
            logger.info(f"Starting download of '{model_name}'...")

        # Model specific parameter adjustments (pszemraj/long-t5-tglobal-base-16384-book-summary)
        if "pszemraj/long-t5-tglobal-base-16384-book-summary" in model_name:
            parsed_model_parameters = {
                "use_safetensors": False,
                **parsed_model_parameters,
            }

        # Add default parameters to handle meta tensors properly and avoid device placement errors
        # These defaults are only applied if not already specified by the user
        model_defaults = {
            "device_map": "auto",  # Automatically handle device placement for meta tensors
            "low_cpu_mem_usage": True,  # Use efficient loading to handle meta tensors correctly
        }
        # Only add defaults if not already specified
        for key, value in model_defaults.items():
            if key not in parsed_model_parameters:
                parsed_model_parameters[key] = value

        # Try loading tokenizer with specific class
        try:
            logger.info(
                f"Loading tokenizer for '{model_name}' with parameters: {parsed_tokenizer_parameters}"
            )
            tokenizer_loaded = model_loaded = False
            try:
                loaded_tokenizer = tokenizer_class.from_pretrained(
                    model_name, **parsed_tokenizer_parameters
                )
                if cache:
                    self.__tokenizer = loaded_tokenizer
                tokenizer_loaded = True
                logger.debug(f"Tokenizer '{model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading tokenizer '{model_name}': {e}")
                logger.warning(
                    f"Falling back to AutoTokenizer for model '{model_name}'."
                )

            # Try loading model with specific class
            try:
                logger.info(
                    f"Loading model '{model_name}' with parameters: {parsed_model_parameters}"
                )
                loaded_model = model_class.from_pretrained(
                    model_name, **parsed_model_parameters
                )
                if cache:
                    self.__model = loaded_model
                model_loaded = True
                logger.debug(f"Model '{model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {e}")
                logger.warning(f"Falling back to AutoModel for model '{model_name}'.")
        finally:
            # Stop progress monitoring if it was started
            if progress_stop_event:
                progress_stop_event.set()
            if progress_thread:
                progress_thread.join(timeout=1)
                if not cache_flag:
                    logger.info(f"Download of '{model_name}' completed.")

        # Fallback to Auto classes if specific classes failed
        if not tokenizer_loaded:
            try:
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **parsed_tokenizer_parameters
                )
                if cache:
                    self.__tokenizer = loaded_tokenizer
                tokenizer_loaded = True
                logger.debug(f"AutoTokenizer for '{model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading AutoTokenizer for '{model_name}': {e}")
                raise OSError(f"Failed to load AutoTokenizer for '{model_name}': {e}")

        if not model_loaded:
            try:
                loaded_model = AutoModel.from_pretrained(
                    model_name, **parsed_model_parameters
                )
                if cache:
                    self.__model = loaded_model
                model_loaded = True
                logger.debug(f"AutoModel for '{model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading AutoModel for '{model_name}': {e}")
                raise OSError(f"Failed to load AutoModel for '{model_name}': {e}")

        # Determine model type after loading
        if tokenizer_loaded and model_loaded:
            loaded_model_type = self.__determine_model_type(model_name)
            if cache:
                self.__model_name = model_name
                self.__model_type = loaded_model_type
            logger.debug(
                f"Model '{model_name}' loaded successfully with type: {loaded_model_type}."
            )
        else:
            raise OSError(
                f"Failed to load model '{model_name}': tokenizer_loaded={tokenizer_loaded}, model_loaded={model_loaded}"
            )

        # If the model is not local, save it to the cache & update the models.json config
        if not cache_flag:
            try:
                # Determine model type before saving
                self.__save_model_to_config(model_name, loaded_model_type)
                logger.debug(
                    f"Model '{model_name}' saved to cache with type '{loaded_model_type}'."
                )
            except OSError as e:
                logger.error(f"Error saving {model_name}: {e}")
                raise

        logger.info(
            f"Model '{model_name}' loaded successfully with type: {loaded_model_type} and tokenizer: {loaded_tokenizer.__class__.__name__}."
        )
        return loaded_model, loaded_tokenizer

    def get_model_tokenizer(self):
        """
        Returns the currently loaded model and tokenizer.
        Returns:
            tuple: A tuple containing the model and tokenizer.
        Raises:
            RuntimeError: If no model or tokenizer is loaded.
        """
        if self.__model is None or self.__tokenizer is None:
            logger.error("No model or tokenizer loaded. Please load a model first.")
            raise RuntimeError(
                "No model or tokenizer loaded. Please load a model first."
            )

        return self.__model, self.__tokenizer

    def get_model_name(self):
        """
        Returns the name of the currently loaded model.
        Returns:
            str: The name of the model.
        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.__model_name is None:
            logger.error("No model loaded. Please load a model first.")
            raise RuntimeError("No model loaded. Please load a model first.")

        return self.__model_name

    def get_model_type(self):
        """
        Returns the type of the currently loaded model.
        Returns:
            str: The type of the model (encoder-only, encoder-decoder, decoder-only).
        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.__model_type is None:
            logger.error("No model loaded. Please load a model first.")
            raise RuntimeError("No model loaded. Please load a model first.")

        return self.__model_type

    def close(self):
        """
        Closes the model and tokenizer, releasing any resources.
        """
        if self.__model is not None:
            del self.__model
            self.__model = None
            logger.debug("Model has been closed.")

        if self.__tokenizer is not None:
            del self.__tokenizer
            self.__tokenizer = None
            logger.debug("Tokenizer has been closed.")

        logger.info("ModelLoaderService has been closed.")

    ### CRUD ###
    def index(self):
        """
        Returns a list of all models in the system.
        Returns:
            list[str]: List of model names.
        Raises:
            RuntimeError: If there is an error retrieving models.
        """
        models = {}
        try:
            models = self.__open_config()
            logger.debug(f"Retrieved models: {models}")
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            raise RuntimeError(f"Failed to retrieve models: {e}")
        return models

    def show(self, model_name: str):
        """
        Returns the details of a specific model.
        Args:
            model_name (str): The name of the model to retrieve.
        Returns:
            dict: Details of the specified model.
        Raises:
            RuntimeError: If there is an error retrieving the model.
        """
        model = {model_name: {}}
        try:
            config = AutoConfig.from_pretrained(model_name)
            config_dict = config.to_dict()

            # Convert torch.dtype objects to strings for JSON serialization
            for key, value in config_dict.items():
                if isinstance(value, torch.dtype):
                    config_dict[key] = str(value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, torch.dtype):
                            value[nested_key] = str(nested_value)

            model[model_name] = config_dict
            logger.debug(f"Found model '{model_name}'")
            return model
        except Exception as e:
            logger.error(f"Error retrieving model '{model_name}': {e}")
            raise RuntimeError(f"Failed to retrieve model '{model_name}': {e}")

    def download_size(self, model_name: str) -> int | None:
        """
        Returns the total size of a model in bytes.
        Args:
            model_name (str): The name of the model to check.
        Returns:
            int: Total size of the model in bytes, or None if not found.
        """
        return self.__get_model_total_size(model_name)

    def auto_class(self, model_name: str) -> str | None:
        """
        Returns the AutoClass of a model.
        Args:
            model_name (str): The name of the model to check.
        Returns:
            str: The AutoClass of the model, or None if not found.
        """
        config = AutoConfig.from_pretrained(model_name)
        if config is None or "architectures" not in config:
            return None
        return config.get("architectures")[0]
