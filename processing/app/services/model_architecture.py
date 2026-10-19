import json
import os

from app.config.paths import CONFIG_DIR


def load_model_architecture_config() -> dict:
    """
    Load the model architecture configuration from the JSON file.

    Returns:
        dict: Configuration containing architecture mappings and patterns.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        json.JSONDecodeError: If the configuration file is not valid JSON.
    """
    config_path = os.path.join(CONFIG_DIR, "model_architectures.json")

    with open(config_path, "r") as f:
        return json.load(f)


def get_architecture_mappings() -> tuple[list[str], list[str], list[str]]:
    """
    Get the architecture mappings for different model types.

    Returns:
        tuple: (encoder_only_architectures, encoder_decoder_architectures, decoder_only_architectures)
    """
    config = load_model_architecture_config()
    architectures = config.get("model_architectures", {})

    encoder_only = architectures.get("encoder_only", [])
    encoder_decoder = architectures.get("encoder_decoder", [])
    decoder_only = architectures.get("decoder_only", [])

    return encoder_only, encoder_decoder, decoder_only


def get_architecture_patterns() -> tuple[list[str], list[str]]:
    """
    Get the architecture patterns for identifying model types.

    Returns:
        tuple: (decoder_only_patterns, encoder_decoder_patterns)
    """
    config = load_model_architecture_config()
    patterns = config.get("architecture_patterns", {})

    decoder_only_patterns = patterns.get("decoder_only_patterns", [])
    encoder_decoder_patterns = patterns.get("encoder_decoder_patterns", [])

    return decoder_only_patterns, encoder_decoder_patterns


def get_fallback_type() -> str:
    """
    Get the fallback model type for unknown architectures.

    Returns:
        str: The fallback model type.
    """
    config = load_model_architecture_config()
    return config.get("fallback_type", "encoder-only")
