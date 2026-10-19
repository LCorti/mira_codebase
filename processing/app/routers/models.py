from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.registry import get_model_loader
from app.services.logging_service import LoggingService
from app.services.model_loader import ModelLoaderService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/models",
    tags=["models"],
)

ModelLoaderDep = Annotated[ModelLoaderService, Depends(get_model_loader)]


@router.get("", status_code=200)
async def index(model_loader: ModelLoaderDep):
    """
    Returns a list of all models in the system.
    Args:
        None
    Returns:
        list[str]: List of model names.
    Raises:
        RuntimeError: If there is an error retrieving models.
    """
    try:
        models = model_loader.index()
    except Exception as e:
        logger.error(f"Error retrieving models: {e}")
        raise RuntimeError(f"Failed to retrieve models: {e}")

    return models


@router.get("/show", status_code=200)
async def show(model_name: str, model_loader: ModelLoaderDep):
    """
    Returns the details of a specific model.
    Args:
        model_name (str): The name of the model to retrieve.
    Returns:
        dict: Details of the specified model.
    Raises:
        RuntimeError: If there is an error retrieving the model.
    """
    try:
        model = model_loader.show(model_name)
    except Exception as e:
        logger.error(f"Error retrieving model '{model_name}': {e}")
        raise RuntimeError(f"Failed to retrieve model '{model_name}': {e}")

    return model


@router.get("/download_size", status_code=200)
async def download_size(model_name: str, model_loader: ModelLoaderDep):
    """
    Returns the total size of a specific model in bytes.
    Args:
        model_name (str): The name of the model to check.
    Returns:
        dict: Total size of the model in bytes.
    Raises:
        RuntimeError: If there is an error retrieving the model size.
    """
    try:
        size = model_loader.download_size(model_name)
        logger.info(f"Model '{model_name}' size: {size} bytes")
        if size is None:
            return {"model": model_name, "size": 0}
    except Exception as e:
        logger.error(f"Error retrieving model '{model_name}' size: {e}")
        return {"model": model_name, "size": 0}

    return {"model": model_name, "size": size}


@router.get("/config", status_code=200)
async def config(model_name: str, model_loader: ModelLoaderDep):
    """
    Returns the configuration of a specific model.
    Args:
        model_name (str): The name of the model to check.
    Returns:
        dict: Configuration of the model.
    Raises:
        RuntimeError: If there is an error retrieving the model configuration.
    """
    try:
        config = model_loader.show(model_name)
    except Exception as e:
        logger.error(f"Error retrieving model '{model_name}' configuration: {e}")
        return {"model": model_name, "config": None}

    return config


@router.post("/load", status_code=200)
async def load_model(model_name: str, model_loader: ModelLoaderDep):
    """
    Loads a specific model into memory.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        dict: Confirmation message.
    Raises:
        RuntimeError: If there is an error loading the model.
    """
    try:
        model_loader.load(model_name)
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    return {"message": f"Model '{model_name}' loaded successfully."}
