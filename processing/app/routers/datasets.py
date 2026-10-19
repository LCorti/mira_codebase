from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config.registry import get_dataset_loader
from app.services.dataset_loader import DatasetLoaderService
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)

DatasetLoaderDep = Annotated[DatasetLoaderService, Depends(get_dataset_loader)]


@router.get("/show-tasks", status_code=200)
async def show_readme_tasks(dataset_name: str, dataset_loader: DatasetLoaderDep):
    """
    Returns the details of all tasks.
    Args:
        dataset_name (str): The name of the dataset to retrieve.
    Returns:
        list[str]: Details of all tasks.
    Raises:
        RuntimeError: If there is an error retrieving the tasks.
    """
    try:
        return dataset_loader.extract_task_categories(dataset_name)
    except Exception as e:
        logger.error(f"Error retrieving tasks: {e}")
        raise RuntimeError(f"Failed to retrieve tasks: {e}")


@router.get("/config", status_code=200)
async def config(dataset_name: str, dataset_loader: DatasetLoaderDep):
    """
    Returns the configuration of a specific dataset.
    Args:
        dataset_name (str): The name of the dataset to check.
    Returns:
        dict: Configuration of the dataset.
    Raises:
        RuntimeError: If there is an error retrieving the dataset configuration.
    """
    try:
        config = dataset_loader.config(dataset_name)
        if not config:
            return {"dataset": dataset_name, "config": None}
    except Exception as e:
        logger.error(f"Error retrieving dataset '{dataset_name}' configuration: {e}")
        return {"dataset": dataset_name, "config": None}

    return config.__dict__


@router.get("/config/splits", status_code=200)
async def config_splits(dataset_name: str, dataset_loader: DatasetLoaderDep):
    """
    Returns the splits configuration of a specific dataset.
    Args:
        dataset_name (str): The name of the dataset to check.
    Returns:
        dict: Splits configuration of the dataset.
    Raises:
        RuntimeError: If there is an error retrieving the splits configuration.
    """
    try:
        splits_config = dataset_loader.config_splits(dataset_name)
        if not splits_config:
            return {"dataset": dataset_name, "splits": None}
    except Exception as e:
        logger.error(
            f"Error retrieving dataset '{dataset_name}' splits configuration: {e}"
        )
        return {"dataset": dataset_name, "splits": None}

    return splits_config


@router.get("", status_code=200)
async def index(
    include_metrics: Annotated[
        bool, Query(description="Whether to include dataset metrics in the response")
    ] = False,
    include_diagnostics: Annotated[
        bool,
        Query(description="Whether to include dataset diagnostics in the response"),
    ] = False,
    *,
    dataset_loader: DatasetLoaderDep,
):
    """
    Returns a list of all datasets in the system.
    Args:
        None
    Returns:
        list[str]: List of dataset names.
    Raises:
        RuntimeError: If there is an error retrieving datasets.
    """
    if include_metrics and type(include_metrics) is not bool:
        include_metrics = include_metrics.lower() in ["true", "1", "yes"]
    if include_diagnostics and type(include_diagnostics) is not bool:
        include_diagnostics = include_diagnostics.lower() in ["true", "1", "yes"]

    try:
        datasets = dataset_loader.index(
            include_metrics=include_metrics, include_diagnostics=include_diagnostics
        )
    except Exception as e:
        logger.error(f"Error retrieving datasets: {e}")
        raise RuntimeError(f"Failed to retrieve datasets: {e}")

    return datasets


@router.get("/show", status_code=200)
async def show(
    dataset_name: str,
    include_metrics: Annotated[
        bool, Query(description="Whether to include dataset metrics in the response")
    ] = False,
    include_diagnostics: Annotated[
        bool,
        Query(description="Whether to include dataset diagnostics in the response"),
    ] = False,
    *,
    dataset_loader: DatasetLoaderDep,
):
    """
    Returns the details of a specific dataset.
    Args:
        dataset_name (str): The name of the dataset to retrieve.
    Returns:
        dict: Details of the specified dataset.
    Raises:
        RuntimeError: If there is an error retrieving the dataset.
    """
    if include_metrics and type(include_metrics) is not bool:
        include_metrics = include_metrics.lower() in ["true", "1", "yes"]
    if include_diagnostics and type(include_diagnostics) is not bool:
        include_diagnostics = include_diagnostics.lower() in ["true", "1", "yes"]

    try:
        dataset = dataset_loader.show(
            dataset_name,
            include_metrics=include_metrics,
            include_diagnostics=include_diagnostics,
        )
    except Exception as e:
        logger.error(f"Error retrieving dataset '{dataset_name}': {e}")
        raise RuntimeError(f"Failed to retrieve dataset '{dataset_name}': {e}")

    return dataset


@router.post("/load", status_code=200)
async def load_dataset(dataset_name: str, dataset_loader: DatasetLoaderDep):
    """
    Loads a specific dataset into memory.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        dict: Confirmation message.
    Raises:
        RuntimeError: If there is an error loading the dataset.
    """
    try:
        _ = dataset_loader.load(dataset_name)
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_name}': {e}")
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    return {"message": f"Dataset '{dataset_name}' loaded successfully."}


@router.get("/metrics", status_code=200)
async def get_metrics(dataset_name: str, dataset_loader: DatasetLoaderDep):
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
        return dataset_loader.get_metrics(dataset_name)
    except Exception as e:
        logger.error(f"Error retrieving metrics for dataset '{dataset_name}': {e}")
        raise RuntimeError(
            f"Failed to retrieve metrics for dataset '{dataset_name}': {e}"
        )


@router.get("/by-tasks", status_code=200)
async def datasets_by_tasks(
    task_types: Annotated[
        list[str],
        Query(description="List of task types to filter datasets by"),
    ],
    include_metrics: Annotated[
        bool, Query(description="Whether to include dataset metrics in the response")
    ] = False,
    *,
    dataset_loader: DatasetLoaderDep,
):
    """
    Returns a list of datasets that are associated with specific task types.
    Args:
        task_types (list[str]): A list of task types to filter datasets by.
    Returns:
        list[dict]: A list of datasets that match the specified task types.
    Raises:
        RuntimeError: If there is an error retrieving datasets.
    """
    try:
        datasets = dataset_loader.find_datasets_by_tasks(
            task_types=task_types, include_metrics=include_metrics
        )
    except Exception as e:
        logger.error(f"Error retrieving datasets by tasks '{task_types}': {e}")
        raise RuntimeError(f"Failed to retrieve datasets by tasks '{task_types}': {e}")

    return datasets
