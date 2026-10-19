from fastapi import APIRouter, Depends, Query

from app.config.registry import get_metric_registry

from app.services.logging_service import LoggingService

from app.controllers.metric_controller import MetricController

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
)

@router.get("", status_code=200)
async def index():
    """
    List all available metrics.

    :return: A list of available metric names.
    """
    return MetricController.get_all_metrics()

@router.get("/metadata", status_code=200)
async def all_metrics():
    """
    Get metadata for all registered metrics.
    
    :param metric_registry: The metric registry service.
    :return: A dictionary mapping metric names to their metadata.
    """
    return MetricController.get_all_metadata()
    
@router.get("/metadata/{metric_name}", status_code=200)
async def metadata_by_task(
    metric_name: str
):
    """
    Get metadata for a specific metric.
    
    :param metric_name: The name of the metric to retrieve metadata for.
    :return: Metadata dictionary for the specified metric.
    """
    return MetricController.get_metadata_by_name(metric_name)

@router.get("/task/{task_type}", status_code=200)
async def find_by_task(
    task_types: list[str] = Query(..., alias="task_type")
):
    """
    Find metrics suitable for specific task types.

    :param task_types: A list of task types to find metrics for (e.g., ["text-classification", "text-generation"]). To narrow down the search, you can specify sub-task types as well (e.g., "text-generation.text-simplification").
    :return: A list of metric names suitable for the specified task types.
    """
    return MetricController.find_metrics_by_task(task_types)

@router.get("/{metric_name}", status_code=200)
async def show(
    metric_name: str
):
    """
    Get details of a specific metric by name.
    
    :param metric_name: The name of the metric to retrieve.
    :return: Metadata of the specified metric.
    """
    return MetricController.get_metric_by_name(metric_name)

@router.post("", status_code=201)
async def create(
    metric_name: str,
    description: str = None,
    metric_kwargs: dict = {},
    metadata: dict = None,
    register_name: str = None
):
    """
    Create a custom metric by loading a Hugging Face evaluate metric.
    
    :param metric_name: The name of the Hugging Face metric to load.
    :param description: Optional custom description for the metric.
    :param metric_kwargs: Additional keyword arguments for loading the metric.
    :param metadata: Optional metadata for the metric.
    :param register_name: Optional name to register the metric under (default: same as metric_name).
    :return: Details of the created custom metric.
    """
    return MetricController.create_custom_metric(
        metric_name=metric_name,
        description=description,
        metric_kwargs=metric_kwargs,
        metadata=metadata,
        register_name=register_name
    )

@router.delete("/{metric_name}", status_code=200)
async def delete_custom_metric(
    metric_name: str
):
    """
    Delete a custom metric from both memory and config.
    
    :param metric_name: The name of the metric to delete.
    :return: Confirmation of deletion.
    """
    return MetricController.delete_custom_metric(metric_name)