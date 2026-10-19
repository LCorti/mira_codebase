from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.registry import get_dataset_loader, get_model_loader
from app.controllers.metric_controller import MetricController
from app.controllers.test_controller import TestController
from app.services.dataset_loader import DatasetLoaderService
from app.services.logging_service import LoggingService
from app.services.model_loader import ModelLoaderService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/web",
    tags=["web"],
)

ModelLoaderDep = Annotated[ModelLoaderService, Depends(get_model_loader)]
DatasetLoaderDep = Annotated[DatasetLoaderService, Depends(get_dataset_loader)]


@router.get("/metadata", status_code=200)
def metadata(model_loader: ModelLoaderDep, dataset_loader: DatasetLoaderDep):
    """
    Returns metadata for all datasets.
    Args:
        None
    Returns:
        dict: Metadata for all datasets.
    Raises:
        RuntimeError: If there is an error retrieving dataset metadata.
    """
    try:
        models = model_loader.index()
        datasets = dataset_loader.index(include_metrics=True)
        metrics = MetricController.get_all_metadata()
        tests = TestController.get_all_default_tests()
        value = {
            "models": models,
            "datasets": datasets,
            "metrics": metrics,
            "tests": tests,
        }
        return value
    except Exception as e:
        logger.error(f"Error retrieving dataset metadata: {e}")
        raise RuntimeError(f"Failed to retrieve dataset metadata: {e}")


@router.get("/metadata/tests", status_code=200)
def test_metadata():
    try:
        return TestController.get_all_default_tests()
    except Exception as e:
        logger.error(f"Error retrieving test metadata: {e}")
        raise RuntimeError(f"Failed to retrieve test metadata: {e}")


@router.post("/groups/update-name", status_code=200)
def update_group_name(group_name: str, new_name: str):
    try:
        return TestController.update_group_name(group_name, new_name)
    except Exception as e:
        logger.error(f"Error updating group name: {e}")
        raise RuntimeError(f"Failed to update group name: {e}")
