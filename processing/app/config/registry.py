import os

from app.models.diagnostics.diagnostic_registry import DiagnosticRegistry
from app.models.metrics.metric_registry import MetricRegistry
from app.services.database import DatabaseService
from app.services.dataset_loader import DatasetLoaderService
from app.services.logging_service import LoggingService
from app.services.model_loader import ModelLoaderService
from app.services.queue import JobQueue
from app.services.tag_service import TagService
from app.services.task_service import TaskService

logger = LoggingService.setup_module_logger(__name__)

__model_loader_service: ModelLoaderService | None = None
__dataset_loader_service: DatasetLoaderService | None = None
__database_service: DatabaseService | None = None
__job_queue: JobQueue | None = None
__task_service: TaskService | None = None
__tag_service: TagService | None = None

__metric_registry: MetricRegistry | None = None
__diagnostic_registry: DiagnosticRegistry | None = None


# Setters for services
def set_model_loader(loader: ModelLoaderService):
    global __model_loader_service
    __model_loader_service = loader


def set_dataset_loader(loader: DatasetLoaderService):
    global __dataset_loader_service
    __dataset_loader_service = loader


def set_database_service(service: DatabaseService):
    global __database_service
    __database_service = service


def set_job_queue(queue: JobQueue):
    global __job_queue
    __job_queue = queue


def set_task_service(service: TaskService):
    global __task_service
    __task_service = service


def set_tag_service(service: TagService):
    global __tag_service
    __tag_service = service


def set_metric_registry(registry: MetricRegistry):
    global __metric_registry
    __metric_registry = registry


def set_diagnostic_registry(registry: DiagnosticRegistry):
    global __diagnostic_registry
    __diagnostic_registry = registry


# Getters for services
def get_model_loader() -> ModelLoaderService:
    if __model_loader_service is None:
        raise ValueError("ModelLoaderService has not been set.")
    return __model_loader_service


def get_dataset_loader() -> DatasetLoaderService:
    if __dataset_loader_service is None:
        raise ValueError("DatasetLoaderService has not been set.")
    return __dataset_loader_service


def get_database_service() -> DatabaseService:
    if __database_service is None:
        raise ValueError("DatabaseService has not been set.")
    return __database_service


def get_job_queue() -> JobQueue:
    if __job_queue is None:
        raise ValueError("JobQueue has not been set.")
    return __job_queue


def get_task_service() -> TaskService:
    if __task_service is None:
        raise ValueError("TaskService has not been set.")
    return __task_service


def get_tag_service() -> TagService:
    if __tag_service is None:
        raise ValueError("TagService has not been set.")
    return __tag_service


def get_metric_registry() -> MetricRegistry:
    if __metric_registry is None:
        raise ValueError("MetricRegistry has not been set.")
    return __metric_registry


def get_diagnostic_registry() -> DiagnosticRegistry:
    if __diagnostic_registry is None:
        raise ValueError("DiagnosticRegistry has not been set.")
    return __diagnostic_registry


def start_helpers():
    """
    Initializes the helper functions for model loading, dataset loading, and database access.
    Args:
        logging_level (int): The logging level to set for the services.
    Returns:
        None
    """
    logger.info("Starting helper functions.")

    global \
        __model_loader_service, \
        __dataset_loader_service, \
        __database_service, \
        __job_queue, \
        __metric_registry, \
        __task_service, \
        __diagnostic_registry

    num_workers = int(os.getenv("NUM_WORKERS", "1"))

    set_model_loader(ModelLoaderService())
    set_dataset_loader(DatasetLoaderService())
    set_database_service(DatabaseService())
    set_job_queue(JobQueue(num_workers))
    set_task_service(TaskService())
    set_metric_registry(MetricRegistry())
    set_diagnostic_registry(DiagnosticRegistry())

    logger.info("Helper functions have been set up.")


def stop_helpers():
    """
    Stops the helper functions for model loading, dataset loading, and database access.
    """

    global \
        __model_loader_service, \
        __dataset_loader_service, \
        __database_service, \
        __job_queue, \
        __metric_registry, \
        __task_service, \
        __diagnostic_registry

    if __model_loader_service:
        __model_loader_service.close()
        logger.info("ModelLoaderService has been closed.")

    if __dataset_loader_service:
        __dataset_loader_service.close()
        logger.info("DatasetLoaderService has been closed.")

    if __database_service:
        __database_service.close()
        logger.info("DatabaseService has been closed.")

    if __job_queue:
        __job_queue.stop()
        logger.info("JobQueue has been closed.")

    if __task_service:
        __task_service.close()
        logger.info("TaskService has been closed.")

    if __metric_registry:
        __metric_registry.close()
        logger.info("MetricRegistry has been closed.")

    if __diagnostic_registry:
        __diagnostic_registry.close()
        logger.info("DiagnosticRegistry has been closed.")

    logger.info("Helper functions have been stopped.")
