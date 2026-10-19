from fastapi import APIRouter, Depends, Query

from app.config.registry import get_task_service

from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)

@router.get("", status_code=200)
async def index(
    task_service = Depends(get_task_service)
):
    """
    Returns a list of all tasks in the system.
    Args:
        None
    Returns:
        list[str]: List of dataset names.
    Raises:
        RuntimeError: If there is an error retrieving datasets.
    """
    try:
        tasks = task_service.index()
        return tasks
    except Exception as e:
        logger.error(f"Error retrieving tasks: {e}")
        raise RuntimeError(f"Failed to retrieve tasks: {e}")

@router.get("/names", status_code=200)
async def names(
    with_subtasks: bool = Query(False, description="Include subtasks in the list"),
    task_service = Depends(get_task_service)
):
    """
    Returns a list of all task names in the system.
    Args:
        with_subtasks (bool): Whether to include subtasks in the list.
    Returns:
        dict: Dictionary of task names.
    Raises:
        RuntimeError: If there is an error retrieving task names.
    """
    try:
        tasks = task_service.index()
        if not with_subtasks:
            return {"task_types": list(tasks.keys())}

        task_dict = {}
        for task_name, task in tasks.items():
            task_dict[task_name] = [task.get("type") for task in task.get("subtasks", [])]
        return {"task_types": list(tasks.keys()), "subtasks": task_dict}
    except Exception as e:
        logger.error(f"Error retrieving task names: {e}")
        raise RuntimeError(f"Failed to retrieve task names: {e}")

@router.get("/show", status_code=200)
async def show(
    task_name: str,
    task_service = Depends(get_task_service)
):
    """
    Returns the details of a specific task.
    Args:
        task_name (str): The name of the task to retrieve.
    Returns:
        dict: Details of the specified task.
    Raises:
        RuntimeError: If there is an error retrieving the task.
    """
    try:
        task = task_service.show(task_name)
        return task
    except Exception as e:
        logger.error(f"Error retrieving task '{task_name}': {e}")
        raise RuntimeError(f"Failed to retrieve task '{task_name}': {e}")

@router.get("/show-multiple", status_code=200)
async def show_multiple(
    task_names: list[str] = Query(..., description="List of task names to retrieve"),
    task_service = Depends(get_task_service)
):
    """
    Returns the details of multiple tasks.
    Args:
        task_names (list[str]): The names of the tasks to retrieve.
    Returns:
        dict: Details of the specified tasks.
    Raises:
        RuntimeError: If there is an error retrieving the tasks.
    """
    try:
        tasks = task_service.show_multiple(task_names)
        return tasks
    except Exception as e:
        logger.error(f"Error retrieving tasks '{task_names}': {e}")
        raise RuntimeError(f"Failed to retrieve tasks '{task_names}': {e}")
    
@router.get("/available-tasks", status_code=200)
async def available_tasks(
    task_service = Depends(get_task_service)
):
    """
    Returns a list of all available task types in the system.
    Args:
        None
    Returns:
        list[str]: List of available task types.
    Raises:
        RuntimeError: If there is an error retrieving available tasks.
    """
    try:
        return task_service.get_available_tasks()
    except Exception as e:
        logger.error(f"Error retrieving available tasks: {e}")
        raise RuntimeError(f"Failed to retrieve available tasks: {e}")