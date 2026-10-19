from fastapi import APIRouter

from app.services.logging_service import LoggingService

from app.controllers.result_controller import ResultController

from app.models.result import Result, ResultUpdate

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/results",
    tags=["results"],
)

@router.get("", status_code=200)
async def index():
    """
    Returns a list of all results in the database.
    Args:
        None
    Returns:
        list[Result]: List of all results serialized as Result models.
    Raises:
        RuntimeError: If there is an error retrieving results from the database.
    """    
    return ResultController.get_all_results()

@router.get("/{result_id}", status_code=200)
async def show(
    result_id: str,
):
    """
    Returns a specific result by its ID.
    Args:
        result_id (str): The ID of the result to retrieve.
    Returns:
        Result: The result serialized as a Result model.
    Raises:
        RuntimeError: If there is an error retrieving the result from the database.
    """
    return ResultController.get_result_by_id(result_id)

@router.post("", status_code=201)
async def create(
    result: Result
):
    """
    Creates a new result in the database.
    Args:
        result (Result): The result data to create.
    Returns:
        Result: The created result serialized as a Result model.
    Raises:
        RuntimeError: If there is an error creating the result in the database.
    """
    return ResultController.create_result(result)

@router.put("/{result_id}", status_code=200)
async def update(
    result_id: str, 
    result: ResultUpdate
):
    """
    Updates an existing result in the database.
    Args:
        result_id (str): The ID of the result to update.
        result (ResultUpdate): The updated result data.
    Returns:
        Result: The updated result serialized as a Result model.
    Raises:
        RuntimeError: If there is an error updating the result in the database.
    """
    return ResultController.update_result(result_id, result)

@router.delete("/{result_id}", status_code=204)
async def delete(
    result_id: str
):
    """
    Deletes a result from the database.
    Args:
        result_id (str): The ID of the result to delete.
    Returns:
        dict: A message indicating the deletion status.
    Raises:
        RuntimeError: If there is an error deleting the result from the database.
    """
    return ResultController.delete_result(result_id)