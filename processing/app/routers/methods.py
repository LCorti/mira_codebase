from fastapi import APIRouter, Query

from app.config.registry import get_database_service

from app.services.logging_service import LoggingService

from app.controllers.method_controller import MethodController

from app.models.method import Method, MethodUpdate

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/methods",
    tags=["methods"],
)

@router.get("", status_code=200)
async def index(
    include: set[str] | None = Query(None, description="Relations to include"),
):
    """
    Returns a list of all methods in the database.
    Args:
        None
    Returns:
        list[Method]: List of all methods serialized as Method models.
    Raises:
        RuntimeError: If there is an error retrieving methods from the database.
    """    
    return MethodController.get_all_methods(include=include)

@router.get("/{method_id}", status_code=200)
async def show(
    method_id: str,
    include: set[str] | None = Query(None, description="Fields to include"),
):
    """
    Returns a specific method by its ID.
    Args:
        method_id (str): The ID of the method to retrieve.
    Returns:
        Method: The method serialized as a Method model.
    Raises:
        RuntimeError: If there is an error retrieving the method from the database.
    """
    return MethodController.get_method_by_id(method_id, include=include)

@router.post("", status_code=201)
async def create(
    method: Method
):
    """
    Creates a new method in the database.
    Args:
        method (Method): The method data to create.
    Returns:
        Method: The created method serialized as a Method model.
    Raises:
        RuntimeError: If there is an error creating the method in the database.
    """
    return MethodController.create_method(method)

@router.put("/{method_id}", status_code=200)
async def update(
    method_id: str, 
    method: MethodUpdate
):
    """
    Updates an existing method in the database.
    Args:
        method_id (str): The ID of the method to update.
        method (MethodUpdate): The updated method data.
    Returns:
        Method: The updated method serialized as a Method model.
    Raises:
        RuntimeError: If there is an error updating the method in the database.
    """
    return MethodController.update_method(method_id, method)

@router.delete("/{method_id}", status_code=204)
async def delete(
    method_id: str
):
    """
    Deletes a method from the database.
    Args:
        method_id (str): The ID of the method to delete.
    Returns:
        dict: A message indicating the deletion status.
    Raises:
        RuntimeError: If there is an error deleting the method from the database.
    """
    return MethodController.delete_method(method_id)