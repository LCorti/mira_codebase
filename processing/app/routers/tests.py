from typing import Annotated

from fastapi import APIRouter, Query

from app.controllers.test_controller import TestController
from app.models.test import Test, TestUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/tests",
    tags=["tests"],
)


@router.get("", status_code=200)
async def index(
    include: Annotated[
        set[str] | None, Query(description="Relations to include")
    ] = None,
):
    """
    Returns a list of all tests in the database.
    Args:
        None
    Returns:
        list[Test]: List of all tests serialized as Test models.
    Raises:
        RuntimeError: If there is an error retrieving tests from the database.
    """
    return TestController.get_all_tests(include)


@router.get("/default", status_code=200)
async def get_default_tests():
    """
    Returns a list of all default tests in the database.
    Args:
        None
    Returns:
        list[Test]: List of all default tests serialized as Test models.
    Raises:
        RuntimeError: If there is an error retrieving default tests from the database.
    """
    return TestController.get_all_default_tests()


@router.get("/{test_id}", status_code=200)
async def show(
    test_id: str,
    include: Annotated[
        set[str] | None, Query(description="Relations to include")
    ] = None,
):
    """
    Returns a specific test by its ID.
    Args:
        test_id (str): The ID of the test to retrieve.
    Returns:
        Test: The test serialized as a Test model.
    Raises:
        RuntimeError: If there is an error retrieving the test from the database.
    """
    return TestController.get_test_by_id(test_id, include=include)


@router.post("/create", status_code=201)
async def create(test: Test):
    """
    Creates a new test in the database.
    Args:
        test (Test): The test data to create.
    Returns:
        Test: The created test serialized as a Test model.
    Raises:
        RuntimeError: If there is an error creating the test in the database.
    """
    return TestController.create_test(test)


@router.post("/create-with-methods", status_code=201)
async def create_with_methods(
    test: Test,
    datasets: list[str] | None = None,
    metrics: dict[str, list[str]] | None = None,
):
    """
    Creates a new test with associated methods in the database.
    Args:
        test (Test): The test data to create.
        datasets (list[str], optional): List of dataset names to associate with the test.
        metrics (dict[str, list[str]], optional): Dictionary of metrics to associate with the test.
    Returns:
        Test: The created test serialized as a Test model.
    Raises:
        RuntimeError: If there is an error creating the test in the database.
    """
    return TestController.create_test_with_methods(test, datasets, metrics)


@router.put("/{test_id}", status_code=200)
async def update(test_id: str, test: TestUpdate):
    """
    Updates an existing test in the database.
    Args:
        test_id (str): The ID of the test to update.
        test (TestUpdate): The updated test data.
    Returns:
        Test: The updated test serialized as a Test model.
    Raises:
        RuntimeError: If there is an error updating the test in the database.
    """
    return TestController.update_test(test_id, test)


@router.delete("/{test_id}", status_code=204)
async def delete(test_id: str):
    """
    Deletes a test from the database.
    Args:
        test_id (str): The ID of the test to delete.
    Returns:
        dict: A message indicating the deletion status.
    Raises:
        RuntimeError: If there is an error deleting the test from the database.
    """
    return TestController.delete_test(test_id)


@router.get("/{test_id}/methods", tags=["tests"])
async def get_test_methods(test_id: str):
    """
    Returns all methods associated with a specific test.
    Args:
        test_id (str): The ID of the test to retrieve methods for.
    Returns:
        list[Method]: List of methods associated with the test.
    Raises:
        RuntimeError: If there is an error retrieving methods from the database.
    """
    return TestController.get_methods(test_id)
