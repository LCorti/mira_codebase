from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from app.config.registry import get_database_service
from app.models.test import Test, TestUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_tests_collection():
    """
    Returns the tests collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("tests")


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_projects_collection():
    """
    Returns the projects collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("projects")


class TestController:
    """
    Controller class for handling test-related business logic.
    """

    @staticmethod
    @contextmanager
    def database_transaction():
        """
        Context manager for database transactions to reduce repetition.
        """
        session = get_database_service().start_session()
        try:
            with session.start_transaction():
                yield session
                session.commit_transaction()
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise

    @staticmethod
    def get_all_tests(include: set[str] | None) -> list[dict[str, Any]]:
        """
        Retrieves all tests from the database.

        Args:
            include (set[str] | None): Fields to include in the response.

        Returns:
            list[dict]: List of all tests serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving tests from the database.
        """
        try:
            tests = _get_tests_collection().find()
            return [Test(**test).to_dict(include=include) for test in tests]
        except Exception as e:
            logger.error(f"Error retrieving tests from the database: {e}")
            return []

    @staticmethod
    def get_all_default_tests() -> list[dict[str, Any]]:
        """
        Retrieves all default tests from the database.

        Args:
            None

        Returns:
            list[dict]: List of all default tests serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving default tests from the database.
        """
        try:
            default_tests = _get_tests_collection().find({"default": True})
            return [Test(**test).to_dict(include={"methods"}) for test in default_tests]
        except Exception as e:
            logger.error(f"Error retrieving default tests from the database: {e}")
            return []

    @staticmethod
    def get_test_by_id(test_id: str, include: set[str] | None) -> dict[str, Any]:
        """
        Retrieves a specific test by its ID.

        Args:
            test_id (str): The ID of the test to retrieve.
            include (set[str] | None): The relations to include in the response.

        Returns:
            dict: The test serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the test from the database.
        """
        try:
            test = _get_tests_collection().find_one({"id": test_id})
            if not test:
                logger.warning(f"Test with ID {test_id} not found.")
                return {}
            return Test(**test).to_dict(include=include)
        except Exception as e:
            logger.error(f"Error retrieving test with ID {test_id}: {e}")
            return {}

    @staticmethod
    def create_test(test: Test) -> dict[str, Any]:
        """
        Creates a new test in the database.

        Args:
            test (Test): The test data to create.

        Returns:
            dict: The created test serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the test in the database.
        """
        try:
            with TestController.database_transaction():
                # Find project
                project = _get_projects_collection().find_one({"id": test.project_id})

                if not project:
                    logger.warning(f"Project with ID {test.project_id} not found.")
                    raise ValueError(f"Project with ID {test.project_id} not found.")

                # Check if project has parameters
                if project.get("parameters") and not test.parameters:
                    test.parameters = project["parameters"]

                # Create new test
                new_test = _get_tests_collection().insert_one(test.to_dict())
                if not new_test.acknowledged:
                    logger.error(f"Failed to create test for project {test.project_id}")
                    raise RuntimeError("Failed to create test")

                updated_test = _get_tests_collection().find_one({"id": test.id})

                # Update the test's project_id
                _get_projects_collection().update_one(
                    {"id": test.project_id}, {"$addToSet": {"test_ids": str(test.id)}}
                )

                # Update the test's method_ids array
                for method_id in test.method_ids:
                    _get_methods_collection().update_one(
                        {"id": method_id}, {"$addToSet": {"test_ids": str(test.id)}}
                    )

                return Test(**updated_test).to_dict()
        except Exception as e:
            logger.error(f"Error creating test: {e}")
            return {}

    @staticmethod
    def create_test_with_methods(
        test: Test, datasets: list[str], metrics: dict[str, list[str]]
    ) -> dict[str, Any]:
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
        try:
            with TestController.database_transaction():
                # Find project
                project = _get_projects_collection().find_one({"id": test.project_id})

                if not project:
                    logger.warning(f"Project with ID {test.project_id} not found.")
                    raise ValueError(f"Project with ID {test.project_id} not found.")

                # Check if project has parameters
                if project.get("parameters") and not test.parameters:
                    test.parameters = project["parameters"]

                # Create new test
                new_test = _get_tests_collection().insert_one(test.to_dict())
                if not new_test.acknowledged:
                    logger.error(f"Failed to create test for project {test.project_id}")
                    raise RuntimeError("Failed to create test")

                updated_test = _get_tests_collection().find_one({"id": test.id})

                # Update the test's project_id
                _get_projects_collection().update_one(
                    {"id": test.project_id}, {"$addToSet": {"test_ids": str(test.id)}}
                )

                # Create methods for test
                method_ids = []
                for dataset in datasets:
                    c_metrics = metrics.get(dataset, [])

                    c_parameters = deepcopy(test.parameters)
                    c_parameters["dataset"] = c_parameters.get("dataset", {}).get(
                        dataset, {}
                    )
                    c_parameters["metrics"] = c_parameters.get("metrics", {}).get(
                        dataset, {}
                    )

                    from app.models.method import Method

                    c_method = Method(
                        name=dataset,
                        description=test.description,
                        dataset_name=dataset,
                        metrics=c_metrics,
                        parameters=c_parameters,
                        test_id=str(test.id),
                        job_ids=[],
                        result_ids=[],
                    )

                    method_response = _get_methods_collection().insert_one(
                        c_method.to_dict()
                    )
                    if not method_response.acknowledged:
                        logger.error(f"Failed to create method for dataset {dataset}")
                        raise RuntimeError("Failed to create method")

                    method_ids.append(str(c_method.id))

                # Update test's method_ids
                test_response = _get_tests_collection().update_one(
                    {"id": test.id}, {"$set": {"method_ids": method_ids}}
                )
                if not test_response.acknowledged:
                    logger.error(f"Failed to update test with ID {test.id}")
                    raise RuntimeError("Failed to update test")

                updated_test = TestController.get_test_by_id(
                    str(test.id), include={"methods"}
                )
                if not updated_test:
                    logger.error(f"Test with ID {test.id} not found after creation.")
                    raise RuntimeError(
                        f"Test with ID {test.id} not found after creation."
                    )

                return {"test": updated_test}
        except Exception as e:
            logger.error(f"Failed to create test: {e}")
            return {}

    @staticmethod
    def update_test(test_id: str, test: TestUpdate) -> dict[str, Any]:
        """
        Updates an existing test in the database.

        Args:
            test_id (str): The ID of the test to update.
            test (TestUpdate): The updated test data.
        Returns:
            dict: The updated test serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the test in the database.
        """
        try:
            with TestController.database_transaction():
                # Fetch the existing test
                current_test = _get_tests_collection().find_one({"id": test_id})
                if not current_test:
                    logger.warning(f"Test with ID {test_id} not found for update.")
                    raise ValueError(f"Test with ID {test_id} not found.")

                # Convert to Test model for type consistency
                current_test = Test(**current_test)

                # Update the project_id
                if current_test.project_id != test.project_id:
                    if current_test.project_id:
                        _get_projects_collection().update_one(
                            {"id": current_test.project_id},
                            {"$pull": {"test_ids": str(test_id)}},
                        )
                    if test.project_id:
                        _get_projects_collection().update_one(
                            {"id": test.project_id},
                            {"$addToSet": {"test_ids": str(test_id)}},
                        )

                # Update the method_ids
                if current_test.method_ids != test.method_ids:
                    for method_id in current_test.method_ids:
                        _get_methods_collection().update_one(
                            {"id": method_id}, {"$pull": {"test_ids": str(test_id)}}
                        )
                    for method_id in test.method_ids or []:
                        _get_methods_collection().update_one(
                            {"id": method_id}, {"$addToSet": {"test_ids": str(test_id)}}
                        )

                # Update the test
                result = _get_tests_collection().update_one(
                    {"id": test_id}, {"$set": test.to_dict()}
                )
                if result.matched_count == 0:
                    logger.warning(f"Test with ID {test_id} not found for update.")
                    raise ValueError(f"Test with ID {test_id} not found.")
                updated_test = _get_tests_collection().find_one({"id": test_id})

                return Test(**updated_test).to_dict()
        except Exception as e:
            logger.error(f"Error updating test with ID {test_id}: {e}")
            return {}

    @staticmethod
    def update_parameters(
        test_ids: str, parameters: dict[str, Any], update_methods: bool
    ) -> dict[str, Any]:
        """
        Updates the parameters of a test.

        Args:
            test_ids (str): The ID of the test to update.
            parameters (dict): The new parameters to set.
            update_methods (bool): Whether to update the methods' parameters as well.

        Returns:
            dict: The updated test serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the test in the database.
        """
        try:
            result = _get_tests_collection().update_one(
                {"id": test_ids}, {"$set": {"parameters": parameters}}
            )
            if result.matched_count == 0:
                logger.warning(
                    f"Test with ID {test_ids} not found for parameter update."
                )
                raise ValueError(f"Test with ID {test_ids} not found.")
            updated_test = _get_tests_collection().find_one({"id": test_ids})
            test = Test(**updated_test)

            # Update each method if required
            if update_methods:
                for method_id in test.method_ids:
                    _get_methods_collection().update_one(
                        {"id": method_id}, {"$set": {"parameters": test.parameters}}
                    )

            return test.to_dict()
        except Exception as e:
            logger.error(f"Error updating parameters for test with ID {test_ids}: {e}")
            return {}

    @staticmethod
    def delete_test(test_id: str) -> dict[str, Any]:
        """
        Deletes a test from the database.

        Args:
            test_id (str): The ID of the test to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the test from the database.
        """
        try:
            with TestController.database_transaction():
                # Fetch the test to get its project_ids and method_ids
                test = _get_tests_collection().find_one({"id": test_id})
                if not test:
                    logger.warning(f"Test with ID {test_id} not found for deletion.")
                    raise ValueError(f"Test with ID {test_id} not found.")

                # Convert test to Test model for type consistency
                test_obj = Test(**test)

                # Remove test_id from the projects' test_ids array
                for project_id in test_obj.project_ids:
                    _get_projects_collection().update_one(
                        {"id": project_id}, {"$pull": {"test_ids": str(test_id)}}
                    )

                # Remove test_id from the methods' test_ids array
                for method_id in test_obj.method_ids:
                    _get_methods_collection().update_one(
                        {"id": method_id}, {"$pull": {"test_ids": str(test_id)}}
                    )

                # Delete the test
                result = _get_tests_collection().delete_one({"id": test_id})
                if result.deleted_count == 0:
                    logger.warning(f"Test with ID {test_id} not found for deletion.")
                    raise ValueError(f"Test with ID {test_id} not found.")

                return {"message": f"Test with ID {test_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting test with ID {test_id}: {e}")
            return {"message": f"Failed to delete test with ID {test_id}"}

    @staticmethod
    def get_methods(test_id: str) -> list:
        """
        Returns all methods associated with a specific test.

        Args:
            test_id (str): The ID of the test to retrieve methods for.

        Returns:
            list: List of methods associated with the test.

        Raises:
            RuntimeError: If there is an error retrieving methods from the database.
        """
        try:
            from app.models.method import Method

            methods = _get_methods_collection().find({"test_id": test_id})
            return [Method(**method).to_dict() for method in methods]
        except Exception as e:
            logger.error(f"Error retrieving methods for test {test_id}: {e}")
            return []
