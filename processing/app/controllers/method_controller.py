from contextlib import contextmanager
from typing import Any

from app.config.registry import get_database_service
from app.models.method import Method, MethodUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_tests_collection():
    """
    Returns the tests collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("tests")


def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("jobs")


class MethodController:
    """
    Controller class for handling method-related business logic.
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
    def get_all_methods(include: set[str] | None) -> list[dict[str, Any]]:
        """
        Retrieves all methods from the database.

        Args:
            include (set[str] | None): Fields to include in the response.

        Returns:
            list[dict]: List of all methods serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving methods from the database.
        """
        try:
            methods = _get_methods_collection().find()
            return [Method(**method).to_dict(include=include) for method in methods]
        except Exception as e:
            logger.error(f"Error retrieving methods from the database: {e}")
            raise RuntimeError(f"Failed to retrieve methods from the database: {e}")

    @staticmethod
    def get_method_by_id(method_id: str, include: set[str] | None) -> dict[str, Any]:
        """
        Retrieves a specific method by its ID.

        Args:
            method_id (str): The ID of the method to retrieve.
            include (set[str] | None): Fields to include in the response.

        Returns:
            dict: The method serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the method from the database.
        """
        try:
            method = _get_methods_collection().find_one({"id": method_id})
            if not method:
                logger.warning(f"Method with ID {method_id} not found.")
                raise ValueError(f"Method with ID {method_id} not found.")
            return Method(**method).to_dict(include=include)
        except Exception as e:
            logger.error(f"Error retrieving method with ID {method_id}: {e}")
            raise RuntimeError(f"Failed to retrieve method with ID {method_id}: {e}")

    @staticmethod
    def create_method(method: Method) -> dict[str, Any]:
        """
        Creates a new method in the database.

        Args:
            method (Method): The method data to create.

        Returns:
            dict: The created method serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the method in the database.
        """
        try:
            with MethodController.database_transaction():
                # Find test
                test = _get_tests_collection().find_one({"id": method.test_id})

                # Check if test has parameters
                if test and test.get("parameters"):
                    method.parameters = {
                        **(test["parameters"] or {}),
                        **(method.parameters or {}),
                    }
                else:
                    method.parameters = method.parameters or {}

                # Create a new method
                result = _get_methods_collection().insert_one(method.to_dict())
                if not result.acknowledged:
                    logger.error(f"Failed to create method: {method.id}")
                    raise RuntimeError("Failed to create method in the database.")

                updated_method = _get_methods_collection().find_one({"id": method.id})

                # Convert to Method model for type consistency
                new_method = Method(**updated_method)

                # Update the test_ids in the tests collection
                _get_tests_collection().update_one(
                    {"id": method.test_id},
                    {"$addToSet": {"method_ids": str(new_method.id)}},
                )

                return new_method.to_dict()
        except Exception as e:
            logger.error(f"Error creating method: {e}")
            raise RuntimeError(f"Failed to create method: {e}")

    @staticmethod
    def update_method(method_id: str, method: MethodUpdate) -> dict[str, Any]:
        """
        Updates an existing method in the database.

        Args:
            method_id (str): The ID of the method to update.
            method (MethodUpdate): The updated method data.

        Returns:
            dict: The updated method serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the method in the database.
        """
        try:
            with MethodController.database_transaction():
                # Fetch the existing method
                current_method = _get_methods_collection().find_one({"id": method_id})
                if not current_method:
                    logger.warning(f"Method with ID {method_id} not found for update.")
                    raise ValueError(f"Method with ID {method_id} not found.")

                # Convert to Method model for type consistency
                current_method = Method(**current_method)

                # Update the test_ids if test_id has changed
                if method.test_id and method.test_id != current_method.test_id:
                    _get_tests_collection().update_one(
                        {"id": current_method.test_id},
                        {"$pull": {"method_ids": str(method_id)}},
                    )
                    _get_tests_collection().update_one(
                        {"id": method.test_id},
                        {"$addToSet": {"method_ids": str(method_id)}},
                    )

                # Update the method
                result = _get_methods_collection().update_one(
                    {"id": method_id}, {"$set": method.to_dict()}
                )
                if result.matched_count == 0:
                    logger.warning(f"Method with ID {method_id} not found for update.")
                    raise ValueError(f"Method with ID {method_id} not found.")
                updated_method = _get_methods_collection().find_one({"id": method_id})

                return Method(**updated_method).to_dict()
        except Exception as e:
            logger.error(f"Error updating method with ID {method_id}: {e}")
            raise RuntimeError(f"Failed to update method with ID {method_id}: {e}")

    @staticmethod
    def update_parameters(method_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Updates the parameters of an existing method in the database.

        Args:
            method_id (str): The ID of the method to update.
            parameters (dict[str, Any]): The updated parameters for the method.

        Returns:
            dict: The updated method serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the method in the database.
        """
        try:
            with MethodController.database_transaction():
                result = _get_methods_collection().update_one(
                    {"id": method_id}, {"$set": {"parameters": parameters}}
                )
                if result.matched_count == 0:
                    logger.warning(f"Method with ID {method_id} not found for update.")
                    raise ValueError(f"Method with ID {method_id} not found.")
                updated_method = _get_methods_collection().find_one({"id": method_id})

                return Method(**updated_method).to_dict()
        except Exception as e:
            logger.error(f"Error updating method with ID {method_id}: {e}")
            raise RuntimeError(f"Failed to update method with ID {method_id}: {e}")

    @staticmethod
    def delete_method(method_id: str):
        """
        Deletes a method from the database.

        Args:
            method_id (str): The ID of the method to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the method from the database.
        """
        try:
            with MethodController.database_transaction():
                # Fetch the method to get its test_ids and job_ids
                method = _get_methods_collection().find_one({"id": method_id})
                if not method:
                    logger.warning(
                        f"Method with ID {method_id} not found for deletion."
                    )
                    raise ValueError(f"Method with ID {method_id} not found.")

                # Convert to Method model for type consistency
                method_obj = Method(**method)

                # Remove method_id from the tests' method_ids array
                _get_tests_collection().update_one(
                    {"id": method_obj.test_id},
                    {"$pull": {"method_ids": str(method_id)}},
                )

                # Delete all jobs associated with this method
                for job_id in method_obj.job_ids:
                    job = _get_jobs_collection().find_one({"id": job_id})
                    if not job:
                        logger.warning(f"Job with ID {job_id} not found for deletion.")
                        continue

                    # Delete the job
                    _get_jobs_collection().delete_one({"id": job_id})

                # Delete the method
                result = _get_methods_collection().delete_one({"id": method_id})
                if result.deleted_count == 0:
                    logger.warning(
                        f"Method with ID {method_id} not found for deletion."
                    )
                    raise ValueError(f"Method with ID {method_id} not found.")

                return {"message": f"Method with ID {method_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting method with ID {method_id}: {e}")
            raise RuntimeError(f"Failed to delete method with ID {method_id}: {e}")
