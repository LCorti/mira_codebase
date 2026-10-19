from contextlib import contextmanager
from typing import Any

from app.config.registry import get_database_service
from app.models.result import Result, ResultUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_results_collection():
    """
    Returns the results collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("results")


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("jobs")


class ResultController:
    """
    Controller class for handling result-related business logic.
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
    def get_all_results() -> list[dict[str, Any]]:
        """
        Retrieves all results from the database.

        Args:
            None

        Returns:
            list[dict]: List of all results serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving results from the database.
        """
        try:
            results = _get_results_collection().find()
            return [Result(**result).to_dict() for result in results]
        except Exception as e:
            logger.error(f"Error retrieving results from the database: {e}")
            raise RuntimeError(f"Failed to retrieve results from the database: {e}")

    @staticmethod
    def get_result_by_id(result_id: str) -> dict[str, Any]:
        """
        Retrieves a specific result by its ID.

        Args:
            result_id (str): The ID of the result to retrieve.

        Returns:
            dict: The result serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the result from the database.
        """
        try:
            result = _get_results_collection().find_one({"id": str(result_id)})
            if not result:
                logger.warning(f"Result with ID {result_id} not found.")
                raise ValueError(f"Result with ID {result_id} not found.")
            return Result(**result).to_dict()
        except Exception as e:
            logger.error(f"Error retrieving result with ID {result_id}: {e}")
            raise RuntimeError(f"Failed to retrieve result with ID {result_id}: {e}")

    @staticmethod
    def create_result(result: Result) -> dict[str, Any]:
        """
        Creates a new result in the database.

        Args:
            result (Result): The result data to create.

        Returns:
            dict: The created result serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the result in the database.
        """
        try:
            with ResultController.database_transaction():
                # Find job
                db_job = _get_jobs_collection().find_one({"id": str(result.job_id)})
                if not db_job:
                    logger.error(f"Job with ID {result.job_id} not found.")
                    raise ValueError(f"Job with ID {result.job_id} not found.")

                from app.models.job import Job

                job = Job(**db_job)

                if job and job.parameters:
                    result.parameters = job.parameters
                else:
                    result.parameters = {}
                    logger.warning(
                        f"Job with ID {result.job_id} not found or has no parameters."
                    )

                # Create new result
                result_insert = _get_results_collection().insert_one(result.to_dict())
                if not result_insert.acknowledged:
                    logger.error(f"Failed to create result for job {result.job_id}.")
                    raise RuntimeError(
                        f"Failed to create result for job {result.job_id}."
                    )

                # Update the method with the new result_id
                _get_methods_collection().update_one(
                    {"id": str(result.method_id)},
                    {"$addToSet": {"result_ids": str(result.id)}},
                )

                # Update the job with the new result_id
                _get_jobs_collection().update_one(
                    {"id": str(result.job_id)}, {"$set": {"result_id": str(result.id)}}
                )

                return result.to_dict()
        except Exception as e:
            logger.error(f"Error creating result: {e}")
            raise RuntimeError(f"Failed to create result: {e}")

    @staticmethod
    def update_result(result_id: str, result: ResultUpdate) -> dict[str, Any]:
        """
        Updates an existing result in the database.

        Args:
            result_id (str): The ID of the result to update.
            result (ResultUpdate): The updated result data.

        Returns:
            dict: The updated result serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the result in the database.
        """
        try:
            with ResultController.database_transaction():
                # Fetch the existing result
                current_result = _get_results_collection().find_one(
                    {"id": str(result_id)}
                )
                if not current_result:
                    logger.warning(f"Result with ID {result_id} not found for update.")
                    raise ValueError(f"Result with ID {result_id} not found.")

                # Convert to Result model for type consistency
                current_result = Result(**current_result)

                # Update the method_id
                if current_result.method_id != result.method_id:
                    _get_methods_collection().update_one(
                        {"id": str(current_result.method_id)},
                        {"$pull": {"result_ids": str(result_id)}},
                    )
                    _get_methods_collection().update_one(
                        {"id": str(result.method_id)},
                        {"$addToSet": {"result_ids": str(result_id)}},
                    )

                # Update the result
                update_result = _get_results_collection().update_one(
                    {"id": str(result_id)}, {"$set": result.to_dict()}
                )
                if update_result.matched_count == 0:
                    logger.warning(f"Result with ID {result_id} not found for update.")
                    raise ValueError(f"Result with ID {result_id} not found.")
                updated_result = _get_results_collection().find_one(
                    {"id": str(result_id)}
                )

                return Result(**updated_result).to_dict()
        except Exception as e:
            logger.error(f"Error updating result with ID {result_id}: {e}")
            raise RuntimeError(f"Failed to update result with ID {result_id}: {e}")

    @staticmethod
    def update_result_attr(
        result_id: str, result_summary: dict[str, Any], results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Updates the results of a job.

        Args:
            result_id (str): The ID of the result to update.
            result_summary (dict[str, Any]): The summary to set for the result.
            results (list[dict[str, Any]]): The new results to set for the result.

        Returns:
            dict: The updated result serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the result in the database.
        """
        try:
            with ResultController.database_transaction():
                result = _get_results_collection().update_one(
                    {"id": str(result_id)},
                    {"$set": {"summary": result_summary, "results": results}},
                )

                if result.matched_count == 0:
                    raise ValueError(f"Result with ID {result_id} not found.")

                updated_result = _get_results_collection().find_one(
                    {"id": str(result_id)}
                )

                return Result(**updated_result).to_dict()
        except Exception as e:
            logger.error(f"Error updating result for ID {result_id}: {e}")
            raise RuntimeError(f"Failed to update result for ID {result_id}: {e}")

    @staticmethod
    def update_diagnostics_attr(
        result_ids: str, diagnostics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Updates the diagnostics of a result.

        Args:
            result_ids (str): The ID of the result to update.
            diagnostics (dict[str, Any]): The diagnostics to set for the result.

        Returns:
            dict: The updated result serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the result in the database.
        """
        try:
            with ResultController.database_transaction():
                result = _get_results_collection().update_one(
                    {"id": str(result_ids)}, {"$set": {"diagnostics": diagnostics}}
                )

                if result.matched_count == 0:
                    raise ValueError(f"Result with ID {result_ids} not found.")

                updated_result = _get_results_collection().find_one(
                    {"id": str(result_ids)}
                )

                return Result(**updated_result).to_dict()
        except Exception as e:
            logger.error(f"Error updating diagnostics for ID {result_ids}: {e}")
            raise RuntimeError(f"Failed to update diagnostics for ID {result_ids}: {e}")

    @staticmethod
    def delete_result(result_id: str):
        """
        Deletes a result from the database.

        Args:
            result_id (str): The ID of the result to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the result from the database.
        """
        try:
            with ResultController.database_transaction():
                # Fetch the result to get its method_id
                result = _get_results_collection().find_one({"id": str(result_id)})
                if not result:
                    logger.warning(
                        f"Result with ID {result_id} not found for deletion."
                    )
                    raise ValueError(f"Result with ID {result_id} not found.")

                # Convert result to Result model for type consistency
                result_obj = Result(**result)

                # Remove result_id from the methods' result_ids array
                _get_methods_collection().update_one(
                    {"id": str(result_obj.method_id)},
                    {"$pull": {"result_ids": str(result_id)}},
                )

                # Delete the result
                delete_result = _get_results_collection().delete_one(
                    {"id": str(result_id)}
                )
                if delete_result.deleted_count == 0:
                    logger.warning(
                        f"Result with ID {result_id} not found for deletion."
                    )
                    raise ValueError(f"Result with ID {result_id} not found.")

                return {"message": f"Result with ID {result_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting result with ID {result_id}: {e}")
            raise RuntimeError(f"Failed to delete result with ID {result_id}: {e}")
