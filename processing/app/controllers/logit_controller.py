from contextlib import contextmanager
from typing import Any

import torch

from app.config.registry import get_database_service
from app.models.logit import Logit
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_logits_collection():
    """
    Returns the logits collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("logits")


class LogitController:
    """
    Controller class for handling logit-related business logic.
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
    def get_logit_by_id(logit_id: str) -> dict[str, Any]:
        """
        Retrieves a specific logit by its ID.

        Args:
            logit_id (str): The ID of the logit to retrieve.

        Returns:
            dict: The logit serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the logit from the database.
        """
        try:
            logit = _get_logits_collection().find_one({"id": str(logit_id)})
            if not logit:
                logger.warning(f"Logit with ID {logit_id} not found.")
                raise ValueError(f"Logit with ID {logit_id} not found.")
            return Logit(**logit).to_dict()
        except Exception as e:
            logger.error(f"Error retrieving logit with ID {logit_id}: {e}")
            raise RuntimeError(f"Failed to retrieve logit with ID {logit_id}: {e}")

    @staticmethod
    def get_logit_tensor_by_id(logit_id: str) -> torch.Tensor | None:
        """
        Retrieves the tensor representation of a specific logit by its ID.

        Args:
            logit_id (str): The ID of the logit to retrieve.

        Returns:
            Tensor: The tensor representation of the logit.

        Raises:
            RuntimeError: If there is an error retrieving the logit from the database.
        """
        try:
            logit = _get_logits_collection().find_one({"id": str(logit_id)})
            if not logit:
                logger.warning(f"Logit with ID {logit_id} not found.")
                return torch.Tensor()  # Return empty tensor if not found
            return Logit(**logit).to_tensor()
        except Exception as e:
            logger.error(f"Error retrieving logit with ID {logit_id}: {e}")
            raise RuntimeError(f"Failed to retrieve logit with ID {logit_id}: {e}")

    @staticmethod
    def create_logit(logit: Logit) -> dict[str, Any]:
        """
        Creates a new logit in the database.

        Args:
            logit (Logit): The logit data to create.

        Returns:
            dict: The created logit serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the logit in the database.
        """
        try:
            with LogitController.database_transaction():
                logit_insert = _get_logits_collection().insert_one(logit.to_dict())
                if not logit_insert.acknowledged:
                    logger.error("Failed to create logit.")
                    raise RuntimeError("Failed to create logit.")
                return logit.to_dict()
        except Exception as e:
            logger.error(f"Error creating logit: {e}")
            raise RuntimeError(f"Failed to create logit: {e}")

    @staticmethod
    def delete_logit(logit_id: str):
        """
        Deletes a logit from the database.

        Args:
            logit_id (str): The ID of the logit to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the logit from the database.
        """
        try:
            with LogitController.database_transaction():
                delete_result = _get_logits_collection().delete_one(
                    {"id": str(logit_id)}
                )
                if delete_result.deleted_count == 0:
                    logger.warning(f"Logit with ID {logit_id} not found for deletion.")
                    raise ValueError(f"Logit with ID {logit_id} not found.")

                return {"message": f"Logit with ID {logit_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting logit with ID {logit_id}: {e}")
            raise RuntimeError(f"Failed to delete logit with ID {logit_id}: {e}")
