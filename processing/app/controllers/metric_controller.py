from contextlib import contextmanager

from app.config.registry import get_database_service, get_metric_registry
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


class MetricController:
    """
    Controller class for handling metric-related business logic.
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
    def get_all_metrics():
        """
        List all available metrics.

        Args:
            None

        Returns:
            dict: A dictionary containing the list of available metric names.

        Raises:
            RuntimeError: If there is an error listing metrics.
        """
        try:
            metrics = get_metric_registry().list_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Error listing metrics: {e!s}")
            raise RuntimeError(f"Failed to list metrics: {e!s}") from e

    @staticmethod
    def get_all_metadata():
        """
        Get metadata for all registered metrics.

        Args:
            None

        Returns:
            dict: A dictionary mapping metric names to their metadata.

        Raises:
            RuntimeError: If there is an error retrieving all metrics.
        """
        try:
            return get_metric_registry().get_all_metadata()
        except Exception as e:
            logger.error(f"Error retrieving all metrics: {e!s}")
            raise RuntimeError(f"Failed to retrieve all metrics: {e!s}") from e

    @staticmethod
    def get_metadata_by_name(metric_name: str):
        """
        Get metadata for a specific metric.

        Args:
            metric_name (str): The name of the metric to retrieve metadata for.

        Returns:
            dict: Metadata dictionary for the specified metric.

        Raises:
            ValueError: If the metric is not found.
            RuntimeError: If there is an error retrieving metadata.
        """
        try:
            metadata = get_metric_registry().get_metadata(metric_name)
            return {"metric": metadata}
        except ValueError:
            logger.error(f"Metric not found: {metric_name}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving metadata for metric '{metric_name}': {e!s}")
            raise RuntimeError(
                f"Failed to retrieve metadata for metric '{metric_name}': {e!s}"
            ) from e

    @staticmethod
    def find_metrics_by_task(task_types: list[str]):
        """
        Find metrics suitable for specific task types.

        Args:
            task_types (list[str]): A list of task types to find metrics for (e.g., ["classification", "text_generation"]). To narrow down the search, you can specify sub-task types as well (e.g., "text-simplification").

        Returns:
            list: A list of metric names suitable for any of the specified task types.

        Raises:
            ValueError: If no metrics are found for the task types.
            RuntimeError: If there is an error finding metrics.
        """
        try:
            metrics = get_metric_registry().find_metrics_by_task(task_types)

            if not metrics:
                return []

            return metrics
        except Exception as e:
            logger.error(f"Error finding metrics for task types '{task_types}': {e!s}")
            raise RuntimeError(
                f"Failed to find metrics for task types '{task_types}': {e!s}"
            ) from e

    @staticmethod
    def get_metric_by_name(metric_name: str):
        """
        Get details of a specific metric by name.

        Args:
            metric_name (str): The name of the metric to retrieve.

        Returns:
            dict: Metadata of the specified metric.

        Raises:
            ValueError: If the metric is not found.
            RuntimeError: If there is an error retrieving the metric.
        """
        try:
            metadata = get_metric_registry().get_metadata(metric_name)
            return {"metric": metadata}
        except ValueError:
            logger.error(f"Metric not found: {metric_name}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving metric '{metric_name}': {e!s}")
            raise RuntimeError(
                f"Failed to retrieve metric '{metric_name}': {e!s}"
            ) from e

    @staticmethod
    def create_custom_metric(
        metric_name: str,
        description: str | None = None,
        metric_kwargs: dict | None = None,
        metadata: dict | None = None,
        register_name: str | None = None,
    ):
        """
        Create a custom metric by loading a Hugging Face evaluate metric.

        Args:
            metric_name (str): The name of the Hugging Face metric to load.
            description (str, optional): Custom description for the metric.
            metric_kwargs (dict): Additional keyword arguments for loading the metric.
            metadata (dict, optional): Metadata for the metric.
            register_name (str, optional): Name to register the metric under (default: same as metric_name).

        Returns:
            dict: Details of the created custom metric.

        Raises:
            RuntimeError: If there is an error creating the metric.
        """
        if not metric_kwargs:
            metric_kwargs = {}

        try:
            with MetricController.database_transaction():
                # Use register_name if provided, otherwise use metric_name
                name = register_name or metric_name

                custom_metric = get_metric_registry().create_custom_metric(
                    metric_name=metric_name, description=description, **metric_kwargs
                )

                # Save metric to config for persistence
                get_metric_registry().save_metric_to_config(custom_metric, metadata)

                return {
                    "metric": custom_metric.get_metric_info(),
                    "registered_as": name,
                }
        except RuntimeError as re:
            logger.error(f"Failed to create custom metric '{metric_name}': {re!s}")
            raise
        except Exception as e:
            logger.error(f"Error creating custom metric '{metric_name}': {e!s}")
            raise RuntimeError(
                f"Failed to create custom metric '{metric_name}': {e!s}"
            ) from e

    @staticmethod
    def delete_custom_metric(metric_name: str):
        """
        Delete a custom metric from both memory and config.

        Args:
            metric_name (str): The name of the metric to delete.

        Returns:
            dict: Confirmation of deletion.

        Raises:
            ValueError: If the metric is not found.
            RuntimeError: If there is an error deleting the metric.
        """
        try:
            with MetricController.database_transaction():
                get_metric_registry().remove_custom_metric_from_config(metric_name)
                return {
                    "message": f"Custom metric '{metric_name}' deleted successfully."
                }
        except ValueError:
            logger.error(f"Metric not found: {metric_name}")
            raise
        except Exception as e:
            logger.error(f"Error deleting metric '{metric_name}': {e!s}")
            raise RuntimeError(f"Failed to delete metric '{metric_name}': {e!s}") from e
