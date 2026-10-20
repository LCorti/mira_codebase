import json
import os

from app.config.paths import CONFIG_DIR
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)


class TaskService:
    def __init__(
        self,
        config_path=None,
        available_tasks_config_path=None,
    ):
        """
        Initializes the TaskService with the directory for tasks and the configuration file path.
        Args:
            config_path (str): Path to the configuration file containing task information.
        Raises:
            FileNotFoundError: If the specified task directory or configuration file does not exist.
        """

        if not config_path:
            self.__config_path = os.path.join(CONFIG_DIR, "tasks.json")
        else:
            self.__config_path = config_path
        if not available_tasks_config_path:
            self.__available_tasks_config_path = os.path.join(
                CONFIG_DIR, "available_tasks.json"
            )
        else:
            self.__available_tasks_config_path = available_tasks_config_path
        self.__available_tasks = None
        self.__tasks = None

        # Check if config file exists
        if not os.path.exists(self.__config_path):
            logger.error(f"Config file not found: {self.__config_path}. ")
            raise FileNotFoundError(f"Config file not found: {self.__config_path}.")

        if not os.path.exists(self.__available_tasks_config_path):
            logger.error(
                f"Available tasks config file not found: {self.__available_tasks_config_path}."
            )
            raise FileNotFoundError(
                f"Available tasks config file not found: {self.__available_tasks_config_path}."
            )

        config = self.__open_config(self.__config_path)
        if not config or not config.items():
            logger.error(f"Config file {self.__config_path} is empty or invalid.")
            raise ValueError(f"Config file {self.__config_path} is empty or invalid.")
        self.__tasks = config

        available_config = self.__open_config(self.__available_tasks_config_path)
        if not available_config or not available_config.items():
            logger.error(
                f"Available tasks config file {self.__available_tasks_config_path} is empty or invalid."
            )
            raise ValueError(
                f"Available tasks config file {self.__available_tasks_config_path} is empty or invalid."
            )
        self.__available_tasks = available_config

        logger.info("TaskService initialized.")

    def __open_config(self, config_path) -> dict:
        """
        Opens the configuration file and returns its content.
        Returns:
            dict: The content of the configuration file.
        Raises:
            OSError: If there is an error reading the configuration file.
            json.JSONDecodeError: If the configuration file is not a valid JSON.
        """
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error opening config file: {e}")
            raise

    def close(self):
        """
        Closes the dataset loader service.
        This method can be used to perform any cleanup if necessary.
        """
        self.__tasks = None
        logger.info("TaskService has been closed.")

    ### CRUD ###
    def index(self) -> dict[str, dict]:
        """
        Returns a list of all tasks in the system.
        Returns:
            dict[str, dict]: A dictionary of all tasks.
        Raises:
            RuntimeError: If there is an error retrieving tasks.
        """
        try:
            return self.__tasks
        except Exception as e:
            logger.error(f"Error retrieving tasks: {e}")
            raise RuntimeError(f"Failed to retrieve tasks: {e}")

    def show(self, task_name: str) -> dict[str, dict]:
        """
        Returns the details of a specific task.
        Args:
            task_name (str): The name of the task to retrieve.
        Returns:
            dict: Details of the specified task.
        Raises:
            RuntimeError: If there is an error retrieving the task.
        """
        task = {task_name: {}}
        try:
            if task_name not in self.__tasks:
                logger.error(f"Task '{task_name}' not found.")
                raise ValueError(f"Task '{task_name}' not found.")

            task[task_name] = self.__tasks[task_name]
            logger.debug(f"Task '{task_name}' found with details: {task}")
            return task
        except Exception as e:
            logger.error(f"Error retrieving task '{task_name}': {e}")
            raise RuntimeError(f"Failed to retrieve task '{task_name}': {e}")

    def show_multiple(self, task_names: list[str]) -> dict[str, dict]:
        """
        Returns the details of multiple tasks.
        Args:
            task_names (list[str]): The names of the tasks to retrieve.
        Returns:
            dict: Details of the specified tasks.
        Raises:
            RuntimeError: If there is an error retrieving the tasks.
        """
        tasks = {}
        try:
            for task_name in task_names:
                if task_name not in self.__tasks:
                    logger.warning(f"Task '{task_name}' not found, skipping.")
                    continue

                tasks[task_name] = self.__tasks[task_name]
            logger.debug(f"Tasks '{task_names}' found with details: {tasks}")
            return tasks
        except Exception as e:
            logger.error(f"Error retrieving tasks '{task_names}': {e}")
            raise RuntimeError(f"Failed to retrieve tasks '{task_names}': {e}")

    def get_available_tasks(self) -> dict[str, list[str]]:
        """
        Returns a list of all available task types in the system.
        Returns:
            list[str]: A list of available task types.
        Raises:
            RuntimeError: If there is an error retrieving available tasks.
        """
        try:
            return self.__available_tasks
        except Exception as e:
            logger.error(f"Error retrieving available tasks: {e}")
            raise RuntimeError(f"Failed to retrieve available tasks: {e}")
