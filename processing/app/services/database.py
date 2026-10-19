import os

from pymongo import MongoClient

from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)


class DatabaseService:
    def __init__(
        self,
        host=None,
        port=None,
        db_name="mira-processing",
        username=None,
        password=None,
    ):
        """
        Initializes the DatabaseService with a MongoDB connection.
        Args:
            host (str): The hostname of the MongoDB server. Defaults to MONGO_HOST env var or 'processing_db'.
            port (int): The port number of the MongoDB server. Defaults to MONGO_PORT env var or 27017.
            db_name (str): The name of the database to connect to
            username (str): MongoDB username. Defaults to MONGO_USERNAME env var.
            password (str): MongoDB password. Defaults to MONGO_PASSWORD env var.
        """
        # Use environment variables if not explicitly provided
        if host is None:
            host = os.getenv("MONGO_HOST", "processing_db")
        if port is None:
            port = int(os.getenv("MONGO_PORT", "27017"))
        if username is None:
            username = os.getenv("MONGO_USERNAME")
        if password is None:
            password = os.getenv("MONGO_PASSWORD")

        # Connect with authentication if credentials are provided
        if username and password:
            self.__client = MongoClient(
                host=host,
                port=port,
                username=username,
                password=password,
                authSource="admin",  # MongoDB default auth database
            )
            logger.info(
                f"DatabaseService initialized with host={host}, port={port}, db_name={db_name} (authenticated)"
            )
        else:
            self.__client = MongoClient(host, port)
            logger.info(
                f"DatabaseService initialized with host={host}, port={port}, db_name={db_name} (no auth)"
            )

        self.__db = self.__client[db_name]

    def get_client(self):
        """
        Returns the MongoDB client.
        Returns:
            MongoClient: The MongoDB client instance.
        """
        return self.__client

    def start_session(self):
        """
        Starts a new MongoDB session.
        Returns:
            A new session object.
        """
        return self.__client.start_session()

    def get_collection(self, collection_name):
        """
        Returns a collection from the database.
        :param collection_name: Name of the collection to retrieve.
        :return: Collection object.
        """
        return self.__db[collection_name]

    def close(self) -> bool:
        """
        Closes the database connection.
        """
        try:
            self.__client.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
            return False
        return True
