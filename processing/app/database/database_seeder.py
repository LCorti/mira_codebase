import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from app.config.registry import set_database_service
from app.database.seeders import seed_methods, seed_projects, seed_tests
from app.services.database import DatabaseService
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="seeders.log")


class DatabaseSeeder:
    """
    Database seeder class that handles seeding data using HTTP requests to API endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the database seeder.

        Args:
            base_url (str): Base URL of the FastAPI application
        """
        self.seeds_dir = Path(__file__).parent.joinpath("seeds")
        self.base_url = base_url.rstrip("/")
        self.client = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def check_api_health(self) -> bool:
        """
        Check if the API server is running and accessible.

        Returns:
            bool: True if the API is accessible, False otherwise
        """
        if not self.client:
            raise RuntimeError(
                "HTTP client not initialized. Use async context manager."
            )

        try:
            response = await self.client.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False

    def load_seed_file(self, filename: str) -> list[dict[str, Any]]:
        """
        Load data from a seed file.

        Args:
            filename (str): Name of the seed file to load

        Returns:
            list[dict[str, Any]]: List of data objects from the seed file

        Raises:
            FileNotFoundError: If the seed file doesn't exist
            json.JSONDecodeError: If the seed file contains invalid JSON
        """
        seed_file_path = self.seeds_dir / filename

        if not seed_file_path.exists():
            raise FileNotFoundError(
                f"Seed file {filename} not found in {self.seeds_dir}"
            )

        try:
            with open(seed_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                logger.info(f"Loaded {len(data)} items from {filename}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in seed file {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading seed file {filename}: {e}")
            raise

    async def seed_projects_collection(
        self, projects_data: list[dict[str, Any]], auto_run: bool = False
    ):
        """
        Seed projects using the separated project seeder.

        Args:
            projects_data (list[dict[str, Any]]): List of project data to seed
            auto_run (bool): Whether to auto-run tests after creating projects
        """
        if not self.client:
            raise RuntimeError(
                "HTTP client not initialized. Use async context manager."
            )

        return await seed_projects(self.client, self.base_url, projects_data, auto_run)

    def seed_tests_collection(self, tests_data: list[dict[str, Any]]):
        """
        Seed tests using the separated test seeder (direct database access).

        Args:
            tests_data (list[dict[str, Any]]): List of test data to seed
        """
        return seed_tests(tests_data)

    def seed_methods_collection(self, methods_data: list[dict[str, Any]]):
        """
        Seed methods using the separated method seeder (direct database access).

        Args:
            methods_data (list[dict[str, Any]]): List of method data to seed
        """
        return seed_methods(methods_data)

    async def seed_all(self, seed_files: dict[str, dict[str, Any]]):
        """
        Seed all collections with data from specified seed files.

        :param seed_files: Dictionary mapping collection names to their seed file configs
        :type seed_files: dict[str, dict[str, Any]]
        """
        results = {}

        try:
            logger.info("Starting database seeding process...")

            # Seed each collection
            for collection_name, config in seed_files.items():
                logger.debug(f"Seeding {collection_name}...")

                data = self.load_seed_file(config["file"])
                if (
                    "project" in collection_name.lower()
                    and "test" not in collection_name.lower()
                    and "method" not in collection_name.lower()
                ):
                    # Check if API is accessible for projects (still uses HTTP)
                    if not await self.check_api_health():
                        raise RuntimeError(
                            f"API server is not accessible at {self.base_url}. Please ensure the FastAPI server is running."
                        )
                    auto_run = config.get("auto_run", False)
                    results[collection_name] = await self.seed_projects_collection(
                        data, auto_run
                    )
                elif "test" in collection_name.lower():
                    # Tests use direct database access, no API check needed
                    results[collection_name] = self.seed_tests_collection(data)
                elif "method" in collection_name.lower():
                    # Methods use direct database access, no API check needed
                    results[collection_name] = self.seed_methods_collection(data)
                else:
                    logger.warning(
                        f"No seeding method implemented for collection: {collection_name}"
                    )
                    results[collection_name] = {
                        "success": 0,
                        "errors": 0,
                        "message": "Not implemented",
                    }

            # Check if seeding was actually successful
            total_success = sum(result.get("success", 0) for result in results.values())
            total_errors = sum(result.get("errors", 0) for result in results.values())

            if total_success == 0 and total_errors > 0:
                error_msg = (
                    f"Database seeding failed: {total_errors} errors, 0 successes"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(
                f"Database seeding completed: {total_success} successes, {total_errors} errors"
            )
            return results

        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            raise


async def main():
    import argparse
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seed the processing database")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Base URL of the processing service (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="MongoDB host (default: from MONGO_HOST env or localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="MongoDB port (default: from MONGO_PORT env or 27017)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="MongoDB username (default: from MONGO_USERNAME env)",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="MongoDB password (default: from MONGO_PASSWORD env)",
    )
    args = parser.parse_args()

    # Determine the base URL
    if args.url:
        base_url = args.url
    else:
        base_url = os.getenv("PROCESSING_SERVICE_URL", "http://localhost:8001")

    logger.info(f"Seeding database using processing service at: {base_url}")

    set_database_service(
        DatabaseService(
            host=args.db_host,
            port=args.db_port,
            username=args.db_user,
            password=args.db_password,
        )
    )

    # Define what to seed
    seed_config = {
        # Projects
        # "projects": {
        #     "file": "projects.json",
        #     "auto_run": False
        # },
        # "project_tests": {
        #     "file": "project_tests.json"
        # },
        # "project_methods": {
        #     "file": "project_methods.json"
        # },
        # SKs /& RKs
        "tests": {"file": "tests.json"},
        "methods": {"file": "methods.json"},
    }

    try:
        async with DatabaseSeeder(base_url=base_url) as seeder:
            await seeder.seed_all(seed_config)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
