from typing import Any

from app.config.registry import get_database_service
from app.controllers.test_controller import TestController
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="seeders.log")


def seed_tests(tests_data: list[dict[str, Any]]) -> dict[str, int]:
    """
    Seed tests directly into the database.

    Args:
        tests_data (list[dict[str, Any]]): List of test data to seed

    Returns:
        dict[str, int]: Dictionary with success and error counts
    """
    logger.info(f"Starting to seed {len(tests_data)} tests")

    # Get database collections
    db_service = get_database_service()
    tests_collection = db_service.get_collection("tests")
    projects_collection = db_service.get_collection("projects")
    methods_collection = db_service.get_collection("methods")

    success_count = 0
    error_count = 0

    for test_data in tests_data:
        try:
            test_name = test_data.get("name", "Unknown")
            logger.info(f"Processing test: {test_name}")

            # Handle existing id field if present (for predefined ObjectIds)
            if "id" in test_data:
                if isinstance(test_data["id"], dict) and "$oid" in test_data["id"]:
                    test_data["id"] = str(test_data["id"]["$oid"])
                elif isinstance(test_data["id"], str):
                    test_data["id"] = str(test_data["id"])

            # Validate required fields
            if not test_data.get("name"):
                raise ValueError("Test name is required")

            # Set default values if not provided
            if "description" not in test_data:
                test_data["description"] = ""
            if "parameters" not in test_data:
                test_data["parameters"] = {}
            if "project_id" not in test_data:
                test_data["project_id"] = ""
            if "method_ids" not in test_data:
                test_data["method_ids"] = []

            with TestController.database_transaction():
                # If project_id is provided and not empty, validate it exists
                if test_data.get("project_id") and test_data["project_id"].strip():
                    project = projects_collection.find_one(
                        {"id": str(test_data["project_id"])}
                    )
                    if not project:
                        logger.warning(
                            f"Project with ID {test_data['project_id']} not found for test {test_name}"
                        )
                        # Don't fail, just clear the project_id
                        test_data["project_id"] = ""

                # Validate method_ids if provided
                validated_method_ids = []
                for method_id in test_data.get("method_ids", []):
                    if method_id and method_id.strip():
                        method = methods_collection.find_one({"id": str(method_id)})
                        if method:
                            validated_method_ids.append(method_id)
                        else:
                            logger.warning(
                                f"Method with ID {method_id} not found for test {test_name}"
                            )
                test_data["method_ids"] = validated_method_ids

                # Check if test already exists (by name or id)
                existing_test = None
                if "id" in test_data:
                    existing_test = tests_collection.find_one({"id": test_data["id"]})

                if existing_test:
                    logger.info(f"Test {test_name} already exists, skipping...")
                    success_count += 1
                    continue

                # Insert the test
                result = tests_collection.insert_one(test_data)
                if not result.acknowledged:
                    raise RuntimeError("Failed to create test in the database.")

                # Update project if project_id is provided
                if test_data.get("project_id") and test_data["project_id"].strip():
                    projects_collection.update_one(
                        {"id": str(test_data["project_id"])},
                        {"$addToSet": {"test_ids": test_data["id"]}},
                    )

                # Update methods if method_ids are provided
                for method_id in validated_method_ids:
                    methods_collection.update_one(
                        {"id": str(method_id)},
                        {"$addToSet": {"test_ids": test_data["id"]}},
                    )

                logger.info(
                    f"Successfully created test: {test_name} with ID: {test_data['id']}"
                )
                success_count += 1

        except Exception as e:
            test_name = test_data.get("name", "Unknown")
            logger.error(f"Failed to create test {test_name}: {e}")
            error_count += 1

    logger.info(
        f"Test seeding completed. Success: {success_count}, Errors: {error_count}"
    )
    return {"success": success_count, "errors": error_count}
