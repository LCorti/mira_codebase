from typing import Any

from app.config.registry import get_database_service
from app.controllers.method_controller import MethodController
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="seeders.log")


def seed_methods(methods_data: list[dict[str, Any]]) -> dict[str, int]:
    """
    Seed methods directly into the database.

    Args:
        methods_data (list[dict[str, Any]]): List of method data to seed

    Returns:
        dict[str, int]: Dictionary with success and error counts
    """
    logger.info(f"Starting to seed {len(methods_data)} methods")

    # Get database collections
    db_service = get_database_service()
    methods_collection = db_service.get_collection("methods")
    tests_collection = db_service.get_collection("tests")

    success_count = 0
    error_count = 0

    for method_data in methods_data:
        try:
            method_name = method_data.get("name", "Unknown")
            logger.debug(f"Processing method: {method_name}")

            # Handle existing id field if present (for predefined ObjectIds)
            if "id" in method_data:
                if isinstance(method_data["id"], dict) and "$oid" in method_data["id"]:
                    method_data["id"] = str(method_data["id"]["$oid"])
                elif isinstance(method_data["id"], str):
                    method_data["id"] = str(method_data["id"])

            # Validate required fields
            if not method_data.get("name"):
                raise ValueError("Method name is required")

            # Set default values if not provided
            if "description" not in method_data:
                method_data["description"] = ""
            if "metrics" not in method_data:
                method_data["metrics"] = []
            if "parameters" not in method_data:
                method_data["parameters"] = {}
            if "test_id" not in method_data:
                method_data["test_id"] = ""
            if "job_ids" not in method_data:
                method_data["job_ids"] = []
            if "result_ids" not in method_data:
                method_data["result_ids"] = []

            with MethodController.database_transaction():
                # If test_id is provided and not empty, validate it exists
                if method_data.get("test_id") and method_data["test_id"].strip():
                    test = tests_collection.find_one(
                        {"id": str(method_data["test_id"])}
                    )
                    if not test:
                        logger.warning(
                            f"Test with ID {method_data['test_id']} not found for method {method_name}"
                        )
                        # Don't fail, just clear the test_id
                        method_data["test_id"] = ""
                    else:
                        # Merge test parameters with method parameters if test has parameters
                        if test.get("parameters"):
                            merged_params = {
                                **(test["parameters"] or {}),
                                **(method_data.get("parameters") or {}),
                            }
                            method_data["parameters"] = merged_params

                # Check if method already exists (by name or id)
                existing_method = None
                if "id" in method_data:
                    existing_method = methods_collection.find_one(
                        {"id": method_data["id"]}
                    )

                if existing_method:
                    logger.info(f"Method {method_name} already exists, skipping...")
                    success_count += 1
                    continue

                # Insert the method
                result = methods_collection.insert_one(method_data)
                if not result.acknowledged:
                    raise RuntimeError("Failed to create method in the database.")

                # Update test if test_id is provided
                if method_data.get("test_id") and method_data["test_id"].strip():
                    tests_collection.update_one(
                        {"id": str(method_data["test_id"])},
                        {"$addToSet": {"method_ids": method_data["id"]}},
                    )

                logger.info(
                    f"Successfully created method: {method_name} with ID: {method_data['id']}"
                )
                success_count += 1

        except Exception as e:
            method_name = method_data.get("name", "Unknown")
            logger.error(f"Failed to create method {method_name}: {e}")
            error_count += 1

    logger.info(
        f"Method seeding completed. Success: {success_count}, Errors: {error_count}"
    )
    return {"success": success_count, "errors": error_count}
