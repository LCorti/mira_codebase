from typing import Any

import httpx

from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="seeders.log")


async def seed_projects(
    client: httpx.AsyncClient,
    base_url: str,
    projects_data: list[dict[str, Any]],
    auto_run: bool = False,
):
    """
    Seed projects using HTTP requests to the projects API endpoint.

    Args:
        client (httpx.AsyncClient): HTTP client instance
        base_url (str): Base URL of the FastAPI application
        projects_data (list[dict[str, Any]]): List of project data to seed
        auto_run (bool): Whether to auto-run tests after creating projects

    Returns:
        dict[str, int]: Dictionary with success and error counts
    """
    logger.info(f"Starting to seed {len(projects_data)} projects")

    success_count = 0
    error_count = 0

    for project_data in projects_data:
        try:
            # Make HTTP POST request to create project
            url = f"{base_url}/projects"
            params = {"auto_run": auto_run} if auto_run else {}

            response = await client.post(url, json=project_data, params=params)

            if response.status_code == 200:
                _ = response.json()
                project_name = project_data.get("name", "Unknown")
                logger.info(f"Successfully created project: {project_name}")
                success_count += 1
            else:
                project_name = project_data.get("name", "Unknown")
                logger.error(
                    f"Failed to create project {project_name}. Status: {response.status_code}, Response: {response.text}"
                )
                error_count += 1

        except Exception as e:
            project_name = project_data.get("name", "Unknown")
            logger.error(f"Failed to create project {project_name}: {e}")
            error_count += 1

    logger.info(
        f"Project seeding completed. Success: {success_count}, Errors: {error_count}"
    )
    return {"success": success_count, "errors": error_count}
