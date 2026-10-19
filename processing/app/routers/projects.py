from fastapi import APIRouter, Query

from app.services.logging_service import LoggingService

from app.controllers.project_controller import ProjectController

from app.models.project import Project, ProjectUpdate

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
)

@router.get("", tags=["projects"])
async def index(
    include: set[str] | None = Query(None, description="Relations to include"),
):
    """
    Returns a list of all projects in the database.
    Args:
        None
    Returns:
        list[Project]: List of all projects serialized as Project models.
    Raises:
        RuntimeError: If there is an error retrieving projects from the database.
    """    
    return ProjectController.get_all_projects(include=include)

@router.get("/{project_id}", tags=["projects"])
async def show(
    project_id: str,
    include: set[str] | None = Query(None, description="Relations to include"),
):
    """
    Returns a specific project by its ID.
    Args:
        project_id (str): The ID of the project to retrieve.
        include (set[str] | None): The relations to include in the response.
    Returns:
        Project: The project serialized as a Project model.
    Raises:
        RuntimeError: If there is an error retrieving the project from the database.
    """
    return ProjectController.get_project_by_id(project_id, include=include)

@router.post("", tags=["projects"])
async def create(
    project: Project
):
    """
    Creates a new project in the database.
    Args:
        project: The project data to create.
        datasets: The datasets associated with the project.
        metrics: The metrics to evaluate the project.
        auto_run: If True, automatically runs the created methods.
    Returns:
        Project: The created project serialized as a Project model.
    Raises:
        RuntimeError: If there is an error creating the project in the database.
    """
    return ProjectController.create_project(project)

@router.post("/duplicate-tests-and-methods", status_code=201)
async def duplicate_tests_and_methods(
    project_id: str,
    test_ids: list[str],
    auto_run: bool = False,
    group_id: str | None = None,
):
    """
    Adds duplicated tests and methods to the project.
    Args:
        project: The project to update.
        test_ids: The IDs of the tests to duplicate.
        auto_run: If True, automatically runs the created methods.
        group_id: Optional group ID to associate with the duplicated tests and methods.
    Returns:
        Project: The created project, tests, and methods.
    Raises:
        RuntimeError: If there is an error creating the project in the database.
    """
    project_dict = ProjectController.get_project_by_id(project_id, include={})
    return ProjectController.duplicate_tests_and_methods(test_ids=test_ids, project=Project(**project_dict), auto_run=auto_run, group_id=group_id)

@router.post("/create-with-tests-and-methods", status_code=201)
async def create_with_tests_and_methods(
    project: Project,
    datasets: list[str],
    metrics: dict[str, list[str]],
    auto_run: bool = False
):
    """
    Creates a new project in the database along with associated tests and methods.
    Args:
        project: The project data to create.
        datasets: The datasets associated with the project.
        metrics: The metrics to evaluate the project.
        auto_run: If True, automatically runs the created methods.
    Returns:
        Project: The created project, tests, and methods
    Raises:
        RuntimeError: If there is an error creating the project in the database.
    """
    return ProjectController.create_project_with_tests_and_methods(project, datasets, metrics, auto_run)

@router.put("/{project_id}", tags=["projects"])
async def update(
    project_id: str, 
    project: ProjectUpdate
):
    """
    Updates an existing project in the database.
    Args:
        project_id (str): The ID of the project to update.
        project (ProjectUpdate): The updated project data.
    Returns:
        Project: The updated project serialized as a Project model.
    Raises:
        RuntimeError: If there is an error updating the project in the database.
    """
    return ProjectController.update_project(project_id, project)

@router.delete("/{project_id}", tags=["projects"])
async def delete(
    project_id: str
):
    """
    Deletes a project from the database.
    Args:
        project_id (str): The ID of the project to delete.
    Returns:
        dict: A message indicating the deletion status.
    Raises:
        RuntimeError: If there is an error deleting the project from the database.
    """
    return ProjectController.delete_project(project_id)