from typing import Annotated

from fastapi import APIRouter, Query

from app.controllers.job_controller import JobController
from app.models.job import Job, JobUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="routes.log")

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
)


@router.get("", status_code=200)
async def index(
    include: Annotated[
        set[str] | None, Query(description="Relations to include")
    ] = None,
):
    """
    Returns a list of all jobs in the database.
    Args:
        None
    Returns:
        list[Job]: List of all jobs serialized as Job models.
    Raises:
        RuntimeError: If there is an error retrieving jobs from the database.
    """
    return JobController.get_all_jobs(include=include)


@router.get("/{job_id}", status_code=200)
async def show(
    job_id: str,
    include: Annotated[set[str] | None, Query(description="Relations to include")],
):
    """
    Returns a specific job by its ID.
    Args:
        job_id (str): The ID of the job to retrieve.
        include (set[str] | None): The relations to include in the response.
    Returns:
        Job: The job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error retrieving the job from the database.
    """
    return JobController.get_job_by_id(job_id, include=include)


@router.get("/batch/show", status_code=200)
async def batch_show(
    job_ids: Annotated[list[str], Query(description="List of job IDs to retrieve")],
    include: Annotated[
        set[str] | None, Query(description="Relations to include")
    ] = None,
):
    """
    Returns a list of jobs by their IDs.
    Args:
        job_ids (list[str]): The IDs of the jobs to retrieve.
    Returns:
        list[Job]: The jobs serialized as Job models.
    Raises:
        RuntimeError: If there is an error retrieving the jobs from the database.
    """
    return JobController.get_job_by_ids(job_ids, include=include)


@router.post("", status_code=201)
async def create(job: Job):
    """
    Creates a new job in the database.
    Args:
        job (Job): The job data to create.
    Returns:
        Job: The created job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error creating the job in the database.
    """
    return JobController.create_job(job)


@router.post("/duplicate", status_code=201)
async def duplicate(job_ids: list[str], job_types: list[str] | None = None):
    """
    Duplicates an existing job in the database.
    Args:
        job_id (str): The ID of the job to duplicate.
    Returns:
        Job: The duplicated job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error duplicating the job in the database.
    """
    if not job_types:
        job_types = []
    return JobController.duplicate_jobs(job_ids, job_types)


@router.post("/duplicate-lxt", status_code=201)
async def duplicate_lxt(job_id: str, sample: dict | None = None):
    """
    Creates a new LXT diagnostic job by based on an existing job's configuration.

    Args:
        job_id (str): The ID of the LXT job to duplicate.
    Returns:
        Job: The duplicated LXT job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error duplicating the LXT job in the database.
    """
    # Retrieve the existing job
    job = JobController.get_job_by_id(job_id, include=set())

    parameters = job["parameters"].copy()
    parameters["task_parameters"]["diagnostic_only"] = True
    parameters["task_parameters"]["diagnostic_type"] = "lxt"
    if sample:
        parameters["task_parameters"]["diagnostic_sample"] = sample

    new_job = Job(
        parameters=parameters,
        method_id=job["method_id"],
        result_id=job["result_id"],
    )

    new_job_object = JobController.create_job(
        new_job, job_type="DIAGNOSIS", overwrite_parameters=False
    )
    if not new_job_object:
        logger.error("Failed to create duplicated LXT job.")
        raise RuntimeError("Failed to create duplicated LXT job.")

    return new_job_object


@router.post("/duplicate-shap", status_code=201)
async def duplicate_shap(job_id: str, sample: dict | None = None):
    """
    Creates a new SHAP diagnostic job based on an existing job's configuration.

    Args:
        job_id (str): The ID of the SHAP job to duplicate.
    Returns:
        Job: The duplicated SHAP job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error duplicating the SHAP job in the database.
    """
    # Retrieve the existing job
    job = JobController.get_job_by_id(job_id, include=set())

    parameters = job["parameters"].copy()
    parameters["task_parameters"]["diagnostic_only"] = True
    parameters["task_parameters"]["diagnostic_type"] = "shap"
    if sample:
        parameters["task_parameters"]["diagnostic_sample"] = sample

    new_job = Job(
        parameters=parameters,
        method_id=job["method_id"],
        result_id=job["result_id"],
    )

    new_job_object = JobController.create_job(
        new_job, job_type="DIAGNOSIS", overwrite_parameters=False
    )
    if not new_job_object:
        logger.error("Failed to create duplicated LXT job.")
        raise RuntimeError("Failed to create duplicated LXT job.")

    return new_job_object


@router.post("/bulk-create", status_code=201)
async def bulk_create(jobs: list[Job], job_types: list[str] | None = None):
    """
    Creates multiple new jobs in the database.
    Args:
        jobs (list[Job]): The list of job data to create.
    Returns:
        list[Job]: The created jobs serialized as Job models.
    Raises:
        RuntimeError: If there is an error creating the jobs in the database.
    """
    if not job_types:
        job_types = ["FULL"] * len(jobs)

    return JobController.create_jobs(jobs, job_types)


@router.put("/{job_id}", status_code=200)
async def update(job_id: str, job: JobUpdate):
    """
    Updates an existing job in the database.
    Args:
        job_id (str): The ID of the job to update.
        job (JobUpdate): The updated job data.
    Returns:
        Job: The updated job serialized as a Job model.
    Raises:
        RuntimeError: If there is an error updating the job in the database.
    """
    return JobController.update_job(job_id, job)


@router.post("/{job_id}/cancel", status_code=200)
async def cancel(job_id: str):
    """
    Cancels a running or pending job.
    Args:
        job_id (str): The ID of the job to cancel.
    Returns:
        dict: A message indicating the cancellation status.
    Raises:
        RuntimeError: If there is an error cancelling the job.
    """
    return JobController.cancel_job(job_id)


@router.delete("/{job_id}", status_code=204)
async def delete(job_id: str):
    """
    Deletes a job from the database.
    Args:
        job_id (str): The ID of the job to delete.
    Returns:
        dict: A message indicating the deletion status.
    Raises:
        RuntimeError: If there is an error deleting the job from the database.
    """
    return JobController.delete_job(job_id)
