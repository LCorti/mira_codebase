from contextlib import contextmanager
from typing import Any

from app.config.registry import get_database_service, get_job_queue
from app.models.job import Job, JobStatus, JobType, JobUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("jobs")


def _get_results_collection():
    """
    Returns the results collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("results")


class JobController:
    """
    Controller class for handling job-related business logic.
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
    def get_all_jobs(include: set[str] | None) -> list[dict[str, Any]]:
        """
        Retrieves all jobs from the database.

        Args:
            include (set[str] | None): Fields to include in the serialized job data.

        Returns:
            list[dict]: List of all jobs serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving jobs from the database.
        """
        try:
            jobs = _get_jobs_collection().find()
            return [Job(**job).to_dict(include=include) for job in jobs]
        except Exception as e:
            logger.error(f"Error retrieving jobs from the database: {e}")
            raise RuntimeError(f"Failed to retrieve jobs from the database: {e}")

    @staticmethod
    def get_job_by_id(job_id: str, include: set[str] | None) -> dict[str, Any]:
        """
        Retrieves a specific job by its ID.

        Args:
            job_id (str): The ID of the job to retrieve.

        Returns:
            dict: The job serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the job from the database.
        """
        try:
            job = _get_jobs_collection().find_one({"id": job_id})
            if not job:
                raise ValueError(f"Job with ID {job_id} not found.")
            return Job(**job).to_dict(include=include)
        except Exception as e:
            logger.error(f"Error retrieving job with ID {job_id}: {e}")
            raise RuntimeError(f"Failed to retrieve job with ID {job_id}: {e}")

    @staticmethod
    def get_job_by_ids(job_ids: list[str], include: set[str] | None) -> dict[str, Any]:
        """
        Retrieves a specific list of jobs by their IDs.

        Args:
            job_ids (list[str]): The IDs of the jobs to retrieve.
            include (set[str] | None): The relations to include in the response.

        Returns:
            dict: Dictionary containing 'jobs' (list of found jobs) and 'not_found' (list of missing job IDs).

        Raises:
            RuntimeError: If there is an error retrieving the jobs from the database.
        """
        # If no job IDs are provided, return empty result
        if not job_ids:
            return {"jobs": [], "not_found": []}

        jobs = []
        not_found = []
        for job_id in job_ids:
            try:
                job = _get_jobs_collection().find_one({"id": job_id})
                if not job:
                    logger.warning(f"Job with ID {job_id} not found.")
                    not_found.append(job_id)
                    continue
                jobs.append(Job(**job).to_dict(include=include))
            except Exception as e:
                logger.error(f"Error retrieving job with ID {job_id}: {e}")
                raise RuntimeError(f"Failed to retrieve job with ID {job_id}: {e}")
        return {"jobs": jobs, "not_found": not_found}

    @staticmethod
    def create_job(
        job: Job, job_type: str = "FULL", overwrite_parameters: bool = True
    ) -> dict[str, Any]:
        """
        Creates a new job in the database.

        Args:
            job (Job): The job data to create.

        Returns:
            dict: The created job serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the job in the database.
        """
        try:
            with JobController.database_transaction():
                # Find method
                db_method = _get_methods_collection().find_one({"id": job.method_id})
                if not db_method:
                    logger.warning(f"Method with ID {job.method_id} not found.")
                    raise ValueError(f"Method with ID {job.method_id} not found.")

                from app.models.method import Method

                method = Method(**db_method)

                # Set method parameters as job parameters
                if method and method.parameters and overwrite_parameters:
                    job.parameters = method.parameters
                    logger.debug(
                        f"job.parameters updated with method parameters: {job.parameters}"
                    )

                job.type = JobType.from_str(job_type) if job_type else JobStatus.FULL

                # Create new job
                result = _get_jobs_collection().insert_one(job.to_dict())
                if not result.acknowledged:
                    logger.error(f"Failed to create job for method {method.id}")
                    raise RuntimeError("Failed to create job in the database.")

                # Update the method's job_ids array
                _get_methods_collection().update_one(
                    {"id": job.method_id}, {"$addToSet": {"job_ids": str(job.id)}}
                )

                # Add queue logic here
                job_queue = get_job_queue()
                if not job_queue.add_job(job):
                    logger.error(f"Failed to add job {job.id} to the queue.")
                    raise RuntimeError(f"Failed to add job {job.id} to the queue.")

                return job.to_dict()
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            raise RuntimeError(f"Failed to create job: {e}")

    @staticmethod
    def duplicate_jobs(
        job_ids: list[str], job_types: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Duplicates an existing job in the database.

        Args:
            job_id (str): The ID of the job to duplicate.

        Returns:
            dict: The duplicated job serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error duplicating the job in the database.
        """
        if job_types is None:
            job_types = []

        if len(job_ids) != len(job_types) and job_types:
            logger.warning("Length mismatch between job_ids and job_types.")
            raise ValueError("Length mismatch between job_ids and job_types.")
        if not job_types:
            logger.debug("No job_types provided, defaulting all to 'FULL'.")
            job_types = ["FULL"] * len(job_ids)
        try:
            duplicated_jobs = []
            with JobController.database_transaction():
                for job_id, job_type in zip(job_ids, job_types):
                    logger.info(f"Duplicating job {job_id} with type {job_type}.")
                    # Fetch the existing job
                    db_job = _get_jobs_collection().find_one({"id": job_id})
                    if not db_job:
                        logger.warning(
                            f"Job with ID {job_id} not found for duplication."
                        )
                        continue  # Skip this job_id

                    current_job = Job(**db_job)
                    current_method = current_job.method()

                    # If job_type is ANALYSIS, duplicate result object
                    if job_type == "ANALYSIS":
                        current_result = current_job.result()
                        from app.models.result import Result

                        new_result = Result(
                            parameters=current_method.parameters,
                            method_id=current_result.method_id,
                            results=[
                                {k: v for k, v in r.items() if k != "metrics"}
                                for r in current_result.results
                            ],
                        )

                    # Create a new job with the same data
                    duplicated_job = Job(
                        method_id=current_job.method_id,
                        result_id=new_result.id if job_type == "ANALYSIS" else "",
                        parameters=current_method.parameters
                        if job_type == "ANALYSIS"
                        else current_job.parameters,
                        type=job_type if job_type else current_job.type,
                        status=JobStatus.PENDING,
                    )

                    # Update the new result's job_id & insert into database
                    if job_type == "ANALYSIS":
                        new_result.job_id = duplicated_job.id
                        result = _get_results_collection().insert_one(
                            new_result.to_dict()
                        )
                        if not result.acknowledged:
                            logger.error(
                                f"Failed to insert duplicated result for job {duplicated_job.id}"
                            )
                            continue  # Skip this job_id

                    # Insert the duplicated job into the database
                    result = _get_jobs_collection().insert_one(duplicated_job.to_dict())
                    if not result.acknowledged:
                        logger.error(
                            f"Failed to insert duplicated job for method {current_job.method_id}"
                        )
                        continue  # Skip this job_id

                    # Update the method's job_ids array
                    _get_methods_collection().update_one(
                        {"id": current_job.method_id},
                        {"$addToSet": {"job_ids": str(duplicated_job.id)}},
                    )

                    # Add queue logic here
                    job_queue = get_job_queue()
                    if not job_queue.add_job(duplicated_job):
                        logger.error(
                            f"Failed to add job {duplicated_job.id} to the queue."
                        )
                        continue  # Skip adding to result if queue fails

                    duplicated_jobs.append(duplicated_job.to_dict())
            return duplicated_jobs
        except Exception as e:
            logger.error(f"Error duplicating job: {e}")
            raise RuntimeError(f"Failed to duplicate job: {e}")

    @staticmethod
    def create_jobs(jobs: list[Job], job_types: list[str]) -> list[dict[str, Any]]:
        """
        Creates multiple new jobs in the database.

        Args:
            jobs (List[Job]): The list of job data to create.

        Returns:
            List[Dict[str, Any]]: The created jobs serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error creating the jobs in the database.
        """

        created_jobs = []
        for job, job_type in zip(jobs, job_types):
            try:
                created_job = JobController.create_job(job, job_type=job_type)
                created_jobs.append(created_job)
            except Exception as e:
                logger.error(f"Error creating job from bulk create: {e}")
                continue
        return created_jobs

    @staticmethod
    def update_job(job_id: str, job: JobUpdate) -> dict[str, Any]:
        """
        Updates an existing job in the database.

        Args:
            job_id (str): The ID of the job to update.
            job (JobUpdate): The updated job data.

        Returns:
            dict: The updated job serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the job in the database.
        """
        try:
            with JobController.database_transaction():
                # Fetch the existing job
                current_job = _get_jobs_collection().find_one({"id": job_id})
                if not current_job:
                    logger.warning(f"Job with ID {job_id} not found for update.")
                    raise ValueError(f"Job with ID {job_id} not found.")

                # Convert job to Job model for type consistency
                current_job = Job(**current_job)

                # Update the job's method_id if it has changed
                if job.method_id and current_job.method_id != job.method_id:
                    _get_methods_collection().update_one(
                        {"id": current_job.method_id},
                        {"$pull": {"job_ids": str(job_id)}},
                    )
                    _get_methods_collection().update_one(
                        {"id": job.method_id}, {"$addToSet": {"job_ids": str(job_id)}}
                    )

                # Update the job
                result = _get_jobs_collection().update_one(
                    {"id": job_id}, {"$set": job.to_dict()}
                )
                if result.matched_count == 0:
                    raise ValueError(f"Job with ID {job_id} not found.")
                updated_job = _get_jobs_collection().find_one({"id": job_id})

                return Job(**updated_job).to_dict()
        except Exception as e:
            logger.error(f"Error updating job with ID {job_id}: {e}")
            raise RuntimeError(f"Failed to update job with ID {job_id}: {e}")

    @staticmethod
    def update_job_status(
        job_id: str, status: str, status_message: str, runtime: float = 0.0
    ) -> dict[str, Any]:
        """
        Updates the status of a job.

        Args:
            job_id (str): The ID of the job to update.
            status (str): The new status of the job.

        Returns:
            dict: The updated job serialized as a dictionary.

        Raises:
            ValueError: If the job with the given ID does not exist.
            RuntimeError: If there is an error updating the job status in the database.
        """
        try:
            with JobController.database_transaction():
                job_status = JobStatus.from_str(status)
                result = _get_jobs_collection().update_one(
                    {"id": job_id},
                    {
                        "$set": {
                            "status": job_status.value,
                            "status_message": status_message,
                            "runtime": round(runtime, 2),
                        }
                    },
                )
                if result.matched_count == 0:
                    logger.warning(f"Job with ID {job_id} not found for status update.")
                    raise ValueError(f"Job with ID {job_id} not found.")

                updated_job = _get_jobs_collection().find_one({"id": job_id})

                return Job(**updated_job).to_dict()
        except Exception as e:
            logger.error(f"Error updating job status for ID {job_id}: {e}")
            raise RuntimeError(f"Failed to update job status for ID {job_id}: {e}")

    @staticmethod
    def cancel_job(job_id: str) -> dict[str, Any]:
        """
        Cancels a job by marking it as cancelled and removing it from the queue.

        Args:
            job_id (str): The ID of the job to cancel.

        Returns:
            dict: A message indicating the cancellation status.

        Raises:
            ValueError: If the job with the given ID does not exist.
            RuntimeError: If there is an error cancelling the job.
        """
        try:
            # Check if job exists and get its current status
            job_doc = _get_jobs_collection().find_one({"id": job_id})
            if not job_doc:
                raise ValueError(f"Job with ID {job_id} not found.")

            job = Job(**job_doc)

            # Check if job can be cancelled
            if job.status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                return {
                    "message": f"Job {job_id} cannot be cancelled because it is already {job.status.value}"
                }

            # Cancel the job in the queue
            from app.config.registry import get_job_queue

            job_queue = get_job_queue()
            success = job_queue.cancel_job(job_id)

            if success:
                return {"message": f"Job {job_id} has been successfully cancelled"}
            else:
                raise RuntimeError(f"Failed to cancel job {job_id}")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            raise RuntimeError(f"Failed to cancel job {job_id}: {e}")

    @staticmethod
    def delete_job(job_id: str):
        """
        Deletes a job from the database.

        Args:
            job_id (str): The ID of the job to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the job from the database.
        """
        try:
            with JobController.database_transaction():
                # Fetch the job to get its method_id
                job = _get_jobs_collection().find_one({"id": job_id})
                if not job:
                    logger.warning(f"Job with ID {job_id} not found for deletion.")
                    raise ValueError(f"Job with ID {job_id} not found.")

                # Convert job to Job model for type consistency
                job_obj = Job(**job)

                # Remove job_id from the method's job_ids array
                _get_methods_collection().update_one(
                    {"id": job_obj.method_id}, {"$pull": {"job_ids": job_id}}
                )

                # Delete the job
                result = _get_jobs_collection().delete_one({"id": job_id})
                if result.deleted_count == 0:
                    raise ValueError(f"Job with ID {job_id} not found.")

                return {"message": f"Job with ID {job_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting job with ID {job_id}: {e}")
            raise RuntimeError(f"Failed to delete job with ID {job_id}: {e}")
