import time
from queue import Empty, Queue
from threading import Event, Thread

# from app.models.job import Job #! Cannot import Job here to avoid circular import issues !
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__)


class JobQueue:
    """
    A simple job queue to manage job processing.
    Uses a thread to process jobs in the background.
    """

    def __init__(self, num_workers: int = 1):
        """
        Initialize the job queue with a specified number of worker threads.

        :param num_workers: Number of worker threads to process jobs.
        :type num_workers: int
        :raises ValueError: If num_workers is less than 1.
        """

        self.__job_queue = Queue()
        self.__stop_event = Event()
        self.__threads = []
        self.__cancelled_jobs = set()  # Track cancelled job IDs
        for _ in range(num_workers):
            try:
                thread = Thread(target=self.__worker, daemon=True)
                thread.start()
                self.__threads.append(thread)
            except Exception as e:
                logger.error(f"Error starting worker thread: {e}")
                logger.fatal("Failed to initialize JobQueue, exiting.")
                raise RuntimeError(f"Failed to start worker thread: {e}")

        logger.info(f"JobQueue initialized with {num_workers} worker(s).")

        self.__add_pending_jobs()

    def __worker(self):
        while not self.__stop_event.is_set():
            try:
                job = self.__job_queue.get(timeout=1)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error getting job from queue: {e}")
                continue

            # Check if job was cancelled before starting execution
            if self.is_job_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled before execution")
                self.__job_queue.task_done()
                continue

            try:
                logger.debug(f"Processing job {job.id}...")

                # Update job status to PROCESSING
                from app.controllers.job_controller import JobController

                JobController.update_job_status(
                    job_id=job.id,
                    status="PROCESSING",
                    status_message="Job is currently running",
                )

                # Record start time for inference duration
                inference_start_time = time.time()

                # Pass the queue instance to the job so it can check for cancellation
                finished_job = job.execute(job_queue=self)

                # Calculate inference duration
                inference_duration = time.time() - inference_start_time
                inference_duration_str = self.format_run_time(inference_duration)
                logger.debug(f"Inference completed in {inference_duration_str} seconds")

                current_job = JobController.get_job_by_id(job.id, include=set())
                current_status = current_job.get("status") if current_job else None
                current_status_message = (
                    current_job.get("status_message") if current_job else ""
                )
                if self.is_job_cancelled(
                    job.id
                ):  # Check if job was cancelled during execution
                    logger.info(f"Job {job.id} was cancelled during execution")
                elif current_status == "WARNING":  # If job has warnings
                    if "DIAGNOSTICS_PLACEHOLDER" in current_status_message:
                        current_status_message = current_status_message.replace(
                            " DIAGNOSTICS_PLACEHOLDER.", ""
                        )

                    if "TIME_PLACEHOLDER" in current_status_message:
                        current_status_message = current_status_message.replace(
                            "TIME_PLACEHOLDER", inference_duration_str
                        )

                    JobController.update_job_status(
                        job_id=job.id,
                        status="WARNING",
                        status_message=current_status_message,
                        runtime=inference_duration,
                    )
                    logger.debug(f"Job {finished_job.id} completed with warnings.")
                else:  # Job completed successfully
                    JobController.update_job_status(
                        job_id=job.id,
                        status="COMPLETED",
                        status_message=f"Job completed successfully in {inference_duration_str} seconds",
                        runtime=inference_duration,
                    )
                    logger.debug(f"Job {finished_job.id} processed successfully.")

            except Exception as e:
                # Check if this was a cancellation exception
                if self.is_job_cancelled(job.id):
                    logger.info(f"Job {job.id} execution was cancelled")
                else:
                    from app.controllers.job_controller import JobController

                    JobController.update_job_status(
                        job_id=job.id, status="FAILED", status_message=str(e)
                    )
                    logger.error(f"Error processing job {job.id}: {e}")
            finally:
                # Clean up cancelled job from tracking set
                if self.is_job_cancelled(job.id):
                    self.__cancelled_jobs.discard(job.id)
                self.__job_queue.task_done()

    def __add_pending_jobs(self):
        """
        Query the MongoDB jobs collection for jobs with 'PENDING' status and add them to the queue for processing.
        Also reset any 'PROCESSING' jobs to 'PENDING' status and delete their associated results.
        """
        try:
            # Import the database service to avoid circular import
            from app.config.registry import get_database_service

            jobs_collection = get_database_service().get_collection("jobs")

            # First, reset any processing jobs to pending and clean up their results
            self.__reset_processing_jobs()

            pending_jobs_cursor = jobs_collection.find({"status": "PENDING"})
            pending_jobs_count = 0

            for job_doc in pending_jobs_cursor:
                try:
                    # Import Job class here to avoid circular import
                    from app.models.job import Job

                    job = Job(**job_doc)

                    # Add job to the queue
                    if self.add_job(job):
                        pending_jobs_count += 1
                        logger.debug(f"Added pending job {job.id} to queue")
                    else:
                        logger.warning(f"Failed to add pending job {job.id} to queue")

                except Exception as e:
                    logger.error(
                        f"Error processing pending job document {job_doc.get('id', 'unknown')}: {e}"
                    )
                    continue

            if pending_jobs_count > 0:
                logger.debug(f"Added {pending_jobs_count} pending jobs to the queue")

        except Exception as e:
            logger.error(f"Error retrieving pending jobs from database: {e}")
            # Don't raise exception here as this shouldn't prevent queue initialization

    def __reset_processing_jobs(self):
        """
        Reset any jobs with 'PROCESSING' status to 'PENDING' and delete their associated result objects.
        This handles cases where jobs were interrupted during processing.
        """
        try:
            # Import the database service to avoid circular import
            from app.config.registry import get_database_service

            jobs_collection = get_database_service().get_collection("jobs")

            processing_jobs_cursor = jobs_collection.find({"status": "PROCESSING"})
            reset_jobs_count = 0

            for job_doc in processing_jobs_cursor:
                try:
                    # Import Job class here to avoid circular import
                    from app.models.job import Job

                    job = Job(**job_doc)

                    # Delete associated result if it exists
                    if job.result_id:
                        try:
                            from app.controllers.result_controller import (
                                ResultController,
                            )

                            ResultController.delete_result(job.result_id)
                            logger.debug(
                                f"Deleted result {job.result_id} for reset job {job.id}"
                            )
                        except Exception as e:
                            # Log warning but continue - result might not exist or already be deleted
                            logger.warning(
                                f"Could not delete result {job.result_id} for job {job.id}: {e}"
                            )

                    # Reset job status to PENDING
                    jobs_collection.update_one(
                        {"id": job_doc["id"]},
                        {"$set": {"status": "PENDING", "result_id": ""}},
                    )

                    reset_jobs_count += 1
                    logger.debug(f"Reset running job {job.id} to pending status")

                except Exception as e:
                    logger.error(
                        f"Error resetting running job document {job_doc.get('id', 'unknown')}: {e}"
                    )
                    continue

            if reset_jobs_count > 0:
                logger.debug(f"Reset {reset_jobs_count} running jobs to pending status")

        except Exception as e:
            logger.error(f"Error resetting running jobs: {e}")
            # Don't raise exception here as this shouldn't prevent queue initialization

    def add_job(self, job) -> bool:
        """
        Add a job to the queue.
        """
        try:
            self.__job_queue.put(job)
        except Exception as e:
            logger.error(f"Error adding job to queue: {e}")
            return False
        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job by marking it as cancelled.
        If the job is currently executing, it will be interrupted.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            bool: True if the job was successfully marked for cancellation
        """
        try:
            self.__cancelled_jobs.add(job_id)
            logger.info(f"Job {job_id} marked for cancellation")

            # Update job status in database
            from app.controllers.job_controller import JobController

            JobController.update_job_status(
                job_id=job_id,
                status="CANCELLED",
                status_message="Job was cancelled by user request",
            )

            return True
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    def is_job_cancelled(self, job_id: str) -> bool:
        """
        Check if a job has been cancelled.

        Args:
            job_id: The ID of the job to check

        Returns:
            bool: True if the job has been cancelled
        """
        return job_id in self.__cancelled_jobs

    def stop(self):
        """
        Signal all workers to stop and wait for them to finish.
        """
        self.__stop_event.set()
        for thread in self.__threads:
            thread.join()

        logger.info("JobQueue stopped successfully.")

    def format_run_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{round(seconds)}s"
        minutes = round(seconds // 60)
        remainingSeconds = round(seconds % 60)
        return f"{minutes}m {remainingSeconds}s"
