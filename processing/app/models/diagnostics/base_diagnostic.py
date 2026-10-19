from abc import ABC

from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")


class Diagnostic(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def _check_job_cancellation(self, job_id: str, job_queue):
        """
        Helper function to check if a job has been cancelled.

        Args:
            job_id: The ID of the job to check
            job_queue: The job queue instance to check cancellation status

        Raises:
            Exception: If the job has been cancelled
        """
        if job_queue and job_id and job_queue.is_job_cancelled(job_id):
            from app.models.job import JobCancelledException

            logger.info(f"{job_id}: Job execution cancelled during inference/analysis")
            raise JobCancelledException(f"Job {job_id} was cancelled during execution")

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs
    ) -> dict | list[dict]:
        """
        Run the diagnostic computation on the provided predictions and references.

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :type results: list[dict]
        :param parameters: Additional parameters for the diagnostic computation.
        :type parameters: dict
        :param kwargs: Additional keyword arguments for the metric computation.
        :type kwargs: Any

        :return: Dictionary containing the diagnostic results.
        """

        raise NotImplementedError("Subclasses must implement the compute method.")

    def __str__(self):
        return f"{self.name} Diagnostic: {self.description}"

    def __repr__(self):
        return f"Diagnostic(name={self.name}, description={self.description})"
