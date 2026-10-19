from pydantic import Field

from app.config.registry import get_database_service
from app.models.base_model import BaseMongoModel
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="models.log")


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


def _get_logits_collection():
    """
    Returns the logits collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("logits")


class Result(BaseMongoModel):
    diagnostics: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    results: list[dict] = Field(default_factory=list[dict])
    parameters: dict = Field(
        default={
            "task_type": "",
            "task_parameters": {},
            "model": {"from_pretrained": {}, "inference": {}},
            "tokenizer": {"from_pretrained": {}, "inference": {}},
            "dataset": {
                "split": "train",
                "input_column": "input",
                "reference_column": "reference",
                "num_samples": 15,
            },
            "metrics": {"METRIC_NAME": {}},
        }
    )

    method_id: str = Field(default="")
    job_id: str = Field(default="")

    def method(self):
        """
        Retrieve the method associated with this result.

        :return: Method object associated with this result.
        """
        if not self.method_id:
            logger.warning(f"Result {self.id} does not have a method_id.")
            return None

        from app.models.method import Method

        try:
            method = _get_methods_collection().find_one({"id": self.method_id})
            logger.debug(f"Retrieved method with ID {self.method_id} from database.")

            if not method:
                raise ValueError(f"No method found for result with ID {self.id}.")

            return Method(**method)
        except Exception as e:
            logger.error(f"Error retrieving method for result with ID {self.id}: {e}")
            raise RuntimeError(f"Failed to retrieve method: {e}")

    def job(self):
        """
        Retrieve the job associated with this result.

        :return: Job object associated with this result.
        """
        if not self.job_id:
            logger.warning(f"Result {self.id} does not have a job_id.")
            return None

        from app.models.job import Job

        try:
            job = _get_jobs_collection().find_one({"id": self.job_id})

            if not job:
                raise ValueError(f"No job found for result with ID {self.id}.")

            return Job(**job)
        except Exception as e:
            logger.error(f"Error retrieving job for result with ID {self.id}: {e}")
            raise RuntimeError(f"Failed to retrieve job: {e}")

    def to_dict(self) -> dict:
        """
        Convert model to dictionary, lazily including only requested relations.

        Args:
            None
        """

        return self.model_dump()


class ResultUpdate(BaseMongoModel):
    diagnostics: dict | None = None
    summary: dict | None = None
    results: list[dict] | None = None
    method_id: str | None = None
    job_id: str | None = None
