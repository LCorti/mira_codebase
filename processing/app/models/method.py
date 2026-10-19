from pydantic import Field

from app.config.registry import get_database_service
from app.models.base_model import BaseMongoModel
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="models.log")


def _get_tests_collection():
    """
    Returns the tests collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("tests")


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


class Method(BaseMongoModel):
    name: str = Field(max_length=200)
    description: str = Field(max_length=500)
    dataset_name: str = Field(max_length=200)
    diagnostics: list[str] = Field(default_factory=list)
    diagnostic_only: bool = Field(default=False)
    metrics: list[str] = Field(default_factory=list)
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
            "diagnostics": {"DIAGNOSTIC_NAME": {}},
        }
    )

    test_id: str = Field(default="")
    job_ids: list[str] = Field(default=[])
    result_ids: list[str] = Field(default=[])

    def test(self):
        """
        Retrieve the test associated with this method.

        :return: Test object associated with this method.
        """
        if not self.test_id:
            logger.warning(f"Method {self.id} does not have a test_id.")
            return None

        from app.models.test import Test

        try:
            test = _get_tests_collection().find_one({"id": self.test_id})
            logger.debug(f"Retrieved test with ID {self.test_id} from database.")

            if not test:
                logger.warning(f"Test with ID {self.test_id} not found.")
                return {}

            return Test(**test)
        except Exception as e:
            logger.error(f"Error retrieving method with ID {self.test_id}: {e}")
            return {}

    def jobs(self):
        """
        Retrieve the jobs associated with this method.

        :return: List of job objects associated with this method.
        """
        if not self.job_ids:
            return []

        from app.models.job import Job

        try:
            jobs = _get_jobs_collection().find(
                {"id": {"$in": [job_id for job_id in self.job_ids]}}
            )

            if not jobs:
                logger.warning(f"No jobs found for method with ID {self.id}.")
                return []

            return [Job(**job) for job in jobs]
        except Exception as e:
            logger.error(f"Error retrieving jobs for method with ID {self.id}: {e}")
            return []

    def results(self):
        """
        Retrieve the results associated with this method.

        :return: List of result objects associated with this method.
        """
        if not self.result_ids:
            logger.warning(f"Method {self.id} does not have any result_ids.")
            return []

        from app.models.result import Result

        try:
            results = _get_results_collection().find(
                {"id": {"$in": [result_id for result_id in self.result_ids]}}
            )
            logger.debug(
                f"Retrieved {len(self.result_ids)} results for method with ID {self.id} from database."
            )

            if not results:
                logger.warning(f"No results found for method with ID {self.id}.")
                return []

            return [Result(**result) for result in results]
        except Exception as e:
            logger.error(f"Error retrieving results for method with ID {self.id}: {e}")
            return []

    def to_dict(self, include: set[str] | None = None):
        """
        Convert model to dictionary, lazily including only requested relations.

        Args:
            include: Set of relations to include (e.g., {'jobs', 'results', 'jobs.result'})
        """
        include = include or set()
        data = self.model_dump()

        if "jobs" in include:
            # Extract nested includes for jobs (e.g., 'jobs.result' -> 'result')
            job_includes = {
                ".".join(rel.split(".")[1:])
                for rel in include
                if rel.startswith("jobs.") and "." in rel
            }

            data["jobs"] = [job.to_dict(include=job_includes) for job in self.jobs()]
            data.pop("job_ids", None)

        if "results" in include:
            data["results"] = [result.to_dict() for result in self.results()]
            data.pop("result_ids", None)

        return data


class MethodUpdate(BaseMongoModel):
    name: str | None = None
    description: str | None = None
    dataset_name: str | None = None
    diagnostics: list[str] | None = None
    metrics: list[str] | None = None
    parameters: dict | None = None
    test_id: str | None = None
    job_ids: list[str] | None = None
    result_ids: list[str] | None = None
