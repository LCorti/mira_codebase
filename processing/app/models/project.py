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


class Project(BaseMongoModel):
    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=500)
    model_name: str = Field(max_length=200)
    parameters: dict = Field(
        default={
            "task_type": "",
            "task_parameters": {},
            "model": {"from_pretrained": {}, "inference": {}},
            "tokenizer": {"from_pretrained": {}, "inference": {}},
            "dataset": {
                "DATASET_NAME": {
                    "split": "train",
                    "input_column": "input",
                    "reference_column": "reference",
                    "num_samples": 15,
                }
            },
            "metrics": {"DATASET_NAME": {"METRIC_NAME": {}}},
        }
    )

    test_ids: list[str] = Field(default=[])

    def tests(self):
        """
        Retrieve the tests associated with this project.

        :return: List of Test objects associated with this project.
        """
        if not self.test_ids:
            logger.warning(f"Project {self.id} does not have any test_ids.")
            return []

        from app.models.test import Test

        try:
            tests = _get_tests_collection().find(
                {"id": {"$in": [test_id for test_id in self.test_ids]}}
            )
            logger.debug(
                f"Retrieved {len(self.test_ids)} tests for project with ID {self.id} from database."
            )

            if not tests:
                logger.warning(f"No tests found for project with ID {self.id}.")
                return []

            return [Test(**test) for test in tests]
        except Exception as e:
            logger.error(f"Error retrieving tests for project with ID {self.id}: {e}")
            return []

    def to_dict(self, include: set[str] | None = None):
        """
        Convert model to dictionary, lazily including only requested relations.

        Args:
            include: Set of relations to include (e.g., {'tests', 'tests.methods'})
        """
        include = include or set()
        data = self.model_dump()

        if "tests" in include:
            # Extract nested includes for tests (e.g., 'tests.results' -> 'results')
            test_includes = {
                ".".join(rel.split(".")[1:])
                for rel in include
                if rel.startswith("tests.") and "." in rel
            }

            data["tests"] = [
                test.to_dict(include=test_includes) for test in self.tests()
            ]
            data.pop("test_ids", None)

        return data


class ProjectUpdate(BaseMongoModel):
    name: str | None = None
    description: str | None = None
    model_name: str | None = None
    parameters: dict | None = None
    test_ids: list[str] | None = None
