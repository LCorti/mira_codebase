from pydantic import Field
from bson import ObjectId

from app.config.registry import get_database_service

from app.services.logging_service import LoggingService

from app.models.base_model import BaseMongoModel

logger = LoggingService.setup_module_logger(__name__, log_filename="models.log")

def _get_projects_collection():
    """
    Returns the projects collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("projects")

def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")
    
class Test(BaseMongoModel):
    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=500)
    default: bool = Field(default=False)
    diagnostic_only: bool = Field(default=False)
    group: str = Field(default="")
    parameters: dict = Field(default={
        "task_type": "",
        "task_parameters": {},
        "model": {
            "from_pretrained": {},
            "inference": {}
        },
        "tokenizer": {
            "from_pretrained": {},
            "inference": {}
        },
        "dataset": {
            "DATASET_NAME": {
                "split": "train",
                "input_column": "input",
                "reference_column": "reference",
                "num_samples": 15
            }
        },
        "metrics": {
            "DATASET_NAME": {
                "METRIC_NAME": {}
            }
        }
    })
    
    project_id: str = Field(default="")
    method_ids: list[str] = Field(default=[])
    metadata: dict = Field(default={
        "description": "",      # Description of the diagnostic test
        "paper_name": "",       # Name of the paper
        "paper_ref": "",        # Citation for the paper
        "paper_source": "",     # URL to the paper
        "tags": [],             # dataset / automatic
        "diagnostic_tag": ""    # should-Know / really-Knows
    })

    def project(self):
        """
        Retrieve the project associated with this test.
        
        :return: Project object associated with this test.
        """
        if not self.project_id:
            logger.warning(f"Test {self.id} does not have a project_id.")
            return {}

        from app.models.project import Project
        try:
            project = _get_projects_collection().find_one({"id": self.project_id})
            logger.debug(f"Retrieved project with ID {self.project_id} from database.")
            
            if not project:
                logger.warning(f"Project with ID {self.project_id} not found.")
                return {}

            return Project(**project)
        except Exception as e:
            logger.error(f"Error retrieving project for test with ID {self.id}: {e}")
            return {}

    def methods(self):
        """
        Retrieve the methods associated with this test.
        
        :return: List of method objects associated with this test.
        """
        if not self.method_ids:
            logger.warning(f"Test {self.id} does not have any method_ids.")
            return []

        from app.models.method import Method
        try:
            methods = _get_methods_collection().find({"id": {"$in": [method_id for method_id in self.method_ids]}})
            logger.debug(f"Retrieved {len(self.method_ids)} methods for test with ID {self.id} from database.")
            
            if not methods:
                logger.warning(f"No methods found for test with ID {self.id}.")
                return []

            return [Method(**method) for method in methods]
        except Exception as e:
            logger.error(f"Error retrieving methods for test with ID {self.id}: {e}")
            return []

    def to_dict(self, include: set[str] | None = None):
        """
        Convert model to dictionary, lazily including only requested relations.
        
        Args:
            include: Set of relations to include (e.g., {'methods', 'methods.jobs'})
        """
        include = include or set()
        data = self.model_dump()
        
        if "methods" in include:
            # Extract nested includes for methods (e.g., 'methods.jobs' -> 'jobs')
            nested_includes = {
                ".".join(rel.split(".")[1:]) 
                for rel in include
                if rel.startswith("methods.") and "." in rel
            }
            
            data["methods"] = [
                method.to_dict(include=nested_includes) 
                for method in self.methods()
            ]
            # Remove ID list since we're including full objects
            data.pop("method_ids", None)
        
        return data
    
class TestUpdate(BaseMongoModel):
    name: str | None = None
    description: str | None = None
    parameters: dict | None = None
    group: int | None = None
    project_id: str | None = None
    method_ids: list[str] | None = None
