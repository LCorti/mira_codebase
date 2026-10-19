from pydantic import Field
from typing import Optional

from app.services.logging_service import LoggingService

from app.models.base_model import BaseMongoModel

logger = LoggingService.setup_module_logger(__name__, log_filename="models.log")

def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    from app.config.registry import get_database_service
    db_service = get_database_service()
    return db_service.get_collection("jobs")

class Tag(BaseMongoModel):
    """Represents a tag that can be applied to LLM inference jobs"""
    # Standardized tag metadata (from configuration)
    name: str = Field(description="Tag name")
    description: str = Field(description="Tag description")
    paper_reference: Optional[str] = Field(default=None, description="Reference to relevant paper")
    example: Optional[str] = Field(default=None, description="Example of this tag")
    category: Optional[str] = Field(default=None, description="Tag category")
    
    # Instance-specific fields (when applied to a job)
    comment: str = Field(default="", description="User-provided comment for this tag instance")
    filters: list = Field(default=["SK", "RK"], description="Defaults to Should-Know and Really-Know")
    
    job_id: str = Field(default="", description="ID of the job this tag is associated with")
    
    def job(self):
        if not self.job_id: return None
        
        from app.models.job import Job
        try:
            job = _get_jobs_collection().find_one({"id": self.job_id})
            logger.debug(f"Retrieved job with ID {self.job_id} from database.")

            if not job: raise ValueError(f"Job with ID {self.job_id} not found.")

            return Job(**job)
        except Exception as e:
            logger.error(f"Error retrieving job with ID {self.job_id}: {e}")
            raise RuntimeError(f"Failed to retrieve job: {e}")

    def to_dict(self) -> dict:
        """
        Convert model to dictionary, lazily including only requested relations.
        
        Args:
            None
        """
        return self.model_dump()