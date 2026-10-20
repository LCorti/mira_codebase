import torch
from pydantic import Field

from app.models.base_model import BaseMongoModel
from app.services.logging_service import LoggingService
from app.services.tensor_parser import string_to_tensor

logger = LoggingService.setup_module_logger(__name__, log_filename="models.log")


class Logit(BaseMongoModel):
    logit_str: str = Field(default="")

    def to_dict(self) -> dict:
        """
        Convert model to dictionary, lazily including only requested relations.

        Args:
            None
        """
        return self.model_dump()

    def to_tensor(self) -> torch.Tensor | None:
        """
        Convert the logit string to a tensor.

        Returns:
            The logit as a tensor.
        """
        return string_to_tensor(self.logit_str)
