from typing import ClassVar

from bson import ObjectId
from django.db import models


def generate_object_id():
    """Generate a new ObjectId string for use as default primary key."""
    return str(ObjectId())


def get_default_project_parameters():
    """Default project parameters factory function."""
    return {
        "task_type": "",
        "task_parameters": {},
        "model": {"from_pretrained": {}, "inference": {}},
        "tokenizer": {"from_pretrained": {}, "inference": {}},
        "dataset": {
            "DATASET_NAME": {
                "split": "train",  # train, validation, test
                "input_column": "input",
                "reference_column": "reference",
                "num_samples": 15,
            }
        },
        "metrics": {
            "DATASET_NAME": {}
        },  # E.g, "DATASET_NAME": {"bertscore": {lang: "en", "batch_size": 16}}
    }


class Project(models.Model):
    id = models.CharField(
        primary_key=True, max_length=24, default=generate_object_id, editable=False
    )
    name = models.CharField(max_length=200, help_text="Project name")
    description = models.TextField(blank=True, help_text="Project description")
    model_name = models.CharField(
        max_length=200, default="", help_text="Name of the model used in this project"
    )

    datasets = models.JSONField(
        help_text="List of datasets used in this project", default=list
    )
    metrics = models.JSONField(
        help_text="Dictionary of metrics used in this project", default=dict
    )

    parameters = models.JSONField(
        help_text="Project parameters", default=get_default_project_parameters
    )
    test_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of test IDs associated with this project",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Field for request
    auto_run = models.BooleanField(
        default=False, help_text="Automatically run this project when created"
    )

    last_synced = models.DateTimeField(
        null=True, blank=True, help_text="Last sync with processing service"
    )
    sync_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("synced", "Synced"),
            ("error", "Error"),
        ],
        default="pending",
    )

    class Meta:
        ordering: ClassVar[list[str]] = ["-created_at"]
        verbose_name = "Project"
        verbose_name_plural = "Projects"

    def __str__(self):
        return self.name

    def get_task_type(self):
        """Extract task type from parameters."""
        return self.parameters.get("task_type", "")

    @property
    def is_synced(self):
        """Check if project is synced with processing service."""
        return self.sync_status == "synced"

    def to_processing_service_format(self):
        """Convert to format expected by processing service."""
        data = {
            "project": {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "model_name": self.model_name,
                "parameters": self.parameters,
                "test_ids": self.test_ids,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
            "datasets": self.datasets,
            "metrics": self.metrics,
        }
        return data
