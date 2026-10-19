from typing import ClassVar

from bson import ObjectId
from django.db import models


def generate_object_id():
    """Generate a new ObjectId string for use as default primary key."""
    return str(ObjectId())


def get_default_test_parameters():
    """Default test parameters factory function."""
    return {
        "task": "",
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


def get_default_metadata():
    """Default metadata factory function."""
    return {
        "description": "",  # Description of the diagnostic test
        "paper_name": "",  # Name of the paper
        "paper_ref": "",  # Citation for the paper
        "paper_source": "",  # URL to the paper
        "tags": [],  # dataset / automatic
        "diagnostic_tag": "",  # should-Know / really-Knows
    }


class Test(models.Model):
    id = models.CharField(
        primary_key=True, max_length=24, default=generate_object_id, editable=False
    )
    name = models.CharField(max_length=200, help_text="Test name")
    description = models.TextField(blank=True, help_text="Test description")
    diagnostic = models.BooleanField(default=False)
    diagnostic_only = models.BooleanField(default=False)
    group = models.CharField(
        null=True, blank=True, help_text="Group number for grouping tests"
    )
    metadata = models.JSONField(
        default=get_default_metadata, help_text="Additional metadata for the test"
    )

    parameters = models.JSONField(help_text="Test parameters", default=dict)

    datasets = models.JSONField(
        help_text="List of datasets used in this project", default=list
    )
    metrics = models.JSONField(
        help_text="Dictionary of metrics used in this project", default=dict
    )

    project = models.ForeignKey(
        "projects.Project",
        on_delete=models.CASCADE,
        related_name="tests",
        help_text="Project this test belongs to",
        default=None,
    )

    # Navigation service specific fields (not in processing service)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Sync fields to track processing service state
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
        verbose_name = "Test"
        verbose_name_plural = "Tests"

    def __str__(self):
        return self.name

    def get_task_type(self):
        """Extract task type from parameters."""
        return self.parameters.get("task_type", "")

    @property
    def is_synced(self):
        """Check if test is synced with processing service."""
        return self.sync_status == "synced"

    def get_default_parameters(self):
        """Default test parameters factory function."""
        if self.parameters and isinstance(self.parameters, dict):
            return self.parameters

        if self.project and hasattr(self.project, "parameters"):
            return self.project.parameters

        return get_default_test_parameters()

    def to_processing_service_format(self):
        """Convert to format expected by processing service."""
        return {
            "test": {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "project_id": self.project_id,
                "method_ids": [],
            },
            "datasets": self.datasets,
            "metrics": self.metrics,
        }
