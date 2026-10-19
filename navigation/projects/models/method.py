from typing import ClassVar

from bson import ObjectId
from django.db import models


def generate_object_id():
    """Generate a new ObjectId string for use as default primary key."""
    return str(ObjectId())


def get_default_method_parameters():
    """Default method parameters factory function."""
    return {
        "dataset": {
            "split": "train",  # train, validation, test
            "input_column": "input",
            "reference_column": "reference",
            "num_samples": 15,
        },
        "metrics": {},  # E.g, "bertscore": {lang: "en", "batch_size": 16}
    }


class Method(models.Model):
    id = models.CharField(
        primary_key=True, max_length=24, default=generate_object_id, editable=False
    )
    name = models.CharField(max_length=200, help_text="Method name")
    description = models.TextField(blank=True, help_text="Method description")
    diagnostic_only = models.BooleanField(default=False)

    dataset = models.CharField(max_length=200, blank=True, help_text="Dataset name")

    metrics = models.JSONField(
        default=list,
        blank=True,
        help_text="Method metrics",
    )

    parameters = models.JSONField(help_text="Method parameters", default=dict)
    test = models.ForeignKey(
        "projects.Test",
        on_delete=models.CASCADE,
        related_name="methods",
        help_text="Test this method belongs to",
        default=None,
    )

    job_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of job IDs associated with this method",
    )
    result_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of result IDs associated with this method",
    )

    # Navigation service specific fields (not in processing service)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    last_synced = models.DateTimeField(
        null=True, blank=True, help_text="Last sync with processing service"
    )
    sync_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("synced", "Synced"),
            ("error", "Error"),
            ("local_only", "Local Only"),
        ],
        default="local_only",
    )

    class Meta:
        ordering: ClassVar[list[str]] = ["-created_at"]
        verbose_name = "Method"
        verbose_name_plural = "Methods"

    def __str__(self):
        return f"{self.name}"

    def get_task_type(self):
        """Extract task type from parameters."""
        import json

        # Handle case where parameters might be a JSON string instead of dict
        if isinstance(self.parameters, str):
            try:
                params = json.loads(self.parameters)
            except (json.JSONDecodeError, TypeError):
                return ""
        else:
            params = self.parameters

        if isinstance(params, dict):
            return params.get("task_type", "")
        return ""

    @property
    def is_synced(self):
        """Check if method is synced with processing service."""
        return self.sync_status == "synced"

    def get_default_parameters(self):
        """Default method parameters factory function."""
        if self.parameters and isinstance(self.parameters, dict):
            return self.parameters

        if self.test and hasattr(self.test, "parameters"):
            test_parameters = self.test.parameters.copy()
            test_parameters["dataset"] = test_parameters.get("dataset", {}).get(
                self.dataset, {}
            )
            test_parameters["metrics"] = test_parameters.get("metrics", {}).get(
                self.dataset, {}
            )

            return test_parameters
        return get_default_method_parameters()

    def track_activity(self, activity_type, changes=None, description=None):
        """Helper method to track activities."""
        from .method_activity import MethodActivity

        _ = MethodActivity.objects.create(
            method=self,
            activity_type=activity_type,
            changes=changes or {},
            description=description,
        )

    def save(self, *args, **kwargs):
        """Override save to track changes."""
        track_changes = kwargs.pop("track_changes", True)  # Allow disabling tracking

        # Check if this is an update by trying to get the existing instance
        is_update = False
        old_instance = None

        if self.pk and track_changes:
            try:
                old_instance = Method.objects.get(pk=self.pk)
                is_update = True
            except Method.DoesNotExist:
                is_update = False

        if is_update and track_changes:  # Existing object
            changes = self._detect_changes(old_instance)

            super().save(*args, **kwargs)

            # Track specific changes
            for change_type, change_data in changes.items():
                self.track_activity(change_type, change_data)
        else:  # New object
            is_new = not is_update
            super().save(*args, **kwargs)
            if is_new and track_changes:
                self.track_activity("created")

    def _detect_changes(self, old_instance):
        """Detect what changed between old and new instance."""
        changes = {}

        # Name change
        if old_instance.name != self.name:
            changes["name_changed"] = {
                "old_value": old_instance.name,
                "new_value": self.name,
            }

        # Description change
        if old_instance.description != self.description:
            changes["description_changed"] = {
                "old_value": old_instance.description,
                "new_value": self.description,
            }

        # Dataset change
        if old_instance.dataset != self.dataset:
            changes["dataset_changed"] = {
                "old_value": old_instance.dataset,
                "new_value": self.dataset,
            }

        # Metrics changes
        old_metrics = set(old_instance.metrics) if old_instance.metrics else set()
        new_metrics = set(self.metrics) if self.metrics else set()

        added_metrics = new_metrics - old_metrics
        removed_metrics = old_metrics - new_metrics

        if added_metrics:
            changes["metrics_added"] = {"added": list(added_metrics)}
        if removed_metrics:
            changes["metrics_removed"] = {"removed": list(removed_metrics)}

        # Parameters changes
        if old_instance.parameters != self.parameters:
            changes["parameters_changed"] = {
                "old_value": old_instance.parameters,
                "new_value": self.parameters,
            }

        return changes

    def to_processing_service_format(self):
        """Convert to format expected by processing service."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
            "parameters": self.parameters,
        }
