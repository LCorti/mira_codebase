from typing import ClassVar

from django.db import models


class MethodActivity(models.Model):
    ACTIVITY_TYPES: ClassVar[list[tuple[str, str]]] = [
        ("created", "Created"),
        ("updated", "Updated"),
        ("metrics_added", "Metrics Added"),
        ("metrics_removed", "Metrics Removed"),
        ("parameters_changed", "Parameters Changed"),
        ("name_changed", "Name Changed"),
        ("description_changed", "Description Changed"),
        ("dataset_changed", "Dataset Changed"),
        ("job_ids_changed", "Job IDs Changed"),
        ("result_ids_changed", "Result IDs Changed"),
        ("sync_status_changed", "Sync Status Changed"),
    ]

    method = models.ForeignKey(
        "projects.Method", on_delete=models.CASCADE, related_name="activities"
    )
    activity_type = models.CharField(max_length=25, choices=ACTIVITY_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Store only the changes, not full snapshots
    changes = models.JSONField(
        default=dict, help_text="Specific changes made (old_value, new_value)"
    )

    description = models.TextField(
        blank=True, null=True, help_text="Human-readable description of the change"
    )

    class Meta:
        ordering: ClassVar[list[str]] = ["-timestamp"]
        verbose_name = "Method Activity"
        verbose_name_plural = "Method Activities"

    def __str__(self):
        return f"{self.method.name} - {self.get_activity_type_display()} at {self.timestamp}"

    def get_change_summary(self):
        """Generate human-readable change summary."""
        if self.activity_type == "metrics_added":
            added = self.changes.get("added", [])
            return f"Added metrics: {', '.join(map(str, added))}"
        elif self.activity_type == "metrics_removed":
            removed = self.changes.get("removed", [])
            return f"Removed metrics: {', '.join(map(str, removed))}"
        elif self.activity_type == "name_changed":
            old_name = self.changes.get("old_value", "")
            new_name = self.changes.get("new_value", "")
            return f"Changed name from '{old_name}' to '{new_name}'"
        elif self.activity_type == "description_changed":
            return "Updated description"
        elif self.activity_type == "dataset_changed":
            old_dataset = self.changes.get("old_value", "")
            new_dataset = self.changes.get("new_value", "")
            return f"Changed dataset from '{old_dataset}' to '{new_dataset}'"
        elif self.activity_type == "parameters_changed":
            return "Updated parameters"
        elif self.activity_type == "created":
            return "Method created"

        return self.description or f"{self.get_activity_type_display()}"

    def get_activity_type_display(self):
        return dict(self.ACTIVITY_TYPES).get(self.activity_type, "Unknown")
