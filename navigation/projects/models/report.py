from typing import ClassVar

from bson import ObjectId
from django.db import models


def generate_object_id():
    """Generate a new ObjectId string for use as default primary key."""
    return str(ObjectId())


class Report(models.Model):
    id = models.CharField(
        primary_key=True, max_length=24, default=generate_object_id, editable=False
    )
    name = models.CharField(max_length=200, help_text="Report name")
    description = models.TextField(blank=True, help_text="Report description")

    # Navigation service specific fields (not in processing service)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering: ClassVar[list[str]] = ["-created_at"]
        verbose_name = "Report"
        verbose_name_plural = "Reports"

    def __str__(self):
        return self.name
