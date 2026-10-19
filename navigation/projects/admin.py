from typing import ClassVar

from django.contrib import admin

from projects.models import Method, MethodActivity, Project, Test


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    """
    Admin interface for Project model.
    """

    list_display: ClassVar[list[str]] = [
        "name",
        "model_name",
        "sync_status",
        "auto_run",
        "created_at",
        "updated_at",
    ]
    list_filter: ClassVar[list[str]] = ["sync_status", "created_at"]
    search_fields: ClassVar[list[str]] = [
        "name",
        "description",
        "model_name",
        "datasets",
        "metrics",
    ]
    readonly_fields: ClassVar[list[str]] = [
        "id",
        "created_at",
        "updated_at",
        "last_synced",
    ]

    fieldsets = (
        ("Basic Information", {"fields": ("name", "description", "auto_run")}),
        (
            "Project Configuration",
            {"fields": ("model_name", "datasets", "metrics", "parameters", "test_ids")},
        ),
        (
            "Sync Information",
            {"fields": ("sync_status", "last_synced"), "classes": ("collapse",)},
        ),
        (
            "Metadata",
            {"fields": ("id", "created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(Test)
class TestAdmin(admin.ModelAdmin):
    """
    Admin interface for Test model.
    """

    list_display: ClassVar[list[str]] = [
        "name",
        "sync_status",
        "created_at",
        "updated_at",
    ]
    list_filter: ClassVar[list[str]] = ["sync_status", "created_at"]
    search_fields: ClassVar[list[str]] = ["name", "description", "datasets", "metrics"]
    readonly_fields: ClassVar[list[str]] = [
        "id",
        "created_at",
        "updated_at",
        "last_synced",
    ]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("name", "description", "diagnostic", "diagnostic_only")},
        ),
        (
            "Test Configuration",
            {"fields": ("project", "datasets", "metrics", "parameters", "metadata")},
        ),
        (
            "Sync Information",
            {"fields": ("sync_status", "last_synced"), "classes": ("collapse",)},
        ),
        (
            "Metadata",
            {"fields": ("id", "created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(Method)
class MethodAdmin(admin.ModelAdmin):
    """
    Admin interface for Method model.
    """

    list_display: ClassVar[list[str]] = [
        "name",
        "test",
        "sync_status",
        "created_at",
        "updated_at",
    ]
    list_filter: ClassVar[list[str]] = ["sync_status", "created_at"]
    search_fields: ClassVar[list[str]] = ["name", "description"]
    readonly_fields: ClassVar[list[str]] = [
        "id",
        "created_at",
        "updated_at",
        "last_synced",
    ]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("name", "description", "test", "diagnostic_only")},
        ),
        ("Method Configuration", {"fields": ("dataset", "metrics", "parameters")}),
        (
            "Sync Information",
            {"fields": ("sync_status", "last_synced"), "classes": ("collapse",)},
        ),
        (
            "Metadata",
            {"fields": ("id", "created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MethodActivity)
class MethodActivityAdmin(admin.ModelAdmin):
    """
    Admin interface for MethodActivity model.
    """

    list_display: ClassVar[list[str]] = [
        "method",
        "activity_type",
        "timestamp",
        "get_change_summary",
    ]
    list_filter: ClassVar[list[str]] = ["activity_type", "timestamp"]
    search_fields: ClassVar[list[str]] = ["method__name", "description"]
    readonly_fields: ClassVar[list[str]] = [
        "method",
        "activity_type",
        "timestamp",
        "changes",
        "description",
    ]

    def has_add_permission(self, request):
        """Disable manual creation of activities."""
        return False

    def has_change_permission(self, request, obj=None):
        """Make activities read-only."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Allow deletion for cleanup purposes."""
        return True

    fieldsets = (
        ("Activity Information", {"fields": ("method", "activity_type", "timestamp")}),
        ("Changes", {"fields": ("changes", "description")}),
    )
