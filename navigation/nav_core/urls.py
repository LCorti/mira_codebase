"""
URL configuration for nav_core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from django.views.generic.base import RedirectView

from routes import api

api_urlpatterns = [
    path("api/status/", api.status, name="api_status"),
    path(
        "api/groups/update-group-name/",
        api.update_group_name,
        name="api_update_group_name",
    ),
    path(
        "api/projects/create_tests_and_methods/",
        api.create_tests_and_methods,
        name="api_create_tests_and_methods",
    ),
    path("api/models/download_size/", api.download_size, name="api_download_size"),
    path("api/models/config/", api.model_config, name="api_model_config"),
    path(
        "api/datasets/metadata/", api.get_datasets_metadata, name="api_dataset_metadata"
    ),
    path("api/datasets/config/", api.dataset_config, name="api_dataset_config"),
    path("api/datasets/show/", api.dataset_show, name="api_dataset_show"),
    path("api/datasets/metrics/", api.dataset_metrics, name="api_dataset_metrics"),
    path(
        "api/datasets/by-task/",
        api.get_datasets_by_task,
        name="api_get_datasets_by_task",
    ),
    path(
        "api/metrics/metadata/all/",
        api.get_metrics_metadata,
        name="api_metrics_metadata_all",
    ),
    path("api/create-jobs/", api.create_jobs, name="api_create_jobs"),
    path(
        "api/create-job-diagnostic/",
        api.create_job_diagnostic,
        name="create_job_diagnostic",
    ),
    path("api/duplicate-jobs/", api.duplicate_jobs, name="api_duplicate_jobs"),
    path("api/jobs/", api.get_jobs, name="api_get_jobs"),
    path("api/cancel-job/", api.cancel_job, name="api_cancel_job"),
    path(
        "api/tasks/available/", api.get_available_tasks, name="api_get_available_tasks"
    ),
    path(
        "api/tasks/from-dataset-readme/",
        api.get_tasks_from_dataset_readme,
        name="api_get_tasks_from_dataset_readme",
    ),
    # Tags endpoints
    path("api/tags/available/", api.get_available_tags, name="api_get_available_tags"),
    path("api/tags/job/", api.get_job_tags, name="api_get_job_tags"),
    path("api/tags/create/", api.create_job_tag, name="api_create_job_tag"),
    path(
        "api/tags/update/<str:tag_id>/", api.update_job_tag, name="api_update_job_tag"
    ),
    path(
        "api/tags/delete/<str:tag_id>/", api.delete_job_tag, name="api_delete_job_tag"
    ),
]

web_urlpatterns = [
    path("admin/", admin.site.urls),
    path("projects/", include("projects.urls")),
    path("", RedirectView.as_view(url="/projects/open", permanent=False)),
]

urlpatterns = api_urlpatterns + web_urlpatterns
