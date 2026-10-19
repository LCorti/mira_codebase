from django.urls import path

from projects.views import method

urlpatterns = [
    # Method configuration routes
    path(
        "<str:project_id>/methods/<str:method_id>/update-parameters/",
        method.update_parameters,
        name="method_update_parameters",
    ),  # /projects/<project_id>/method/<method_id>/update-parameters/
    # Method CRUD routes (for future expansion)
    path(
        "<str:project_id>/methods/<str:method_id>/edit/",
        method.edit,
        name="method_edit",
    ),
    # Generate reports and comparisons
    path(
        "<str:project_id>/methods/<str:method_id>/compare/",
        method.compare,
        name="method_compare",
    ),
    path(
        "<str:project_id>/methods/<str:method_id>/generate-report/",
        method.generate_report,
        name="projects_method_generate_report",
    ),
    path(
        "<str:project_id>/methods/<str:method_id>/generate-pdf-report/",
        method.generate_pdf_report,
        name="projects_method_generate_pdf_report",
    ),
    # Export data
    path(
        "<str:project_id>/methods/<str:method_id>/export-results/",
        method.export_results,
        name="projects_method_export_results",
    ),
    # Report viewing and management routes
    path(
        "<str:project_id>/reports/<str:report_id>/view/",
        method.view_report,
        name="project_view_report",
    ),
    path(
        "<str:project_id>/reports/<str:report_id>/download/",
        method.download_report,
        name="project_download_report",
    ),
    path(
        "<str:project_id>/reports/<str:report_id>/delete/",
        method.delete_report,
        name="project_delete_report",
    ),
]
