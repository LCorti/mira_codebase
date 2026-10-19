import json
import os
from datetime import datetime

import requests
from django.conf import settings
from django.contrib import messages
from django.http import QueryDict
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from projects.forms import ProjectForm
from projects.models import MethodActivity, Project
from projects.utils.report_utils import get_reports_for_project
from routes.api import get_available_tasks, get_datasets_by_task, get_metrics_metadata

PROCESSING_SERVICE_URL = settings.PROCESSING_SERVICE_URL
PROCESSING_SERVICE_TIMEOUT = settings.PROCESSING_SERVICE_TIMEOUT


def open(request):
    """
    View to display the open project page.
    """
    projects = Project.objects.all()

    context = {
        "projects": projects,
        "total_projects": projects.count(),
    }

    return render(request, "open_project.html", context)


def show(request, project_id, section=None, test_id=None, method_id=None):
    """
    View to list all projects with pagination.
    """
    project = get_object_or_404(Project, id=project_id)

    # Valid sidebar sections
    valid_sections = ["project", "search", "diagnosis", "reports"]
    active_section = section if section in valid_sections else None

    # Make request to get diagnostic tests
    diagnostic_tests = []
    try:
        if PROCESSING_SERVICE_URL:
            response = requests.get(
                f"{PROCESSING_SERVICE_URL}/web/metadata/tests",
                timeout=3,  # Shorter timeout to prevent blocking
            )
            if response.status_code == 200:
                diagnostic_tests = response.json()
            else:
                # Service responded but with an error
                messages.warning(
                    request,
                    "Processing service is experiencing issues. Data list may be incomplete.",
                )
    except requests.exceptions.Timeout:
        messages.warning(
            request,
            "Processing service is not responding. You can still create projects, but data selection may be limited.",
        )
    except requests.exceptions.ConnectionError:
        messages.warning(
            request,
            "Cannot connect to processing service. Data list will be loaded dynamically.",
        )
    except requests.RequestException as e:
        messages.warning(
            request, f"Error connecting to processing service: {str(e)[:100]}"
        )

    # Make request to get tests and methods for project from processing service
    tests = []
    try:
        url = f"{PROCESSING_SERVICE_URL}/projects/{project.id}?include=tests&include=tests.methods&include=tests.methods.jobs&include=tests.methods.jobs.result&include=tests.methods.jobs.log&include=tests.methods.jobs.tags"
        response = requests.get(
            url,
            timeout=3,  # Shorter timeout to prevent blocking
        )

        if response.status_code == 200:
            tests = response.json().get("tests", [])
        else:
            messages.warning(
                request,
                "Processing service is experiencing issues. Test list may be incomplete.",
            )
    except requests.exceptions.Timeout:
        messages.warning(
            request,
            "Processing service is not responding. You can still view the project, but test data may be limited.",
        )
    except requests.exceptions.ConnectionError:
        messages.warning(
            request,
            "Cannot connect to processing service. Test data will be loaded dynamically.",
        )
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        messages.warning(
            request, f"Error connecting to processing service: {str(e)[:100]}"
        )
    # print(f"Tests: {tests}")

    mock_request = type("MockRequest", (), {"GET": QueryDict()})()
    metrics_response = get_metrics_metadata(mock_request)

    metrics = {}
    if hasattr(metrics_response, "content"):
        response_data = json.loads(metrics_response.content)
        metrics = response_data.get("metrics", {})
    # print(f"Metrics: {metrics}")

    task_type = project.get_task_type()
    mock_request = type(
        "MockRequest", (), {"GET": QueryDict(f"task_type={task_type}")}
    )()
    available_datasets = get_datasets_by_task(mock_request)

    datasets_dict = {}
    if hasattr(available_datasets, "content"):
        response_data = json.loads(available_datasets.content)
        datasets_dict = response_data.get("datasets", {})
    # print(f"Datasets: {datasets_dict}")

    # Enrich tests data with activity information
    enriched_tests = enrich_tests_with_activities(tests)
    # print(f"Enriched Tests: {enriched_tests}")

    # Group tests by their 'group' attribute
    grouped_tests = group_tests_by_group(enriched_tests)

    # Get reports for this project
    project_reports = get_reports_for_project(project.name)
    reports = project_reports.get(project.name, [])

    # Calculate report statistics
    report_stats = {
        "total": len(reports),
        "html_count": len([r for r in reports if r["file_type"] == ".html"]),
        "pdf_count": len([r for r in reports if r["file_type"] == ".pdf"]),
    }

    context = {
        "active_section": active_section,
        "project": project,
        "tests": enriched_tests,
        "grouped_tests": grouped_tests,
        "search_data": json.dumps(enriched_tests),
        "datasets": datasets_dict,
        "metrics": metrics,
        "active_test_id": test_id,
        "active_method_id": method_id,
        "diagnostic_tests": diagnostic_tests,
        "reports": reports,
        "report_stats": report_stats,
    }

    return render(request, "dashboard.html", context)


def create(request):
    """
    View to display the project creation wizard.
    """
    if request.method == "POST":
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save()
            messages.success(request, f'Project "{project.name}" created successfully!')
            return redirect("projects_show", project_id=project.id)

    metadata = {}
    try:
        if PROCESSING_SERVICE_URL:
            response = requests.get(
                f"{PROCESSING_SERVICE_URL}/web/metadata",
                timeout=3,  # Shorter timeout to prevent blocking
            )
            if response.status_code == 200:
                metadata = response.json()
            else:
                # Service responded but with an error
                messages.warning(
                    request,
                    "Processing service is experiencing issues. Data list may be incomplete.",
                )
    except requests.exceptions.Timeout:
        messages.warning(
            request,
            "Processing service is not responding. You can still create projects, but data selection may be limited.",
        )
    except requests.exceptions.ConnectionError:
        messages.warning(
            request,
            "Cannot connect to processing service. Data list will be loaded dynamically.",
        )
    except requests.RequestException as e:
        messages.warning(
            request, f"Error connecting to processing service: {str(e)[:100]}"
        )

    models = metadata.get("models", {})
    datasets = metadata.get("datasets", {})
    metrics = metadata.get("metrics", {})
    tests = metadata.get("tests", {})

    # Ensure tests metadata is properly formatted for template
    if tests:
        # Handle both dictionary and list formats
        if isinstance(tests, dict):
            for _, test_data in tests.values():
                if (
                    isinstance(test_data, dict)
                    and "metadata" in test_data
                    and isinstance(test_data["metadata"], dict)
                ):
                    # Ensure metadata is a JSON string for the template
                    test_data["metadata"] = json.dumps(test_data["metadata"])
                    # Also add parsed metadata for easy template access
                    test_data["metadata_parsed"] = test_data["metadata"]
        elif isinstance(tests, list):
            for test_data in tests:
                if isinstance(test_data, dict) and "metadata" in test_data:
                    # Ensure metadata is a JSON string for the template
                    if isinstance(test_data["metadata"], dict):
                        # Store parsed version for template access
                        test_data["metadata_parsed"] = test_data["metadata"]
                        test_data["metadata"] = json.dumps(test_data["metadata"])
                    elif isinstance(test_data["metadata"], str):
                        # If it's already a string, try to parse it for template access
                        try:
                            test_data["metadata_parsed"] = json.loads(
                                test_data["metadata"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            test_data["metadata_parsed"] = {}

    # print(f"Models: {models}")
    # print(f"Datasets: {datasets}")
    # print(f"Metrics: {metrics}")
    # print(f"Tests: {tests}")

    mock_request = type("MockRequest", (), {"GET": QueryDict()})()
    available_tasks_response = get_available_tasks(mock_request)

    # Extract tasks data from JsonResponse and flatten all arrays into one
    if hasattr(available_tasks_response, "content"):
        response_data = json.loads(available_tasks_response.content)
        tasks_dict = response_data.get("tasks", {})
        available_tasks = [
            task for task_list in tasks_dict.values() for task in task_list
        ]
    else:
        available_tasks = []

    context = {
        "form": ProjectForm(),
        "models": models,
        "modelsJson": json.dumps(models),
        "datasets": datasets,
        "metrics": metrics,
        "tests": tests,
        "availableTasks": available_tasks,
    }

    return render(request, "setup_wizard.html", context)


def edit(request, project_id):
    """
    View to edit an existing project.
    """
    project = get_object_or_404(Project, id=project_id)

    if request.method == "POST":
        form = ProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            messages.success(request, f'Project "{project.name}" updated successfully!')
            return redirect("projects_show", project_id=project.id)
    else:
        form = ProjectForm(instance=project)

    context = {
        "form": form,
        "project": project,
    }
    return render(request, "edit.html", context)


@require_http_methods(["DELETE", "POST"])
def delete(request, project_id):
    """
    View to delete a project.
    """
    project = get_object_or_404(Project, id=project_id)
    project_name = project.name
    project.delete()
    messages.success(request, f'Project "{project_name}" deleted successfully!')
    return redirect("projects_index")


def enrich_tests_with_activities(tests):
    """
    Enrich tests data with activity information from the navigation service database.
    Also extract tag names from jobs for search functionality.
    """
    enriched_tests = []

    for test in tests:
        enriched_test = test.copy()
        enriched_methods = []

        for method in test.get("methods", []):
            enriched_method = method.copy()
            method_id = method.get("id")

            if method_id:
                # Get activities for this method
                activities = MethodActivity.objects.filter(
                    method_id=method_id
                ).order_by("-timestamp")[:10]  # Get latest 10 activities

                # Convert activities to a format suitable for the frontend
                activity_list = []
                for activity in activities:
                    activity_data = {
                        "id": activity.id,
                        "activity_type": activity.activity_type,
                        "activity_type_display": activity.get_activity_type_display(),
                        "timestamp": activity.timestamp.isoformat(),
                        "changes": activity.changes,
                        "description": activity.description,
                        "change_summary": activity.get_change_summary(),
                    }
                    activity_list.append(activity_data)

                enriched_method["activities"] = activity_list
            else:
                enriched_method["activities"] = []

            # Extract tag names from jobs for search functionality
            tag_names = set()
            for job in method.get("jobs", []):
                for tag in job.get("tags", []):
                    tag_name = tag.get("name", "").strip()
                    if tag_name:
                        tag_names.add(tag_name)

            enriched_method["tag_names"] = list(tag_names)

            enriched_methods.append(enriched_method)

        enriched_test["methods"] = enriched_methods
        enriched_tests.append(enriched_test)

    return enriched_tests


def group_tests_by_group(tests):
    """
    Group tests by their 'group' attribute.
    Tests without a group or with an empty group will be placed in an 'ungrouped' category.
    """
    grouped_tests = {}
    ungrouped_tests = []

    for test in tests:
        group = test.get("group")

        # If group is None, empty string, or just whitespace, treat as ungrouped
        if not group or not group.strip():
            ungrouped_tests.append(test)
        else:
            group = group.strip()
            if group not in grouped_tests:
                grouped_tests[group] = []
            grouped_tests[group].append(test)

    # Create the final structure
    result = []

    # Add grouped tests first (sorted by group name)
    for group_name in sorted(grouped_tests.keys()):
        result.append(
            {
                "group_name": group_name,
                "tests": grouped_tests[group_name],
                "is_group": True,
            }
        )

    # Add ungrouped tests at the end if any exist
    if ungrouped_tests:
        result.append({"group_name": None, "tests": ungrouped_tests, "is_group": False})

    return result


@require_http_methods(["POST"])
def pull_all_from_processing_service(request):
    """
    View to pull all projects, tests, and methods from the processing service
    and create instances in the navigation service.
    """
    from projects.services import processing_sync

    try:
        success = processing_sync.pull_all_from_processing_service()

        if success:
            messages.success(
                request, "Successfully pulled all data from processing service."
            )
        else:
            messages.error(request, "Failed to pull data from processing service.")

    except Exception as e:
        messages.error(request, f"Error pulling data from processing service: {e}")

    # Redirect to the projects list page
    return redirect("projects:open")


def get_project_structure(request, project_id):
    """
    Get the hierarchical structure of a project including tests, methods, and jobs.
    Returns JSON data for the project tree structure.
    """
    from django.http import JsonResponse

    try:
        project = get_object_or_404(Project, id=project_id)

        # Check if PROCESSING_SERVICE_URL is configured
        if not PROCESSING_SERVICE_URL:
            return JsonResponse(
                {
                    "error": "Processing service not configured",
                    "project": {
                        "id": project.id,
                        "name": project.name,
                        "description": project.description,
                    },
                    "tests": [],
                }
            )

        # Get tests for this project directly from processing service
        url = f"{PROCESSING_SERVICE_URL}/projects/{project.id}?include=tests&include=tests.methods&include=tests.methods.jobs&include=tests.methods.jobs.result&include=tests.methods.jobs.log&include=tests.methods.jobs.tags"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return JsonResponse(
                {
                    "error": f"Processing service returned status {response.status_code}",
                    "project": {
                        "id": project.id,
                        "name": project.name,
                        "description": project.description,
                    },
                    "tests": [],
                }
            )

        try:
            project_data = response.json()
            tests_data = project_data.get("tests", [])
        except json.JSONDecodeError as e:
            return JsonResponse(
                {
                    "error": f"Invalid JSON response from processing service: {e!s}",
                    "project": {
                        "id": project.id,
                        "name": project.name,
                        "description": project.description,
                    },
                    "tests": [],
                }
            )

        # Structure the data for the frontend
        structure_data = {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
            },
            "tests": tests_data,
        }

        print(
            f"Returning project structure with {len(tests_data)} tests"
        )  # Debug logging

        return JsonResponse(structure_data)

    except requests.exceptions.ConnectionError:
        # Fallback to local Django models if processing service is not available
        print("Processing service not available, using local Django models")
        return _get_project_structure_from_django(project)
    except requests.exceptions.Timeout:
        print("Processing service timeout, using local Django models")
        return _get_project_structure_from_django(project)

    except Exception as e:
        print(f"Error in get_project_structure: {e!s}")  # Debug logging
        return JsonResponse(
            {"error": f"Error fetching project structure: {e!s}"}, status=500
        )


def _get_project_structure_from_django(project):
    """
    Fallback function to get project structure from local Django models
    when processing service is not available.
    """
    from django.http import JsonResponse

    from projects.models import Test

    try:
        # Get tests from Django models
        tests = Test.objects.filter(project=project).prefetch_related("methods")

        tests_data = []
        for test in tests:
            # Get methods for this test
            methods_data = []
            for method in test.methods.all():
                method_data = {
                    "id": method.id,
                    "name": method.name,
                    "description": method.description,
                    "dataset": method.dataset,
                    "metrics": method.metrics,
                    "parameters": method.parameters,
                    "jobs": [],  # We can't get jobs from Django models easily
                }
                methods_data.append(method_data)

            test_data = {
                "id": test.id,
                "name": test.name,
                "description": test.description,
                "methods": methods_data,
            }
            tests_data.append(test_data)

        structure_data = {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
            },
            "tests": tests_data,
        }

        return JsonResponse(structure_data)

    except Exception as e:
        return JsonResponse(
            {
                "error": f"Error fetching project structure from Django models: {e!s}",
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                },
                "tests": [],
            },
            status=500,
        )


def generate_project_report(request, project_id):
    """
    Generate a comprehensive project report including selected tests, methods, and jobs.
    """
    from datetime import datetime

    from django.http import HttpResponse
    from django.template.loader import render_to_string

    from routes.api import get_jobs

    if request.method != "POST":
        return redirect("projects_show", project_id=project_id)

    project = get_object_or_404(Project, id=project_id)

    # Extract form data
    report_name = request.POST.get("report_name", "").strip()
    report_description = request.POST.get("report_description", "").strip()
    report_format = request.POST.get("report_format", "html")
    include_options = request.POST.getlist("include_options")
    max_samples = int(request.POST.get("max_samples", 10))
    selected_jobs_data = json.loads(request.POST.get("selected_jobs_data", "[]"))

    print(
        f"1. Generating report '{report_name}' for project '{project.name}' with {len(selected_jobs_data)} selected jobs"
    )  # Debug logging

    if not report_name:
        messages.error(request, "Report name is required.")
        return redirect(
            "projects_show_section", project_id=project_id, section="reports"
        )

    if not selected_jobs_data:
        messages.error(
            request, "Please select at least one job to include in the report."
        )
        return redirect(
            "projects_show_section", project_id=project_id, section="reports"
        )

    try:
        # Extract all unique job IDs
        job_ids = list({job["job_id"] for job in selected_jobs_data})

        print(
            f"2. Fetching data for {len(job_ids)} unique job IDs from processing service"
        )  # Debug logging

        # Get job data from processing service
        mock_body = json.dumps({"job_ids": job_ids})
        mock_request = type(
            "MockRequest", (), {"method": "POST", "body": mock_body.encode("utf-8")}
        )()
        jobs_response = get_jobs(mock_request)

        print(
            f"3. Processing service response status for jobs: {jobs_response.status_code}"
        )  # Debug logging

        if jobs_response.status_code != 200:
            messages.error(request, "Failed to fetch job data from processing service.")
            return redirect(
                "projects_show_section", project_id=project_id, section="reports"
            )

        jobs_data = json.loads(jobs_response.content)["jobs"]

        print(
            f"4. Retrieved {len(jobs_data)} jobs from processing service"
        )  # Debug logging

        # Organize jobs by test and method
        organized_data = organize_jobs_by_structure(
            selected_jobs_data, jobs_data, project
        )

        # Prepare context for template
        context = {
            "project": project,
            "report_name": report_name,
            "report_description": report_description,
            "organized_data": organized_data,
            "include_options": include_options,
            "max_samples": max_samples,
            "generated_at": datetime.now(),
            "total_jobs": len(job_ids),
            "total_tests": len(organized_data),
            "total_methods": sum(len(test["methods"]) for test in organized_data),
            "report_type": "Project Report",
        }

        print(
            f"5. Rendering report template in format: {report_format}"
        )  # Debug logging

        # Choose template based on format
        if report_format == "pdf":
            template_name = "project_report_pdf.html"
            content_type = "application/pdf"

            # Generate PDF
            import pdfkit

            html_content = render_to_string(template_name, context)

            pdf_options = {
                "page-size": "A4",
                "encoding": "UTF-8",
                "no-outline": None,
                "quiet": "",
                "margin-top": "0.75in",
                "margin-right": "0.75in",
                "margin-bottom": "0.75in",
                "margin-left": "0.75in",
                "enable-local-file-access": None,
            }

            pdf_content = pdfkit.from_string(html_content, False, options=pdf_options)

            # Save PDF locally if configured
            save_project_report_locally(pdf_content, report_name, project.name, "pdf")

            response = HttpResponse(pdf_content, content_type=content_type)
            filename = f"project_report_{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        else:
            template_name = "project_report.html"
            content_type = "text/html"

            html_content = render_to_string(template_name, context)

            # Save HTML locally if configured
            save_project_report_locally(
                html_content.encode("utf-8"), report_name, project.name, "html"
            )

            response = HttpResponse(html_content, content_type=content_type)
            filename = f"project_report_{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        response["Content-Disposition"] = f'attachment; filename="{filename}"'

        print(
            "6. Report generation completed successfully, sending response"
        )  # Debug logging

        return response

    except Exception as e:
        print(f"Error generating project report: {e!s}")  # Debug logging
        messages.error(request, f"Error generating project report: {e!s}")
        return redirect(
            "projects_show_section", project_id=project_id, section="reports"
        )


def organize_jobs_by_structure(selected_jobs_data, jobs_data, project):
    """
    Organize job data by test and method structure.
    """
    from projects.models import Method, Test

    # Create a mapping of job IDs to job data
    jobs_map = {job["id"]: job for job in jobs_data}

    # Group selected jobs by test and method
    test_method_jobs = {}

    for job_selection in selected_jobs_data:
        test_id = job_selection["test_id"]
        method_id = job_selection["method_id"]
        job_id = job_selection["job_id"]

        if test_id not in test_method_jobs:
            test_method_jobs[test_id] = {}

        if method_id not in test_method_jobs[test_id]:
            test_method_jobs[test_id][method_id] = []

        if job_id in jobs_map:
            test_method_jobs[test_id][method_id].append(jobs_map[job_id])

    # Build the organized structure
    organized_data = []

    for test_id, methods_data in test_method_jobs.items():
        try:
            test = Test.objects.get(id=test_id, project=project)
            test_data = {
                "id": test.id,
                "name": test.name,
                "description": test.description,
                "methods": [],
            }

            for method_id, jobs in methods_data.items():
                try:
                    method = Method.objects.get(id=method_id, test=test)
                    method_data = {
                        "id": method.id,
                        "name": method.name,
                        "description": method.description,
                        "dataset": method.dataset,
                        "metrics": method.metrics,
                        "parameters": method.parameters,
                        "jobs": jobs,
                    }
                    test_data["methods"].append(method_data)
                except Method.DoesNotExist:
                    continue

            if test_data["methods"]:  # Only include tests that have methods with jobs
                organized_data.append(test_data)

        except Test.DoesNotExist:
            continue

    return organized_data


def save_project_report_locally(content, report_name, project_name, file_format):
    """
    Save the project report to local storage if REPORT_DIR is configured.
    """
    report_dir = getattr(settings, "REPORT_DIR", None)

    if not report_dir or report_dir == "Not configured":
        return

    try:
        # Create project-specific directory
        project_report_dir = os.path.join(report_dir, project_name)
        os.makedirs(project_report_dir, exist_ok=True)

        # Generate filename
        safe_report_name = "".join(
            c for c in report_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"project_report_{safe_report_name}_{timestamp}.{file_format}"
        file_path = os.path.join(project_report_dir, filename)

        # Save the file
        import builtins

        if file_format == "pdf":
            # PDF content is already bytes
            with builtins.open(file_path, "wb") as f:
                f.write(content)
        else:
            # HTML content - handle both string and bytes
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            with builtins.open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        print(f"Project report saved locally to: {file_path}")

    except Exception as e:
        print(f"Warning: Could not save project report locally: {e!s}")
