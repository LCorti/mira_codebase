import csv
import json
import os
from datetime import datetime
from io import StringIO

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse, JsonResponse, QueryDict
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from projects.forms.method_forms import MethodEditForm
from projects.models import Method, Project
from projects.utils.report_utils import get_report_content
from routes.api import dataset_show, get_jobs, get_metrics_metadata

DEFAULT_THRESHOLDS = "0.9,0.65,0.3,0.2"


def edit(request, project_id, method_id):
    """Edit an existing method."""
    project = get_object_or_404(Project, id=project_id)
    method = get_object_or_404(Method, id=method_id, test__project=project)

    if request.method == "POST":
        form = MethodEditForm(request.POST, instance=method)
        if form.is_valid():
            try:
                form.save()
                messages.success(
                    request,
                    f'Method "{method.name}" configuration updated successfully.',
                )
                return redirect(
                    "projects_show_method",
                    project_id=project_id,
                    section="project",
                    method_id=method_id,
                )
            except Exception as e:
                messages.error(request, f"Error updating method configuration: {e!s}")
                print(f"Error saving method: {e}")
        else:
            # Log validation errors for debugging
            print(f"Form validation failed. Errors: {form.errors}")
            messages.error(request, f"Validation errors: {form.errors.as_text()}")
    else:
        form = MethodEditForm(instance=method)  # Create form instance for GET requests

    mock_request = type(
        "MockRequest", (), {"GET": QueryDict(f"dataset_name={method.dataset}")}
    )()
    dataset_response = dataset_show(mock_request)
    parsed_dataset = json.loads(dataset_response.content)
    dataset_name = parsed_dataset.get("dataset", "N/A")

    # Handle case where config is None (e.g., timeout or dataset not loaded yet)
    dataset_config = parsed_dataset.get("config")
    if dataset_config and dataset_config != "null":
        try:
            dataset_content = json.loads(dataset_config).get(dataset_name, {})
        except (json.JSONDecodeError, TypeError):
            dataset_content = {}
            messages.warning(
                request,
                f'Dataset "{dataset_name}" configuration could not be loaded. It may still be downloading from HuggingFace. Please refresh the page.',
            )
    else:
        dataset_content = {}
        if parsed_dataset.get("message"):
            messages.warning(
                request, f'Dataset "{dataset_name}": {parsed_dataset["message"]}'
            )

    # print(f"Dataset content: {dataset_content}")

    mock_request = type("MockRequest", (), {"GET": QueryDict()})()
    metrics_response = get_metrics_metadata(mock_request)
    parsed_metrics = json.loads(metrics_response.content)["metrics"]
    # print(f"Metrics: {parsed_metrics}")

    dataset_metrics = {
        key: value
        for key, value in parsed_metrics.items()
        if key in dataset_content.get("metrics", [])
    }
    # print(f"Dataset metrics: {dataset_metrics}")

    context = {
        "form": form,
        "project": project,
        "method": method,
        "method_parameters": json.dumps(method.parameters)
        if method.parameters
        else "{}",
        "dataset_name": dataset_name,
        "dataset_content": dataset_content,
        "dataset_metrics": dataset_metrics,
        "metrics": parsed_metrics,
    }

    return render(request, "configure_method.html", context)


@csrf_exempt
def update_parameters(request, project_id, method_id):
    """AJAX endpoint to update method parameters."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        project = get_object_or_404(Project, id=project_id)
        method = get_object_or_404(Method, id=method_id, test__project=project)

        data = json.loads(request.body)
        parameters = data.get("parameters", {})

        # Validate parameters structure
        if not isinstance(parameters, dict):
            return JsonResponse(
                {"error": "Parameters must be a valid JSON object"}, status=400
            )

        # Update method parameters
        method.parameters = parameters
        method.save()

        return JsonResponse(
            {
                "success": True,
                "message": "Parameters updated successfully",
                "parameters": method.parameters,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Error updating parameters: {e!s}"}, status=500)


def compare(request, project_id, method_id):
    project = get_object_or_404(Project, id=project_id)
    method = get_object_or_404(Method, id=method_id, test__project=project)

    if request.method == "POST":
        return JsonResponse(
            {"error": "POST method not allowed for comparison view"}, status=405
        )

    # Get the job IDs from the query parameters
    job_ids = request.GET.getlist("jobs")

    # Create a proper mock request for the get_jobs API endpoint
    mock_body = json.dumps({"job_ids": job_ids})
    mock_request = type(
        "MockRequest", (), {"method": "POST", "body": mock_body.encode("utf-8")}
    )()
    jobs_response = get_jobs(mock_request)
    jobs_json = json.loads(jobs_response.content)["jobs"]
    # print(f"Jobs JSON: {jobs_json}")

    task_type = method.parameters.get("task_type")
    if not task_type:
        messages.error(request, "Method task type is not defined.")
        return redirect("projects_show", project_id=project_id)

    # Determine the number of samples to display (use the job with the most results)
    max_samples = 0
    for job in jobs_json:
        if job.get("result") and job["result"].get("results"):
            max_samples = max(max_samples, len(job["result"]["results"]))

    # Create a range for template iteration
    samples_range = list(range(max_samples)) if max_samples > 0 else []

    context = {
        "task_type": task_type,
        "project": project,
        "method": method,
        "job_ids": job_ids,
        "selected_jobs_count": len(job_ids),
        "jobs": jobs_json,
        "samples_range": samples_range,
        "max_samples": max_samples,
        "document_k": method.parameters.get("task_parameters", {}).get("k_dense", 5),
    }

    return render(request, "compare_jobs.html", context)


def generate_report(request, project_id, method_id):
    """Generate a downloadable HTML report for job comparison."""
    project = get_object_or_404(Project, id=project_id)
    method = get_object_or_404(Method, id=method_id, test__project=project)

    # Get the job IDs from the query parameters
    job_ids = request.GET.getlist("jobs")

    # Create a proper mock request for the get_jobs API endpoint
    mock_body = json.dumps({"job_ids": job_ids})
    mock_request = type(
        "MockRequest", (), {"method": "POST", "body": mock_body.encode("utf-8")}
    )()
    jobs_response = get_jobs(mock_request)
    jobs_json = json.loads(jobs_response.content)["jobs"]

    task_type = method.parameters.get("task_type")
    if not task_type:
        messages.error(request, "Method task type is not defined.")
        return redirect("projects_show", project_id=project_id)

    # Determine the number of samples to display (use the job with the most results)
    max_samples = 0
    for job in jobs_json:
        if job.get("result") and job["result"].get("results"):
            max_samples = max(max_samples, len(job["result"]["results"]))

    # Create a range for template iteration
    samples_range = list(range(max_samples)) if max_samples > 0 else []

    context = {
        "task_type": task_type,
        "project": project,
        "method": method,
        "job_ids": job_ids,
        "selected_jobs_count": len(job_ids),
        "jobs": jobs_json,
        "jobs_json": json.dumps(
            jobs_json
        ),  # Add JSON serialized version for JavaScript
        "job_ids_json": json.dumps(
            job_ids
        ),  # Add JSON serialized version for JavaScript
        "samples_range": samples_range,
        "max_samples": max_samples,
        "generated_at": datetime.now(),
        "report_title": f"Diagnostic Report - {method.test.project.model_name}",
        "metric_thresholds": method.parameters.get(
            "metric_thresholds", DEFAULT_THRESHOLDS
        ),
    }

    # Render the report template
    html_content = render_to_string("job_comparison_report.html", context)

    # Save the report locally if REPORT_DIR is configured
    report_dir = getattr(settings, "REPORT_DIR", None)
    if report_dir and report_dir != "Not configured":
        try:
            # Ensure the base REPORT_DIR exists
            os.makedirs(report_dir, exist_ok=True)

            # Sanitize project name for use in filesystem paths
            safe_project_name = project.name.replace("/", "_").replace("\\", "_")

            # Create the project-specific reports directory if it doesn't exist
            project_report_dir = os.path.join(report_dir, safe_project_name)
            os.makedirs(project_report_dir, exist_ok=True)

            # Sanitize method name for use in filename
            safe_method_name = method.name.replace("/", "_").replace("\\", "_")

            # Generate filename
            filename = f"job_comparison_report_{safe_project_name}_{safe_method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            file_path = os.path.join(project_report_dir, filename)

            # Save the HTML content to the local file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Report saved locally to: {file_path}")
        except Exception as e:
            print(f"Warning: Could not save report locally: {e!s}")

    # Create the HTTP response with the HTML file
    response = HttpResponse(html_content, content_type="text/html")
    safe_project_name = project.name.replace("/", "_").replace("\\", "_")
    safe_method_name = method.name.replace("/", "_").replace("\\", "_")
    filename = f"job_comparison_report_{safe_project_name}_{safe_method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    return response


def generate_pdf_report(request, project_id, method_id):
    """Generate a downloadable PDF report for job comparison."""
    import pdfkit  # Import here to avoid issues if pdf generation is not used elsewhere

    project = get_object_or_404(Project, id=project_id)
    method = get_object_or_404(Method, id=method_id, test__project=project)

    # Get the job IDs from the query parameters
    job_ids = request.GET.getlist("jobs")

    # Create a proper mock request for the get_jobs API endpoint
    mock_body = json.dumps({"job_ids": job_ids})
    mock_request = type(
        "MockRequest", (), {"method": "POST", "body": mock_body.encode("utf-8")}
    )()
    jobs_response = get_jobs(mock_request)
    jobs_json = json.loads(jobs_response.content)["jobs"]

    task_type = method.parameters.get("task_type")
    if not task_type:
        messages.error(request, "Method task type is not defined.")
        return redirect("projects_show", project_id=project_id)

    # Determine the number of samples to display (use the job with the most results)
    max_samples = 0
    for job in jobs_json:
        if job.get("result") and job["result"].get("results"):
            max_samples = max(max_samples, len(job["result"]["results"]))

    # Create a range for template iteration
    samples_range = list(range(max_samples)) if max_samples > 0 else []

    context = {
        "task_type": task_type,
        "project": project,
        "method": method,
        "job_ids": job_ids,
        "selected_jobs_count": len(job_ids),
        "jobs": jobs_json,
        "jobs_json": json.dumps(
            jobs_json
        ),  # Add JSON serialized version for JavaScript
        "job_ids_json": json.dumps(
            job_ids
        ),  # Add JSON serialized version for JavaScript
        "samples_range": samples_range,
        "max_samples": max_samples,
        "generated_at": datetime.now(),
        "report_title": f"Job Comparison Report - {method.name}",
        "metric_thresholds": method.parameters.get(
            "metric_thresholds", DEFAULT_THRESHOLDS
        ),
    }

    # Render the PDF-optimized report template to HTML
    html_content = render_to_string("job_comparison_report_pdf.html", context)

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

    # Generate PDF from the HTML content
    pdf_content = pdfkit.from_string(html_content, False, options=pdf_options)

    # Save the PDF report locally if REPORT_DIR is configured
    report_dir = getattr(settings, "REPORT_DIR", None)
    if report_dir and report_dir != "Not configured":
        try:
            # Ensure the base REPORT_DIR exists
            os.makedirs(report_dir, exist_ok=True)

            # Sanitize project name for use in filesystem paths
            safe_project_name = project.name.replace("/", "_").replace("\\", "_")

            # Create the project-specific reports directory if it doesn't exist
            project_report_dir = os.path.join(report_dir, safe_project_name)
            os.makedirs(project_report_dir, exist_ok=True)

            # Sanitize method name for use in filename
            safe_method_name = method.name.replace("/", "_").replace("\\", "_")

            # Generate filename
            filename = f"job_comparison_report_{safe_project_name}_{safe_method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = os.path.join(project_report_dir, filename)

            # Save the PDF content to the local file
            with open(file_path, "wb") as f:
                f.write(pdf_content)

            print(f"PDF report saved locally to: {file_path}")
        except Exception as e:
            print(f"Warning: Could not save PDF report locally: {e!s}")

    response = HttpResponse(pdf_content, content_type="application/pdf")
    safe_project_name = project.name.replace("/", "_").replace("\\", "_")
    safe_method_name = method.name.replace("/", "_").replace("\\", "_")
    filename = f"job_comparison_report_{safe_project_name}_{safe_method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    return response


def view_report(request, project_id, report_id):
    """View a specific report in the browser."""
    project = get_object_or_404(Project, id=project_id)

    content, content_type = get_report_content(report_id, project.name)

    if content is None:
        return HttpResponse("Report not found", status=404)

    response = HttpResponse(content, content_type=content_type)

    # Don't force download for viewing
    if content_type == "application/pdf":
        response["Content-Disposition"] = "inline"

    return response


def download_report(request, project_id, report_id):
    """Download a specific report."""
    from projects.utils.report_utils import get_reports_for_project

    project = get_object_or_404(Project, id=project_id)

    # Get report metadata to determine filename
    reports = get_reports_for_project(project.name)
    project_reports = reports.get(project.name, [])
    report = next((r for r in project_reports if r["id"] == report_id), None)

    if not report:
        return HttpResponse("Report not found", status=404)

    content, content_type = get_report_content(report_id, project.name)

    if content is None:
        return HttpResponse("Report not found", status=404)

    response = HttpResponse(content, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="{report["name"]}"'

    return response


@require_http_methods(["DELETE"])
def delete_report(request, project_id, report_id):
    """Delete a specific report file."""
    from projects.utils.report_utils import get_reports_for_project

    project = get_object_or_404(Project, id=project_id)

    # Get report metadata to find the file path
    reports = get_reports_for_project(project.name)
    project_reports = reports.get(project.name, [])
    report = next((r for r in project_reports if r["id"] == report_id), None)

    if not report:
        return JsonResponse({"error": "Report not found"}, status=404)

    try:
        # Delete the file
        if os.path.exists(report["file_path"]):
            os.remove(report["file_path"])
            return JsonResponse(
                {"success": True, "message": "Report deleted successfully"}
            )
        else:
            return JsonResponse({"error": "Report file not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"Error deleting report: {e!s}"}, status=500)


def export_results(request, project_id, method_id):
    """Export job results in JSON or CSV format."""
    project = get_object_or_404(Project, id=project_id)
    method = get_object_or_404(Method, id=method_id, test__project=project)

    # Get the export format from query parameters (default to JSON)
    export_format = request.GET.get("format", "json").lower()
    job_ids = request.GET.getlist("jobs")

    if not job_ids:
        return JsonResponse({"error": "No job IDs provided"}, status=400)

    try:
        # Create a proper mock request for the get_jobs API endpoint
        mock_body = json.dumps({"job_ids": job_ids})
        mock_request = type(
            "MockRequest", (), {"method": "POST", "body": mock_body.encode("utf-8")}
        )()
        jobs_response = get_jobs(mock_request)
        jobs_json = json.loads(jobs_response.content)["jobs"]

        # Structure the data for export
        jobs_data = []
        for job in jobs_json:
            export_data = {
                "job_id": job.get("id"),
                "method_name": method.name,
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "finished_at": job.get("finished_at"),
                "runtime_seconds": job.get("runtime_seconds"),
                "parameters": job.get("parameters", {}),
                "results": job.get("result", {}).get("results", {}),
                "metrics": job.get("result", {}).get("metrics", {}),
                "diagnostics": job.get("result", {}).get("diagnostics", {}),
                "inference_results": job.get("result", {}).get("results", []),
            }
            jobs_data.append(export_data)

        if not jobs_data:
            return JsonResponse({"error": "No valid job data found"}, status=404)

        if export_format == "csv":
            return _export_csv(jobs_data, method.name)
        else:
            return _export_json(jobs_data, method.name)

    except Exception as e:
        return JsonResponse({"error": f"Error exporting results: {e!s}"}, status=500)


def _export_json(jobs_data, method_name):
    """Export jobs data as JSON."""
    response_data = {
        "method_name": method_name,
        "export_timestamp": datetime.now().isoformat(),
        "total_jobs": len(jobs_data),
        "jobs": jobs_data,
    }

    response = HttpResponse(
        json.dumps(response_data, indent=2), content_type="application/json"
    )
    safe_method_name = method_name.replace("/", "_").replace("\\", "_")
    filename = (
        f"{safe_method_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def _export_csv(jobs_data, method_name):
    """Export jobs data as CSV."""
    output = StringIO()

    if not jobs_data:
        return HttpResponse("No data to export", content_type="text/plain")

    # Get all possible field names from the jobs data
    fieldnames = set()
    for job in jobs_data:
        fieldnames.update(_flatten_dict(job).keys())

    fieldnames = sorted(fieldnames)

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for job in jobs_data:
        flattened_job = _flatten_dict(job)
        writer.writerow(flattened_job)

    response = HttpResponse(output.getvalue(), content_type="text/csv")
    safe_method_name = method_name.replace("/", "_").replace("\\", "_")
    filename = (
        f"{safe_method_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def _flatten_dict(data, parent_key="", sep="_"):
    """Flatten nested dictionaries for CSV export."""
    items = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            # For lists, create numbered columns or serialize as JSON string
            if value and isinstance(value[0], dict):
                # If it's a list of dicts, serialize as JSON
                items.append((new_key, json.dumps(value)))
            else:
                # For simple lists, join with semicolons
                items.append((new_key, ";".join(map(str, value))))
        else:
            items.append((new_key, value))

    return dict(items)
