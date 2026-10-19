import json

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse, QueryDict
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from projects.forms import TestForm
from projects.models import Project, Test
from routes.api import get_datasets_by_task, get_metrics_metadata

PROCESSING_SERVICE_URL = settings.PROCESSING_SERVICE_URL
PROCESSING_SERVICE_TIMEOUT = settings.PROCESSING_SERVICE_TIMEOUT


def create(request):
    if request.method == "POST":
        form = TestForm(request.POST)
        if form.is_valid():
            test = form.save()
            messages.success(request, f'Test "{test.name}" created successfully!')
            return redirect(
                "projects_show_section",
                project_id=str(test.project.id),
                section="project",
            )

    return redirect("projects_show", project_id=str(test.project.id))


def edit(request, project_id, test_id):
    """Edit an existing test."""
    project = get_object_or_404(Project, id=project_id)
    test = get_object_or_404(Test, id=test_id, project=project)

    if request.method == "POST":
        form = TestForm(request.POST, instance=test)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, f'Test "{test.name}" updated successfully!')
                return redirect(
                    "projects_show_test",
                    project_id=test.project.id,
                    section="project",
                    test_id=test.id,
                )
            except Exception as e:
                messages.error(request, f"Error updating test: {e!s}")
    else:
        form = TestForm(instance=test)

    mock_request = type(
        "MockRequest", (), {"GET": QueryDict(f"task_type={test.get_task_type()}")}
    )()
    datasets_response = get_datasets_by_task(mock_request)
    parsed_datasets = json.loads(datasets_response.content)["datasets"]
    # print(f"Datasets: {parsed_datasets}")

    mock_request = type("MockRequest", (), {"GET": QueryDict()})()
    metrics_response = get_metrics_metadata(mock_request)
    parsed_metrics = json.loads(metrics_response.content)["metrics"]
    # print(f"Metrics: {parsed_metrics}")

    context = {
        # 'form': form,
        "project": project,
        "test": test,
        "test_parameters": json.dumps(test.parameters) if test.parameters else "{}",
        "datasets": parsed_datasets,
        "metrics": parsed_metrics,
    }

    return render(request, "configure_test.html", context)


@csrf_exempt
def update_parameters(request, project_id, test_id):
    """AJAX endpoint to update test parameters."""
    if request.method != "POST":
        return JsonResponse({"error": "Test not allowed"}, status=405)

    try:
        project = get_object_or_404(Project, id=project_id)
        test = get_object_or_404(Test, id=test_id, project=project)

        data = json.loads(request.body)
        parameters = data.get("parameters", {})

        # Validate parameters structure
        if not isinstance(parameters, dict):
            return JsonResponse(
                {"error": "Parameters must be a valid JSON object"}, status=400
            )

        # Update test parameters
        test.parameters = parameters
        test.save()

        return JsonResponse(
            {
                "success": True,
                "message": "Parameters updated successfully",
                "parameters": test.parameters,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Error updating parameters: {e!s}"}, status=500)
