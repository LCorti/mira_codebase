import json

import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from projects.models import Method, Project, Test

PROCESSING_SERVICE_URL = settings.PROCESSING_SERVICE_URL


# Health Check
def status(request):
    """
    API endpoint to check processing service status asynchronously.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "status": "Not Configured",
                "message": "Processing service URL not configured",
            }
        )

    try:
        # Use a shorter timeout and disable retries to prevent blocking
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/health",
            timeout=(2, 3),  # (connect_timeout, read_timeout)
            allow_redirects=False,
        )
        if response.status_code == 200:
            return JsonResponse(
                {"status": "Connected", "message": "Processing service is healthy"}
            )
        else:
            return JsonResponse(
                {
                    "status": "Disconnected",
                    "message": f"Processing service returned status {response.status_code}",
                }
            )
    except requests.exceptions.Timeout:
        return JsonResponse(
            {
                "status": "Disconnected",
                "message": "Connection timeout - service may be down or slow",
            }
        )
    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {
                "status": "Disconnected",
                "message": "Cannot connect to processing service",
            }
        )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "status": "Disconnected",
                "message": f"Service error: {str(e)[:100]}",  # Limit error message length
            }
        )


# Projects
def create_tests_and_methods(request):
    """
    API endpoint to create tests and methods for a specific project.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    project_id = request.POST.get("project_id")
    if not project_id:
        return JsonResponse({"error": "project_id parameter is required"}, status=400)

    # Handle both form data and JSON
    test_ids_str = request.POST.get("test_ids")
    if test_ids_str:
        try:
            test_ids = json.loads(test_ids_str)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid test_ids format"}, status=400)
    else:
        test_ids = request.POST.getlist("test_ids")

    if not test_ids:
        return JsonResponse({"error": "test_ids parameter is required"}, status=400)

    # Get optional group_name parameter
    group_name = request.POST.get("group_name", "").strip()

    try:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        url = f"{PROCESSING_SERVICE_URL}/projects/duplicate-tests-and-methods?project_id={project_id}&auto_run=false"
        if group_name:
            url += f"&group_id={group_name}"
        response = requests.post(url, headers=headers, json=test_ids, timeout=3)
        if response.status_code == 201:
            response_data = response.json()

            # Create Test and Method objects in Django
            try:
                # Get the project instance
                project = Project.objects.get(id=project_id)

                created_tests = []
                created_methods = []

                for test_data in response_data:
                    # Create Test object
                    test = Test.objects.create(
                        id=test_data["id"],
                        name=test_data.get("name", "Unnamed Test"),
                        description=test_data.get("description", ""),
                        diagnostic=True,
                        diagnostic_only=test_data.get("diagnostic_only", False),
                        group=test_data.get("group", None),
                        parameters=test_data.get("parameters", {}),
                        datasets=list(
                            test_data.get("parameters", {}).get("dataset", {}).keys()
                        ),
                        metrics=test_data.get("parameters", {}).get("metrics", {}),
                        project=project,
                        sync_status="synced",
                        metadata=test_data.get("metadata", {}),
                    )
                    created_tests.append(test)

                    # Create Method objects for this test
                    for method_data in test_data.get("methods", []):
                        method = Method.objects.create(
                            id=method_data["id"],
                            name=method_data.get("name", "Unnamed Method"),
                            description=method_data.get("description", ""),
                            diagnostic_only=test_data.get("diagnostic_only", False),
                            dataset=method_data.get("dataset_name", ""),
                            metrics=method_data.get("metrics", []),
                            parameters=method_data.get("parameters", {}),
                            test=test,
                            job_ids=method_data.get("job_ids", []),
                            result_ids=method_data.get("result_ids", []),
                            sync_status="synced",
                        )
                        created_methods.append(method)

                return JsonResponse(
                    {
                        "status": "success",
                        "data": response_data,
                        "created_tests": len(created_tests),
                        "created_methods": len(created_methods),
                    }
                )

            except Project.DoesNotExist:
                return JsonResponse(
                    {"error": f"Project with id {project_id} not found"}, status=404
                )
            except Exception as e:
                return JsonResponse(
                    {"error": f"Error creating Django objects: {e!s}"}, status=500
                )
        else:
            return JsonResponse(
                {"error": f"Error creating tests and methods: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)


# Model Endpoints
def download_size(request):
    """
    API endpoint to get the download size of a specific model.
    """
    model_name = request.GET.get("model_name")
    if not model_name:
        return JsonResponse(
            {"error": "model_name parameter is required", "size": 0}, status=400
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "model": model_name,
                "size": 0,
                "message": "Processing service URL not configured",
            }
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/models/download_size?model_name={model_name}",
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(data)
        else:
            return JsonResponse(
                {
                    "model": model_name,
                    "size": 0,
                    "message": f"Error retrieving size: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse(
            {"model": model_name, "size": 0, "message": f"Failed to connect: {e!s}"}
        )


def model_config(request):
    """
    API endpoint to get the configuration of a specific model.
    """
    model_name = request.GET.get("model_name")
    if not model_name:
        return JsonResponse(
            {"error": "model_name parameter is required", "auto_class": None},
            status=400,
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "model": model_name,
                "config": None,
                "message": "Processing service URL not configured",
            }
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/models/config?model_name={model_name}", timeout=3
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {
                    "model": model_name,
                    "config": json.dumps(data.get(model_name, {})),
                    "message": "Config retrieved successfully",
                }
            )
        else:
            return JsonResponse(
                {
                    "model": model_name,
                    "config": None,
                    "message": f"Error retrieving config: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "model": model_name,
                "config": None,
                "message": f"Failed to connect: {e!s}",
            }
        )


# Dataset Endpoints
def get_datasets_metadata(request):
    """
    API endpoint to retrieve datasets metadata.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse({"message": "Processing service URL not configured"})

    try:
        response = requests.get(f"{PROCESSING_SERVICE_URL}/datasets", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {"datasets": data, "message": "Datasets retrieved successfully"}
            )
        else:
            return JsonResponse(
                {
                    "datasets": None,
                    "message": f"Error retrieving datasets: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse({"message": f"Failed to connect: {e!s}"})


def get_datasets_by_task(request):
    """
    API endpoint to retrieve datasets by task types.
    """
    task_type = request.GET.getlist("task_type")
    if not task_type:
        return JsonResponse(
            {"error": "task_type parameter is required", "datasets": None}, status=400
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "task_type": task_type,
                "datasets": None,
                "message": "Processing service URL not configured",
            }
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/datasets?task_types={task_type}&include_metrics=true",
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {
                    "task_type": task_type,
                    "datasets": data,
                    "message": "Datasets retrieved successfully",
                }
            )
        else:
            return JsonResponse(
                {
                    "task_type": task_type,
                    "datasets": None,
                    "message": f"Error retrieving datasets: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "task_type": task_type,
                "datasets": None,
                "message": f"Failed to connect: {e!s}",
            }
        )


def dataset_config(request):
    """
    API endpoint to get the HF configuration of a specific dataset.
    """
    dataset_name = request.GET.get("dataset_name")
    if not dataset_name:
        return JsonResponse(
            {"error": "dataset_name parameter is required", "config": None}, status=400
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "config": None,
                "message": "Processing service URL not configured",
            }
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/datasets/config?dataset_name={dataset_name}",
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "config": json.dumps(data),
                    "message": "Config retrieved successfully",
                }
            )
        else:
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "config": None,
                    "message": f"Error retrieving config: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "config": None,
                "message": f"Failed to connect: {e!s}",
            }
        )


def dataset_show(request):
    """
    API endpoint to get the configuration of a specific dataset.
    """
    dataset_name = request.GET.get("dataset_name")
    if not dataset_name:
        return JsonResponse(
            {"error": "dataset_name parameter is required", "splits": None}, status=400
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "splits": None,
                "message": "Processing service URL not configured",
            }
        )

    try:
        # Use longer timeout for dataset loading as it may need to download from HuggingFace
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/datasets/show?dataset_name={dataset_name}&include_metrics=true&include_diagnostics=true",
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "config": json.dumps(data),
                    "message": "Config retrieved successfully",
                }
            )
        else:
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "config": None,
                    "message": f"Error retrieving config: {response.status_code}",
                }
            )
    except requests.Timeout:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "config": None,
                "message": "Request timeout - dataset may be downloading from HuggingFace. Please try again.",
            }
        )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "config": None,
                "message": f"Failed to connect: {e!s}",
            }
        )


def dataset_metrics(request):
    """
    API endpoint to get the metrics of a specific dataset (does not have to be cached).
    """
    dataset_name = request.GET.get("dataset_name")
    if not dataset_name:
        return JsonResponse(
            {"error": "dataset_name parameter is required", "metrics": None}, status=400
        )

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "metrics": None,
                "message": "Processing service URL not configured",
            }
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/datasets/metrics?dataset_name={dataset_name}",
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "metrics": json.dumps(data),
                    "message": "Metrics retrieved successfully",
                }
            )
        else:
            return JsonResponse(
                {
                    "dataset": dataset_name,
                    "metrics": None,
                    "message": f"Error retrieving metrics: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse(
            {
                "dataset": dataset_name,
                "metrics": None,
                "message": f"Failed to connect: {e!s}",
            }
        )


# Metrics Endpoints
def get_metrics_metadata(request):
    """
    API endpoint to retrieve metrics metadata.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse({"message": "Processing service URL not configured"})

    try:
        response = requests.get(f"{PROCESSING_SERVICE_URL}/metrics/metadata", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(
                {"metrics": data, "message": "Metrics retrieved successfully"}
            )
        else:
            return JsonResponse(
                {
                    "metrics": None,
                    "message": f"Error retrieving metrics: {response.status_code}",
                }
            )
    except requests.RequestException as e:
        return JsonResponse({"message": f"Failed to connect: {e!s}"})


# Jobs Endpoints
def get_jobs(request):
    """
    API endpoint to retrieve jobs.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            job_ids = data.get("job_ids", [])

            if not job_ids:
                return JsonResponse(
                    {"error": "job_ids parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {"Accept": "application/json"}
                job_ids_params = "&".join([f"job_ids={job_id}" for job_id in job_ids])
                url = f"{PROCESSING_SERVICE_URL}/jobs/batch/show?{job_ids_params}&include=result&include=log&include=tags"

                response = requests.get(url, headers=headers, timeout=3)
                if response.status_code == 200:
                    response_data = response.json()
                    # Handle both old format (list) and new format (dict with jobs/not_found)
                    if isinstance(response_data, dict) and "jobs" in response_data:
                        return JsonResponse(
                            {
                                "message": "Jobs retrieved successfully",
                                "jobs": response_data.get("jobs", []),
                                "not_found": response_data.get("not_found", []),
                            },
                            status=200,
                        )
                    else:
                        # Fallback for old format
                        return JsonResponse(
                            {
                                "message": "Jobs retrieved successfully",
                                "jobs": response_data,
                                "not_found": [],
                            },
                            status=200,
                        )
                else:
                    return JsonResponse(
                        {"error": f"Error retrieving jobs: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def create_jobs(request):
    """
    API endpoint to create new jobs.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            method_ids = data.get("method_ids", [])
            job_types = data.get("job_types", [])

            if not method_ids:
                print("Method IDs missing in request")
                return JsonResponse(
                    {"error": "method_ids parameter is required"}, status=400
                )

            if not job_types:
                print("Job types not provided, defaulting to FULL")
                job_types = ["FULL"] * len(method_ids)

            if not PROCESSING_SERVICE_URL:
                print("Processing service URL not configured")
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                body = {
                    "jobs": [{"method_id": mid} for mid in method_ids],
                    "job_types": job_types if job_types else ["FULL"] * len(method_ids),
                }
                url = f"{PROCESSING_SERVICE_URL}/jobs/bulk-create"
                response = requests.post(url, json=body, headers=headers, timeout=3)
                if response.status_code == 201:
                    return JsonResponse(
                        {
                            "message": "Jobs created successfully",
                            "jobs": response.json(),
                        },
                        status=201,
                    )
                else:
                    print(f"Error creating jobs: {response.status_code}")
                    return JsonResponse(
                        {"error": f"Error creating job: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                print(f"Failed to connect: {e!s}")
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            print("Invalid JSON data in request")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def create_job_diagnostic(request):
    """
    API endpoint to create new diagnostic jobs.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            job_id = data.get("job_id", "")
            sample = data.get("sample", {})
            diagnostic = data.get("diagnostic", "lxt")

            if not job_id:
                print("Job ID missing in request")
                return JsonResponse(
                    {"error": "job_id parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                print("Processing service URL not configured")
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                body = {"sample": sample}
                url = f"{PROCESSING_SERVICE_URL}/jobs/duplicate-{diagnostic}?job_id={job_id}"
                response = requests.post(url, json=body, headers=headers, timeout=3)
                if response.status_code == 201:
                    return JsonResponse(
                        {"message": "Job created successfully", "job": response.json()},
                        status=201,
                    )
                else:
                    print(f"Error creating job: {response.status_code}")
                    return JsonResponse(
                        {"error": f"Error creating job: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                print(f"Failed to connect: {e!s}")
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            print("Invalid JSON data in request")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def duplicate_jobs(request):
    """
    API endpoint to duplicate existing jobs.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            job_ids = data.get("job_ids", [])
            job_types = data.get("job_types", [])

            if not job_ids:
                return JsonResponse(
                    {"error": "job_ids parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                # Send the request body as expected by the duplicate endpoint
                body = {"job_ids": job_ids, "job_types": job_types}
                url = f"{PROCESSING_SERVICE_URL}/jobs/duplicate"
                response = requests.post(url, json=body, headers=headers, timeout=3)
                if response.status_code == 201:
                    return JsonResponse(
                        {
                            "message": "Jobs duplicated successfully",
                            "jobs": response.json(),
                        },
                        status=201,
                    )
                else:
                    return JsonResponse(
                        {"error": f"Error duplicating jobs: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def cancel_job(request):
    """
    API endpoint to cancel a job.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            job_id = data.get("job_id")

            if not job_id:
                return JsonResponse(
                    {"error": "job_id parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {"Accept": "application/json"}
                url = f"{PROCESSING_SERVICE_URL}/jobs/{job_id}/cancel"
                response = requests.post(url, headers=headers, timeout=3)
                if response.status_code == 200:
                    return JsonResponse(
                        {
                            "message": "Job cancelled successfully",
                            "job": response.json(),
                        },
                        status=200,
                    )
                else:
                    return JsonResponse(
                        {"error": f"Error cancelling job: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


# Tasks Endpoints
def get_available_tasks(request):
    """
    API endpoint to retrieve available tasks.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    try:
        headers = {"Accept": "application/json"}

        url = f"{PROCESSING_SERVICE_URL}/tasks/available-tasks"
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            return JsonResponse(
                {
                    "message": "Available tasks retrieved successfully",
                    "tasks": response.json(),
                },
                status=200,
            )
        else:
            return JsonResponse(
                {"error": f"Error retrieving available tasks: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"message": f"Failed to connect: {e!s}"}, status=500)


def get_datasets_by_task(request):
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    try:
        headers = {"Accept": "application/json"}

        task_type = request.GET.get("task_type")
        url = f"{PROCESSING_SERVICE_URL}/datasets/by-tasks?task_types={task_type}&include_metrics=true"
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            return JsonResponse(
                {
                    "message": "Datasets retrieved successfully",
                    "datasets": response.json(),
                },
                status=200,
            )
        else:
            return JsonResponse(
                {"error": f"Error retrieving datasets: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)


def get_tasks_from_dataset_readme(request):
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    try:
        headers = {"Accept": "application/json"}

        dataset_name = request.GET.get("dataset_name")
        url = (
            f"{PROCESSING_SERVICE_URL}/datasets/show-tasks?dataset_name={dataset_name}"
        )
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            return JsonResponse(
                {"message": "Tasks retrieved successfully", "tasks": response.json()},
                status=200,
            )
        else:
            return JsonResponse(
                {"error": f"Error retrieving tasks: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)


# Groups Endpoints
def update_group_name(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            group_name = data.get("group_name")
            new_name = data.get("new_name")

            if not group_name:
                return JsonResponse(
                    {"error": "group_name parameter is required"}, status=400
                )

            if not new_name:
                return JsonResponse(
                    {"error": "new_name parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {"Accept": "application/json"}
                url = f"{PROCESSING_SERVICE_URL}/web/groups/update-name?group_name={group_name}&new_name={new_name}"
                response = requests.post(url, headers=headers, timeout=3)
                if response.status_code == 200:
                    return JsonResponse(
                        {
                            "message": "Group name updated successfully",
                            "group": response.json(),
                        },
                        status=200,
                    )
                else:
                    return JsonResponse(
                        {"error": f"Error updating group name: {response.status_code}"},
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


# Tags Endpoints
def get_available_tags(request):
    """
    API endpoint to retrieve available tags.
    """
    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    try:
        response = requests.get(f"{PROCESSING_SERVICE_URL}/tags/available", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(data)
        else:
            return JsonResponse(
                {"error": f"Error retrieving available tags: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)


def get_job_tags(request):
    """
    API endpoint to retrieve tags for a specific job.
    """
    job_id = request.GET.get("job_id")
    if not job_id:
        return JsonResponse({"error": "job_id parameter is required"}, status=400)

    if not PROCESSING_SERVICE_URL:
        return JsonResponse(
            {"error": "Processing service URL not configured"}, status=500
        )

    try:
        response = requests.get(
            f"{PROCESSING_SERVICE_URL}/tags/instances?job_id={job_id}", timeout=3
        )
        if response.status_code == 200:
            data = response.json()
            return JsonResponse(data, safe=False)
        else:
            return JsonResponse(
                {"error": f"Error retrieving job tags: {response.status_code}"},
                status=response.status_code,
            )
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)


@csrf_exempt
def create_job_tag(request):
    """
    API endpoint to create a new job tag.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            tag_name = data.get("tag_name")
            job_id = data.get("job_id")
            comment = data.get("comment", "")
            filters = data.get("filters", [])

            if not tag_name:
                return JsonResponse(
                    {"error": "tag_name parameter is required"}, status=400
                )

            if not job_id:
                return JsonResponse(
                    {"error": "job_id parameter is required"}, status=400
                )

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                payload = {
                    "tag_name": tag_name,
                    "job_id": job_id,
                    "comment": comment,
                    "filters": filters,
                }
                response = requests.post(
                    f"{PROCESSING_SERVICE_URL}/tags/instances",
                    headers=headers,
                    json=payload,
                    timeout=3,
                )
                if response.status_code == 201:
                    data = response.json()
                    return JsonResponse(data, status=201)
                else:
                    error_data = response.json() if response.content else {}
                    return JsonResponse(
                        {
                            "error": error_data.get(
                                "detail", f"Error creating tag: {response.status_code}"
                            )
                        },
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def update_job_tag(request, tag_id):
    """
    API endpoint to update a job tag.
    """

    if request.method == "PUT":
        try:
            data = json.loads(request.body)
            comment = data.get("comment")
            filters = data.get("filters")

            if not PROCESSING_SERVICE_URL:
                return JsonResponse(
                    {"error": "Processing service URL not configured"}, status=500
                )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                payload = {}
                if comment is not None:
                    payload["comment"] = comment
                if filters is not None:
                    payload["filters"] = filters

                response = requests.put(
                    f"{PROCESSING_SERVICE_URL}/tags/instances/{tag_id}",
                    headers=headers,
                    json=payload,
                    timeout=3,
                )
                if response.status_code == 200:
                    data = response.json()
                    return JsonResponse(data)
                else:
                    error_data = response.json() if response.content else {}
                    return JsonResponse(
                        {
                            "error": error_data.get(
                                "detail", f"Error updating tag: {response.status_code}"
                            )
                        },
                        status=response.status_code,
                    )
            except requests.RequestException as e:
                return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def delete_job_tag(request, tag_id):
    """
    API endpoint to delete a job tag.
    """

    if request.method == "DELETE":
        if not PROCESSING_SERVICE_URL:
            return JsonResponse(
                {"error": "Processing service URL not configured"}, status=500
            )

        try:
            response = requests.delete(
                f"{PROCESSING_SERVICE_URL}/tags/instances/{tag_id}", timeout=3
            )
            if response.status_code == 204:
                return JsonResponse({"message": "Tag deleted successfully"})
            else:
                error_data = response.json() if response.content else {}
                return JsonResponse(
                    {
                        "error": error_data.get(
                            "detail", f"Error deleting tag: {response.status_code}"
                        )
                    },
                    status=response.status_code,
                )
        except requests.RequestException as e:
            return JsonResponse({"error": f"Failed to connect: {e!s}"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
