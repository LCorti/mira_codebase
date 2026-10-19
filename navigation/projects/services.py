import requests
from django.conf import settings
from django.utils import timezone

from .models import Method, Project, Test


class ProcessingServiceSync:
    """
    Service class for synchronizing projects with the processing service.
    """

    def __init__(self):
        self.base_url = settings.PROCESSING_SERVICE_URL
        self.timeout = settings.PROCESSING_SERVICE_TIMEOUT

    def test_connection(self, url: str | None = None) -> bool:
        """Test if processing service is reachable."""
        try:
            test_url = f"{url}/health" if url else f"{self.base_url}/health"
            response = requests.get(test_url, timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"Connection test failed: {e}")
            return False

    def _sync_tests_and_methods_from_response(
        self, project: Project, tests_data: list
    ) -> None:
        """
        Sync tests and methods from processing service response to navigation service.

        Args:
            project: Django Project instance
            tests_data: List of test data from processing service response
        """
        try:
            for test_data in tests_data:
                # Extract datasets from the test parameters if available
                datasets = []
                metrics = {}

                # If test has parameters with dataset info, extract dataset names
                if "parameters" in test_data and "dataset" in test_data["parameters"]:
                    datasets = list(test_data["parameters"]["dataset"].keys())
                if "parameters" in test_data and "metrics" in test_data["parameters"]:
                    metrics = {
                        k: list(v.keys())
                        for k, v in test_data["parameters"]["metrics"].items()
                    }

                # Create or update test
                test, test_created = Test.objects.get_or_create(
                    id=test_data["id"],
                    defaults={
                        "name": test_data["name"],
                        "description": test_data["description"],
                        "parameters": test_data.get("parameters", {}),
                        "project": project,
                        "sync_status": "synced",
                        "last_synced": timezone.now(),
                        "datasets": datasets,
                        "metrics": metrics,
                    },
                )

                # If test already exists, update it
                if not test_created:
                    Test.objects.filter(id=test_data["id"]).update(
                        name=test_data["name"],
                        description=test_data["description"],
                        parameters=test_data.get("parameters", test.parameters),
                        sync_status="synced",
                        last_synced=timezone.now(),
                    )
                    test.refresh_from_db()

                # Sync methods for this test
                if "methods" in test_data:
                    for method_data in test_data["methods"]:
                        method, method_created = Method.objects.get_or_create(
                            id=method_data["id"],
                            defaults={
                                "name": method_data["name"],
                                "description": method_data["description"],
                                "dataset": method_data.get("dataset_name", ""),
                                "metrics": method_data.get("metrics", []),
                                "parameters": method_data.get("parameters", {}),
                                "test": test,
                                "job_ids": method_data.get("job_ids", []),
                                "result_ids": method_data.get("result_ids", []),
                                "sync_status": "synced",
                                "last_synced": timezone.now(),
                            },
                        )

                        # If method already exists, update it
                        if not method_created:
                            Method.objects.filter(id=method_data["id"]).update(
                                name=method_data["name"],
                                description=method_data["description"],
                                dataset=method_data.get("dataset_name", method.dataset),
                                metrics=method_data.get("metrics", method.metrics),
                                parameters=method_data.get(
                                    "parameters", method.parameters
                                ),
                                job_ids=method_data.get("job_ids", method.job_ids),
                                result_ids=method_data.get(
                                    "result_ids", method.result_ids
                                ),
                                sync_status="synced",
                                last_synced=timezone.now(),
                            )
                            method.refresh_from_db()

            # Update project's test_ids based on synced tests
            synced_test_ids = [test_data["id"] for test_data in tests_data]
            Project.objects.filter(id=project.id).update(test_ids=synced_test_ids)

        except Exception as e:
            print(f"Error syncing tests and methods: {e}")
            # Log error but don't fail the entire sync operation

    def sync_project_to_processing_service(
        self, project: Project, request_type: str
    ) -> bool:
        """
        Send project to processing service.

        Args:
            project: Django Project instance

        Returns:
            bool: True if sync successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            data = project.to_processing_service_format()
            if request_type == "UPDATE":  # Update existing project
                url = f"{self.base_url}/projects/{project.id}"
                response = requests.put(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            elif request_type == "CREATE":  # Create new project
                url = f"{self.base_url}/projects/create-with-tests-and-methods?auto_run={project.auto_run}"
                response = requests.post(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            else:
                print(f"Unknown request type: {request_type}")
                Project.objects.filter(id=project.id).update(sync_status="error")
                return False

            if response.status_code not in [200, 201]:
                print(
                    f"Failed to sync project: {response.status_code} - {response.text}"
                )
                Project.objects.filter(id=project.id).update(sync_status="error")
                return False

            response_data = response.json()

            # Sync tests and methods from processing service response
            if request_type == "CREATE" and "tests" in response_data["project"]:
                self._sync_tests_and_methods_from_response(
                    project, response_data["project"]["tests"]
                )

            Project.objects.filter(id=project.id).update(
                sync_status="synced", last_synced=timezone.now()
            )
            return True

        except requests.RequestException:
            Project.objects.filter(id=project.id).update(sync_status="error")
            return False

    def _sync_methods_from_response(self, test: Test, methods_data: list) -> None:
        """
        Sync methods from processing service response to navigation service.

        Args:
            test: Django Test instance
            methods_data: List of method data from processing service response
        """
        try:
            for method_data in methods_data:
                method, method_created = Method.objects.get_or_create(
                    id=method_data["id"],
                    defaults={
                        "name": method_data["name"],
                        "description": method_data["description"],
                        "dataset": method_data.get("dataset_name", ""),
                        "metrics": method_data.get("metrics", []),
                        "parameters": method_data.get("parameters", {}),
                        "test": test,
                        "job_ids": method_data.get("job_ids", []),
                        "result_ids": method_data.get("result_ids", []),
                        "sync_status": "synced",
                        "last_synced": timezone.now(),
                    },
                )

                # If method already exists, update it
                if not method_created:
                    Method.objects.filter(id=method_data["id"]).update(
                        name=method_data["name"],
                        description=method_data["description"],
                        dataset=method_data.get("dataset_name", method.dataset),
                        metrics=method_data.get("metrics", method.metrics),
                        parameters=method_data.get("parameters", method.parameters),
                        job_ids=method_data.get("job_ids", method.job_ids),
                        result_ids=method_data.get("result_ids", method.result_ids),
                        sync_status="synced",
                        last_synced=timezone.now(),
                    )

            # Update test's method_ids based on synced methods
            synced_method_ids = [method_data["id"] for method_data in methods_data]
            Test.objects.filter(id=test.id).update(method_ids=synced_method_ids)

        except Exception as e:
            print(f"Error syncing tests and methods: {e}")
            # Log error but don't fail the entire sync operation

    def sync_test_to_processing_service(self, test: Test, request_type: str) -> bool:
        try:
            data = test.to_processing_service_format()

            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            if request_type == "UPDATE":  # Update existing test
                url = f"{self.base_url}/tests/{test.id}"
                response = requests.put(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            elif request_type == "CREATE":  # Create new test
                url = f"{self.base_url}/tests/create-with-methods"
                response = requests.post(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            else:
                print(f"Unknown request type: {request_type}")
                Test.objects.filter(id=test.id).update(sync_status="error")
                return False

            if response.status_code not in [200, 201]:
                print(f"Failed to sync test: {response.status_code} - {response.text}")
                Test.objects.filter(id=test.id).update(sync_status="error")
                return False

            response_data = response.json()

            # Sync tests and methods from processing service response
            if request_type == "CREATE" and "methods" in response_data["test"]:
                self._sync_methods_from_response(test, response_data["test"]["methods"])

            Test.objects.filter(id=test.id).update(
                sync_status="synced", last_synced=timezone.now()
            )
            return True
        except requests.RequestException:
            Test.objects.filter(id=test.id).update(sync_status="error")
            return False

    def sync_method_to_processing_service(self, method: Method) -> bool:
        try:
            data = method.to_processing_service_format()

            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            url = f"{self.base_url}/methods/{method.id}"
            response = requests.put(
                url, json=data, headers=headers, timeout=self.timeout
            )

            if response.status_code not in [200, 201]:
                print(
                    f"Failed to sync method: {response.status_code} - {response.text}"
                )
                Method.objects.filter(id=method.id).update(sync_status="error")
                return False

            response_data = response.json()

            Method.objects.filter(id=method.id).update(
                sync_status="synced", last_synced=timezone.now()
            )
            return True
        except requests.RequestException:
            Method.objects.filter(id=method.id).update(sync_status="error")
            return False

    def pull_all_from_processing_service(self, url: str | None = None) -> bool:
        """
        Pull all projects, tests, and methods from the processing service and create instances
        in the navigation service. This creates a complete copy of the processing service data.

        Returns:
            bool: True if pull successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            # Fetch all projects with nested tests and methods
            url = (
                f"{url}/projects?include=tests&include=tests.methods"
                if url
                else f"{self.base_url}/projects?include=tests&include=tests.methods"
            )
            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                print(
                    f"Failed to fetch projects from processing service: {response.status_code} - {response.text}"
                )
                return False

            projects_data = response.json()

            # Process each project
            for project_data in projects_data:
                self._create_or_update_project_from_data(project_data)

            return True

        except requests.RequestException as e:
            print(f"Error pulling data from processing service: {e}")
            return False

    def _create_or_update_project_from_data(self, project_data: dict) -> Project:
        """
        Create or update a project from processing service data.

        Args:
            project_data: Project data from processing service response

        Returns:
            Project: The created or updated project instance
        """
        try:
            # Extract datasets and metrics from project parameters
            datasets = []
            metrics = {}

            if "parameters" in project_data and "dataset" in project_data["parameters"]:
                datasets = list(project_data["parameters"]["dataset"].keys())
            if "parameters" in project_data and "metrics" in project_data["parameters"]:
                metrics = {
                    k: list(v.keys())
                    for k, v in project_data["parameters"]["metrics"].items()
                }

            # Create or update project
            project, project_created = Project.objects.get_or_create(
                id=project_data["id"],
                defaults={
                    "name": project_data["name"],
                    "description": project_data["description"],
                    "model_name": project_data.get("model_name", ""),
                    "parameters": project_data.get("parameters", {}),
                    "datasets": datasets,
                    "metrics": metrics,
                    "test_ids": [],  # Will be updated after processing tests
                    "sync_status": "synced",
                    "last_synced": timezone.now(),
                },
            )

            # Set the _syncing flag to prevent signals from triggering
            project._syncing = True

            # If project already exists, update it
            if not project_created:
                Project.objects.filter(id=project_data["id"]).update(
                    name=project_data["name"],
                    description=project_data["description"],
                    model_name=project_data.get("model_name", project.model_name),
                    parameters=project_data.get("parameters", project.parameters),
                    datasets=datasets,
                    metrics=metrics,
                    sync_status="synced",
                    last_synced=timezone.now(),
                )
                project.refresh_from_db()
                # Set the _syncing flag after refresh
                project._syncing = True
                project.refresh_from_db()

            # Process tests for this project
            if "tests" in project_data:
                test_ids = []
                for test_data in project_data["tests"]:
                    test = self._create_or_update_test_from_data(test_data, project)
                    test_ids.append(test.id)

                # Update project's test_ids
                Project.objects.filter(id=project.id).update(test_ids=test_ids)

            return project

        except Exception as e:
            print(
                f"Error creating/updating project {project_data.get('id', 'unknown')}: {e}"
            )
            raise

    def _create_or_update_test_from_data(
        self, test_data: dict, project: Project
    ) -> Test:
        """
        Create or update a test from processing service data.

        Args:
            test_data: Test data from processing service response
            project: Parent project instance

        Returns:
            Test: The created or updated test instance
        """
        try:
            # Extract datasets and metrics from test parameters
            datasets = []
            metrics = {}

            if "parameters" in test_data and "dataset" in test_data["parameters"]:
                datasets = list(test_data["parameters"]["dataset"].keys())
            if "parameters" in test_data and "metrics" in test_data["parameters"]:
                metrics = {
                    k: list(v.keys())
                    for k, v in test_data["parameters"]["metrics"].items()
                }

            # Create or update test
            test, test_created = Test.objects.get_or_create(
                id=test_data["id"],
                defaults={
                    "name": test_data["name"],
                    "description": test_data["description"],
                    "diagnostic": test_data.get("default", False),
                    "diagnostic_only": test_data.get("diagnostic_only", False),
                    "parameters": test_data.get("parameters", {}),
                    "datasets": datasets,
                    "metrics": metrics,
                    "project": project,
                    "sync_status": "synced",
                    "last_synced": timezone.now(),
                },
            )

            # Set the _syncing flag to prevent signals from triggering
            test._syncing = True

            # If test already exists, update it
            if not test_created:
                Test.objects.filter(id=test_data["id"]).update(
                    name=test_data["name"],
                    description=test_data["description"],
                    parameters=test_data.get("parameters", test.parameters),
                    datasets=datasets,
                    metrics=metrics,
                    project=project,
                    sync_status="synced",
                    last_synced=timezone.now(),
                )
                test.refresh_from_db()
                # Set the _syncing flag after refresh
                test._syncing = True

            # Process methods for this test
            if "methods" in test_data:
                for method_data in test_data["methods"]:
                    self._create_or_update_method_from_data(method_data, test)

            return test

        except Exception as e:
            print(f"Error creating/updating test {test_data.get('id', 'unknown')}: {e}")
            raise

    def _create_or_update_method_from_data(
        self, method_data: dict, test: Test
    ) -> Method:
        """
        Create or update a method from processing service data.

        Args:
            method_data: Method data from processing service response
            test: Parent test instance

        Returns:
            Method: The created or updated method instance
        """
        try:
            # Create or update method
            method, method_created = Method.objects.get_or_create(
                id=method_data["id"],
                defaults={
                    "name": method_data["name"],
                    "description": method_data["description"],
                    "diagnostic_only": method_data.get("diagnostic_only", False),
                    "dataset": method_data.get("dataset_name", ""),
                    "metrics": method_data.get("metrics", []),
                    "parameters": method_data.get("parameters", {}),
                    "test": test,
                    "job_ids": method_data.get("job_ids", []),
                    "result_ids": method_data.get("result_ids", []),
                    "sync_status": "synced",
                    "last_synced": timezone.now(),
                },
            )

            # Set the _syncing flag to prevent signals from triggering
            method._syncing = True

            # If method already exists, update it
            if not method_created:
                Method.objects.filter(id=method_data["id"]).update(
                    name=method_data["name"],
                    description=method_data["description"],
                    dataset=method_data.get("dataset_name", method.dataset),
                    metrics=method_data.get("metrics", method.metrics),
                    parameters=method_data.get("parameters", method.parameters),
                    test=test,
                    job_ids=method_data.get("job_ids", method.job_ids),
                    result_ids=method_data.get("result_ids", method.result_ids),
                    sync_status="synced",
                    last_synced=timezone.now(),
                )
                method.refresh_from_db()
                # Set the _syncing flag after refresh
                method._syncing = True

            return method

        except Exception as e:
            print(
                f"Error creating/updating method {method_data.get('id', 'unknown')}: {e}"
            )
            raise


# Global service instance
processing_sync = ProcessingServiceSync()
