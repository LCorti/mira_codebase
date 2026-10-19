from unittest.mock import MagicMock, patch

from django.test import TestCase

from projects.models import Method, Project, Test
from projects.services import ProcessingServiceSync


class TestProcessingServiceSync(TestCase):
    def setUp(self):
        self.sync_service = ProcessingServiceSync()

    @patch("projects.services.requests.get")
    def test_pull_all_from_processing_service_success(self, mock_get):
        """Test successful pull of all data from processing service."""
        # Mock response data
        mock_response_data = [
            {
                "id": "test_project_id",
                "name": "Test Project",
                "description": "Test Description",
                "model_name": "test-model",
                "parameters": {
                    "dataset": {"test_dataset": {}},
                    "metrics": {"test_dataset": {"rouge": {}}},
                },
                "tests": [
                    {
                        "id": "test_test_id",
                        "name": "Test Test",
                        "description": "Test Test Description",
                        "parameters": {
                            "dataset": {"test_dataset": {}},
                            "metrics": {"test_dataset": {"rouge": {}}},
                        },
                        "methods": [
                            {
                                "id": "test_method_id",
                                "name": "Test Method",
                                "description": "Test Method Description",
                                "dataset_name": "test_dataset",
                                "metrics": ["rouge"],
                                "parameters": {},
                                "job_ids": [],
                                "result_ids": [],
                            }
                        ],
                    }
                ],
            }
        ]

        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_get.return_value = mock_response

        # Call the method
        result = self.sync_service.pull_all_from_processing_service()

        # Assertions
        self.assertTrue(result)

        # Check that project was created
        project = Project.objects.get(id="test_project_id")
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.description, "Test Description")
        self.assertEqual(project.model_name, "test-model")

        # Check that test was created
        test = Test.objects.get(id="test_test_id")
        self.assertEqual(test.name, "Test Test")
        self.assertEqual(test.project, project)

        # Check that method was created
        method = Method.objects.get(id="test_method_id")
        self.assertEqual(method.name, "Test Method")
        self.assertEqual(method.test, test)
        self.assertEqual(method.dataset, "test_dataset")

    @patch("projects.services.requests.get")
    def test_pull_all_from_processing_service_failure(self, mock_get):
        """Test failure when processing service returns error."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        # Call the method
        result = self.sync_service.pull_all_from_processing_service()

        # Assertions
        self.assertFalse(result)

        # Check that no objects were created
        self.assertEqual(Project.objects.count(), 0)
        self.assertEqual(Test.objects.count(), 0)
        self.assertEqual(Method.objects.count(), 0)

    @patch("projects.services.requests.get")
    def test_pull_all_from_processing_service_update_existing(self, mock_get):
        """Test updating existing objects when pulling data."""
        # Create existing project
        _ = Project.objects.create(
            id="test_project_id", name="Old Name", description="Old Description"
        )

        # Mock response data with updated information
        mock_response_data = [
            {
                "id": "test_project_id",
                "name": "Updated Project Name",
                "description": "Updated Description",
                "model_name": "updated-model",
                "parameters": {
                    "dataset": {"test_dataset": {}},
                    "metrics": {"test_dataset": {"rouge": {}}},
                },
                "tests": [],
            }
        ]

        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_get.return_value = mock_response

        # Call the method
        result = self.sync_service.pull_all_from_processing_service()

        # Assertions
        self.assertTrue(result)

        # Check that project was updated (not created new)
        self.assertEqual(Project.objects.count(), 1)
        updated_project = Project.objects.get(id="test_project_id")
        self.assertEqual(updated_project.name, "Updated Project Name")
        self.assertEqual(updated_project.description, "Updated Description")
        self.assertEqual(updated_project.model_name, "updated-model")
