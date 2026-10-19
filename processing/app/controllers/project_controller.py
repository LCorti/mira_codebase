import threading
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from bson import ObjectId

from app.config.registry import (
    get_database_service,
    get_dataset_loader,
    get_job_queue,
    get_model_loader,
)
from app.models.job import JobStatus, JobType
from app.models.project import Project, ProjectUpdate
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")


def _get_projects_collection():
    """
    Returns the projects collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("projects")


def _get_tests_collection():
    """
    Returns the tests collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("tests")


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("jobs")


class ProjectController:
    """
    Controller class for handling project-related business logic.
    """

    @staticmethod
    @contextmanager
    def database_transaction():
        """
        Context manager for database transactions to reduce repetition.
        """
        session = get_database_service().start_session()
        try:
            with session.start_transaction():
                yield session
                session.commit_transaction()
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise

    @staticmethod
    def get_all_projects(include: set[str] | None) -> list[dict[str, Any]]:
        """
        Retrieves all projects from the database.

        Args:
            None

        Returns:
            list: List of all projects as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving projects from the database.
        """
        try:
            projects = _get_projects_collection().find()
            return [Project(**project).to_dict(include=include) for project in projects]
        except Exception as e:
            logger.error(f"Error retrieving projects from the database: {e}")
            raise RuntimeError(f"Failed to retrieve projects from the database: {e}")

    @staticmethod
    def get_project_by_id(project_id: str, include: set[str] | None) -> dict[str, Any]:
        """
        Retrieves a specific project by its ID.

        Args:
            project_id (str): The ID of the project to retrieve.

        Returns:
            dict: The project as a dictionary.
            Project: The project as a Pydantic model.

        Raises:
            RuntimeError: If there is an error retrieving the project from the database.
        """
        try:
            project = _get_projects_collection().find_one({"id": project_id})
            if not project:
                return {}
            return Project(**project).to_dict(include=include)
        except Exception as e:
            logger.error(f"Error retrieving project with ID {project_id}: {e}")
            raise RuntimeError(f"Failed to retrieve project with ID {project_id}: {e}")

    @staticmethod
    def create_project(project: Project) -> dict[str, Any]:
        """
        Creates a new project in the database.

        Args:
            project (Project): The project to create.

        Returns:
            dict: The created project as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the project in the database.
        """
        try:
            with ProjectController.database_transaction():
                result = _get_projects_collection().insert_one(project.to_dict())
                if not result.acknowledged:
                    logger.error(f"Failed to create project {project.id}")
                    raise RuntimeError("Failed to create project")

                # For each method, update project_id
                if project.test_ids:
                    for test_id in project.test_ids:
                        test_result = _get_tests_collection().find_one_and_update(
                            {"id": test_id}, {"$set": {"project_id": project.id}}
                        )
                        if not test_result:
                            logger.warning(
                                f"Test with ID {test_id} not found for project {project.id}"
                            )

                return project.to_dict()
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise RuntimeError(f"Failed to create project: {e}")

    @staticmethod
    def _create_dataset_parameters(
        project_parameters: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        """
        Creates dataset-specific parameters by merging project parameters with dataset configurations.

        Args:
            project_parameters: The base project parameters.
            dataset: The dataset name.

        Returns:
            dict: Dataset-specific parameters.
        """
        c_parameters = deepcopy(project_parameters)
        c_parameters["dataset"] = project_parameters.get("dataset", {}).get(dataset, {})
        c_parameters["metrics"] = project_parameters.get("metrics", {}).get(dataset, {})
        return c_parameters

    @staticmethod
    def _merge_project_test_parameters(
        project_parameters: dict[str, Any], test_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merges project parameters with test parameters according to specific rules:
        1. Test inherits task_type from project
        2. Test inherits task_parameters from project, but test's own task_parameters take precedence
        3. Test inherits model and tokenizer parameters from project

        Args:
            project_parameters: The project's parameters
            test_parameters: The test's existing parameters

        Returns:
            dict: Merged parameters with test taking precedence where appropriate
        """
        merged_parameters = deepcopy(test_parameters)

        # 1. Override task_type with project's task_type
        merged_parameters["task_type"] = project_parameters.get("task_type", "")

        # 2. Merge task_parameters: project as base, test overrides duplicates
        project_task_params = project_parameters.get("task_parameters", {})
        test_task_params = merged_parameters.get("task_parameters", {})
        merged_parameters["task_parameters"] = {
            **project_task_params,
            **test_task_params,
        }

        # 3. Override model and tokenizer with project's values
        merged_parameters["model"] = project_parameters.get("model", {})
        merged_parameters["tokenizer"] = project_parameters.get("tokenizer", {})

        return merged_parameters

    @staticmethod
    def _create_default_test(project: Project) -> str:
        """
        Creates a default test for the given project.

        Args:
            project: The project for which to create a test.

        Returns:
            Id: The ID of the created test.
        """
        from app.models.test import Test

        test_body = Test(
            name="Datasets",
            description=f"Baseline dataset(s) for project {project.name}.",
            parameters=project.parameters,
            project_id=str(project.id),
            method_ids=[],  # This will be populated later
        )
        test_result = _get_tests_collection().insert_one(test_body.to_dict())
        if not test_result.acknowledged:
            logger.error(f"Failed to create default test for project {project.id}")
            raise RuntimeError("Failed to create default test in the database.")

        logger.debug(f"Created default test {test_body.id} for project {project.id}")
        return test_body.id

    @staticmethod
    def _duplicate_methods_LXT(
        project: Project,
        method_ids: list[str],
        test_id: str,
        auto_run: bool,
        reference_dataset_name: str,
        reference_dataset_params: dict[str, Any],
        refernce_task_params: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Duplicates existing LXT methods for the given project, overriding dataset information
        from the paired test's method.

        Args:
            project: The project for which to duplicate methods.
            method_ids: List of method IDs to duplicate.
            test_id: The ID of the test to associate the duplicated methods with.
            auto_run: Whether to automatically create and queue jobs for the duplicated methods.
            reference_dataset_name: The dataset name from the paired test's first method.
            reference_dataset_params: The dataset parameters from the paired test's first method.
            refernce_task_params: The task parameters from the paired test's first method.

        Returns:
            List[str]: List of IDs of the duplicated methods.
        """
        logger.info(
            f"Duplicating {len(method_ids)} LXT methods for test {test_id} in project {project.id}"
        )

        if not refernce_task_params:
            refernce_task_params = {}

        duplicated_method_ids = []
        for method_id in method_ids:
            try:
                # Fetch the original method
                original_method = _get_methods_collection().find_one({"id": method_id})
                if not original_method:
                    logger.warning(
                        f"Method with ID {method_id} not found for duplication"
                    )
                    continue

                # Merge project parameters with method parameters
                updated_parameters = ProjectController._merge_project_test_parameters(
                    project.parameters, original_method.get("parameters", {})
                )

                # Override dataset_name and parameters["dataset"] with values from paired test
                updated_parameters["dataset"] = reference_dataset_params

                # If task_type is "text-retrieval", override task_parameters with those from paired test
                if updated_parameters.get("task_type") == "text-retrieval":
                    updated_parameters["task_parameters"] = refernce_task_params

                # Create a duplicate with updated test_id and dataset information
                from app.models.method import Method

                duplicate_method = Method(
                    name=original_method.get(
                        "name", f"Duplicate of {original_method.get('name', 'Unknown')}"
                    ),
                    description=original_method.get(
                        "description",
                        f"Duplicate of {original_method.get('description', 'Unknown')}",
                    ),
                    dataset_name=reference_dataset_name,
                    diagnostics=original_method.get("diagnostics", []),
                    diagnostic_only=original_method.get("diagnostic_only", False),
                    metrics=original_method.get("metrics", []),
                    parameters=updated_parameters,
                    test_id=str(test_id),
                )

                method_result = _get_methods_collection().insert_one(
                    duplicate_method.to_dict()
                )
                if not method_result.acknowledged:
                    logger.error(f"Failed to create LXT method: {duplicate_method.id}")
                    raise RuntimeError("Failed to create LXT method in the database.")

                duplicated_method_ids.append(str(duplicate_method.id))
                logger.debug(
                    f"Created LXT method {duplicate_method.id} with dataset {reference_dataset_name}"
                )

                # If auto_run is enabled, create a job for the duplicated method
                if auto_run:
                    from app.models.job import Job

                    job = Job(
                        type=JobType.FULL,
                        status=JobStatus.PENDING,
                        parameters=updated_parameters,
                        method_id=str(duplicate_method.id),
                        result_id="",  # This will be populated after job execution
                    )
                    job_result = _get_jobs_collection().insert_one(job.to_dict())
                    if not job_result.acknowledged:
                        logger.error(
                            f"Failed to create job for LXT method {duplicate_method.id}"
                        )
                        continue

                    # Update method with job ID
                    _get_methods_collection().update_one(
                        {"id": duplicate_method.id},
                        {"$addToSet": {"job_ids": str(job.id)}},
                    )

                    # Add queue logic here
                    job_queue = get_job_queue()
                    if not job_queue.add_job(job):
                        logger.error(f"Failed to add job {job.id} to the queue.")
                        raise RuntimeError(f"Failed to add job {job.id} to the queue.")

            except Exception as e:
                logger.error(f"Error duplicating LXT method {method_id}: {e}")
                return []

        return duplicated_method_ids

    @staticmethod
    def _duplicate_methods(
        project: Project, method_ids: list[str], test_id: str, auto_run: bool
    ) -> list[str]:
        """
        Duplicates existing methods for the given project.

        Args:
            method_ids: List of method IDs to duplicate.
            test_id: The ID of the test to associate the duplicated methods with.

        Returns:
            List[str]: List of IDs of the duplicated methods.
        """

        logger.info(
            f"Duplicating {len(method_ids)} methods for test {test_id} in project {project.id}"
        )

        duplicated_method_ids = []
        for method_id in method_ids:
            try:
                # Fetch the original method
                original_method = _get_methods_collection().find_one({"id": method_id})
                if not original_method:
                    logger.warning(
                        f"Method with ID {method_id} not found for duplication"
                    )
                    continue

                logger.info(
                    f"DEBUG: original_method parameters={original_method.get('parameters', {})}"
                )
                logger.info(f"DEBUG: project parameters={project.parameters}")

                # Merge project parameters with method parameters
                updated_parameters = ProjectController._merge_project_test_parameters(
                    project.parameters, original_method.get("parameters", {})
                )

                logger.info(f"DEBUG: updated_parameters={updated_parameters}")

                # If task_type is "summarization", remove perplexity metric if present
                duplicate_metrics = original_method.get("metrics", [])
                if updated_parameters.get("task_type") == "summarization":
                    if "perplexity" in duplicate_metrics:
                        duplicate_metrics = [
                            m for m in duplicate_metrics if m != "perplexity"
                        ]

                    original_metrics = updated_parameters.get("metrics", [])
                    if "perplexity" in original_metrics:
                        updated_parameters["metrics"] = [
                            m for m in original_metrics if m != "perplexity"
                        ]

                # Create a duplicate with updated test_id
                from app.models.method import Method

                duplicate_method = Method(
                    name=original_method.get(
                        "name", f"Duplicate of {original_method.get('name', 'Unknown')}"
                    ),
                    description=original_method.get(
                        "description",
                        f"Duplicate of {original_method.get('description', 'Unknown')}",
                    ),
                    dataset_name=original_method.get("dataset_name", "Unknown"),
                    diagnostics=original_method.get("diagnostics", []),
                    diagnostic_only=original_method.get("diagnostic_only", False),
                    metrics=duplicate_metrics,
                    parameters=updated_parameters,
                    test_id=str(test_id),
                )

                method_result = _get_methods_collection().insert_one(
                    duplicate_method.to_dict()
                )
                if not method_result.acknowledged:
                    logger.error(f"Failed to create method: {duplicate_method.id}")
                    raise RuntimeError("Failed to create method in the database.")

                duplicated_method_ids.append(str(duplicate_method.id))

                # If auto_run is enabled, create a job for the duplicated method
                if auto_run:
                    from app.models.job import Job

                    job = Job(
                        type=JobType.FULL,
                        status=JobStatus.PENDING,
                        parameters=original_method.get("parameters", {}),
                        method_id=str(duplicate_method.id),
                        result_id="",  # This will be populated after job execution
                    )
                    job_result = _get_jobs_collection().insert_one(job.to_dict())
                    if not job_result.acknowledged:
                        logger.error(
                            f"Failed to create job for method {duplicate_method.id}"
                        )
                        continue

                    # Update method with job ID
                    _get_methods_collection().update_one(
                        {"id": duplicate_method.id},
                        {"$addToSet": {"job_ids": str(job.id)}},
                    )

                    # Add queue logic here
                    job_queue = get_job_queue()
                    if not job_queue.add_job(job):
                        logger.error(f"Failed to add job {job.id} to the queue.")
                        raise RuntimeError(f"Failed to add job {job.id} to the queue.")

            except Exception as e:
                logger.error(f"Error duplicating method {method_id}: {e}")
                return []

        return duplicated_method_ids

    @staticmethod
    def _duplicate_tests(
        test_ids: list[str],
        project: Project,
        auto_run: bool = False,
        group_id: str | None = None,
    ) -> list[str]:
        """
        Duplicates existing tests for the given project.

        Args:
            test_ids: List of test IDs to duplicate.
            project: The project for which to duplicate tests.
            auto_run: If True, automatically creates and queues jobs for the duplicated methods.

        Returns:
            List[str]: List of IDs of the duplicated tests.
        """
        duplicated_test_ids = []
        duplicated_method_ids = {}

        if group_id is None:
            group_id = str(ObjectId())

        logger.info(
            f"Duplicating {len(test_ids)} tests for project {project.id} with group ID {group_id}"
        )

        for test_id in test_ids:
            try:
                # Fetch the original test
                original_test = _get_tests_collection().find_one({"id": test_id})
                if not original_test:
                    logger.warning(f"Test with ID {test_id} not found for duplication")
                    continue

                # Merge project parameters with test parameters
                updated_parameters = ProjectController._merge_project_test_parameters(
                    project.parameters, original_test.get("parameters", {})
                )

                # Create a duplicate with updated project_id and name
                from app.models.test import Test

                duplicate_test = Test(
                    name=original_test.get(
                        "name", f"Duplicate of {original_test.get('name', 'Unknown')}"
                    ),
                    description=original_test.get(
                        "description",
                        f"Duplicate of {original_test.get('description', 'Unknown')}",
                    ),
                    default=False,
                    diagnostic_only=original_test.get("diagnostic_only", False),
                    parameters=updated_parameters,
                    project_id=str(project.id),
                    method_ids=[],
                    group=group_id
                    if len(test_ids) > 1
                    else "",  # Only set group if duplicating multiple tests
                    metadata=original_test.get("metadata", {}),
                )
                test_result = _get_tests_collection().insert_one(
                    duplicate_test.to_dict()
                )
                if not test_result.acknowledged:
                    logger.error(
                        f"Failed to create duplicate test for project {project.id}"
                    )
                    raise RuntimeError(
                        "Failed to create duplicate test in the database."
                    )

                # Create edited methods for LXT, SHAP, SHAP (Encoder) tests
                if duplicate_test.name in [
                    "LXT",
                    "SHAP",
                    "SHAP (Encoder)",
                    "Integrated Gradients (Encoder)",
                ]:
                    # Find the paired test (the other test in test_ids)
                    other_test_id = next(
                        (tid for tid in test_ids if tid != test_id), None
                    )
                    if not other_test_id:
                        logger.error(f"No paired test found for LXT test {test_id}")
                        raise RuntimeError(
                            "LXT test requires a paired test for dataset information."
                        )

                    # Fetch the paired test
                    other_test = _get_tests_collection().find_one({"id": other_test_id})
                    if not other_test:
                        logger.error(
                            f"Paired test with ID {other_test_id} not found for LXT test {test_id}"
                        )
                        raise RuntimeError(
                            f"Paired test with ID {other_test_id} not found."
                        )

                    # Get the first method from the paired test
                    other_method_ids = other_test.get("method_ids", [])
                    if not other_method_ids:
                        logger.error(
                            f"No methods found in paired test {other_test_id} for LXT test {test_id}"
                        )
                        raise RuntimeError(
                            f"No methods found in paired test {other_test_id}."
                        )

                    # Fetch the first method
                    reference_method = _get_methods_collection().find_one(
                        {"id": other_method_ids[0]}
                    )
                    if not reference_method:
                        logger.error(
                            f"Reference method {other_method_ids[0]} not found for LXT test {test_id}"
                        )
                        raise RuntimeError(
                            f"Reference method {other_method_ids[0]} not found."
                        )

                    # Extract dataset information from the reference method
                    reference_dataset_name = reference_method.get(
                        "dataset_name", "Unknown"
                    )
                    reference_dataset_params = reference_method.get(
                        "parameters", {}
                    ).get("dataset", {})
                    reference_task_params = reference_method.get("parameters", {}).get(
                        "task_parameters", {}
                    )

                    # Duplicate LXT methods with the reference dataset information
                    method_ids = ProjectController._duplicate_methods_LXT(
                        project,
                        original_test.get("method_ids", []),
                        str(duplicate_test.id),
                        auto_run,
                        reference_dataset_name,
                        reference_dataset_params,
                        refernce_task_params=reference_task_params
                        if project.parameters.get("task_type") == "text-retrieval"
                        else {},
                    )
                else:
                    # Create methods for the duplicated test
                    method_ids = ProjectController._duplicate_methods(
                        project,
                        original_test.get("method_ids", []),
                        str(duplicate_test.id),
                        auto_run,
                    )
                duplicated_method_ids[test_id] = method_ids

                # Update duplicated test with method ids
                _get_tests_collection().update_one(
                    {"id": duplicate_test.id}, {"$set": {"method_ids": method_ids}}
                )

                # Add to duplicated test ID
                duplicated_test_ids.append(duplicate_test.id)
                logger.debug(
                    f"Duplicated test {test_id} as {duplicate_test.id} for project {project.id}"
                )

            except Exception as e:
                logger.error(f"Error duplicating test {test_id}: {e}")
                return []

        return duplicated_test_ids, duplicated_method_ids

    @staticmethod
    def _create_methods_for_datasets(
        project: Project,
        datasets: list[str],
        metrics: dict[str, list[str]],
        test_id: str,
        auto_run: bool,
    ) -> list[str]:
        """
        Creates methods for each dataset in the project.

        Args:
            project: The project.
            datasets: The datasets to create methods for.
            metrics: The metrics to use for each dataset.
            test_id: The ID of the test these methods belong to.

        Returns:
            List[str]: List of created method IDs.
        """
        method_ids = []

        for dataset in datasets:
            c_parameters = ProjectController._create_dataset_parameters(
                project.parameters, dataset
            )

            from app.models.method import Method

            method_body = Method(
                name=dataset,
                description=f"Configuration for {dataset} in project {project.name}.",
                dataset_name=dataset,
                metrics=metrics.get(dataset, []),
                parameters=c_parameters,
                test_id=str(test_id),
                job_ids=[],  # This will be populated later
                result_ids=[],  # This will be populated later
            )
            method_result = _get_methods_collection().insert_one(method_body.to_dict())
            if not method_result.acknowledged:
                logger.error(f"Failed to create method for dataset {dataset}")
                continue

            method_ids.append(str(method_body.id))
            logger.debug(f"Created method {method_body.id} for dataset {dataset}")

            # If auto_run is enabled, create a job for the method
            if auto_run:
                from app.models.job import Job

                job = Job(
                    type=JobType.FULL,
                    status=JobStatus.PENDING,
                    parameters=c_parameters,
                    method_id=str(method_body.id),
                    result_id="",  # This will be populated after job execution
                )
                job_result = _get_jobs_collection().insert_one(job.to_dict())
                if not job_result.acknowledged:
                    logger.error(f"Failed to create job for method {method_body.id}")
                    continue

                # Update method with job ID
                _get_methods_collection().update_one(
                    {"id": method_body.id}, {"$addToSet": {"job_ids": str(job.id)}}
                )
                logger.debug(f"Created job {job.id} for method {method_body.id}")

                # Add queue logic here
                job_queue = get_job_queue()
                if not job_queue.add_job(job):
                    logger.error(f"Failed to add job {job.id} to the queue.")
                    raise RuntimeError(f"Failed to add job {job.id} to the queue.")

                logger.debug(
                    f"Added job {job.id} to the queue for method {method_body.id}"
                )

        logger.debug(f"Created {len(method_ids)} methods for test {test_id}")
        return method_ids

    @staticmethod
    def _update_project_with_test(project_id: str, test_id: str):
        """
        Updates the project with the newly created test ID.

        Args:
            project_id: The ID of the project to update.
            test_id: The ID of the test to add.
        """
        result = _get_projects_collection().update_one(
            {"id": project_id}, {"$addToSet": {"test_ids": str(test_id)}}
        )
        if result.modified_count == 0:
            logger.warning(
                f"No project was updated when adding test {test_id} to project {project_id}"
            )
        else:
            logger.debug(f"Added test {test_id} to project {project_id}")

    @staticmethod
    def _update_project_with_tests(project_id: str, test_ids: list[str]):
        """
        Updates the project with multiple test IDs.

        Args:
            project_id: The ID of the project to update.
            test_ids: List of test IDs to add to the project.
        """
        if not test_ids:
            logger.debug(f"No test IDs provided for project {project_id}")
            return

        result = _get_projects_collection().update_one(
            {"id": project_id}, {"$addToSet": {"test_ids": {"$each": test_ids}}}
        )
        if result.modified_count == 0:
            logger.warning(
                f"No project was updated when adding tests {test_ids} to project {project_id}"
            )
        else:
            logger.debug(f"Added {len(test_ids)} tests to project {project_id}")

    @staticmethod
    def _update_test_with_methods(test_id: str, method_ids: list[str]):
        """
        Updates the test with the newly created method IDs.

        Args:
            test_id: The ID of the test to update.
            method_ids: List of method IDs to add.
        """
        result = _get_tests_collection().update_one(
            {"id": test_id}, {"$set": {"method_ids": method_ids}}
        )
        if result.modified_count == 0:
            logger.warning(
                f"No test was updated when adding methods {method_ids} to test {test_id}"
            )
        else:
            logger.debug(f"Added {len(method_ids)} methods to test {test_id}")

    @staticmethod
    def duplicate_tests_and_methods(
        test_ids: list[str],
        project: Project,
        auto_run: bool,
        group_id: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Duplicates existing tests and their methods for the given project.

        Args:
            test_ids: List of test IDs to duplicate.
            project: The project for which to duplicate tests.

        Returns:
            dict[str, List[str]]: A dictionary containing the duplicated test IDs and method IDs.
        """
        duplicated_test_ids, _ = ProjectController._duplicate_tests(
            test_ids, project, auto_run=auto_run, group_id=group_id
        )

        # Update project with test_ids
        _get_projects_collection().update_one(
            {"id": project.id},
            {"$addToSet": {"test_ids": {"$each": duplicated_test_ids}}},
        )

        # Retrieve updated project
        updated_project = _get_projects_collection().find_one({"id": project.id})
        if not updated_project:
            logger.warning(
                f"Project {project.id} not found after updating with test IDs"
            )
            return {}

        # Get tests
        tests = _get_tests_collection().find({"id": {"$in": duplicated_test_ids}})
        from app.models.test import Test

        test_dict = [Test(**test).to_dict(include={"methods"}) for test in tests]

        return test_dict

    @staticmethod
    def create_project_with_tests_and_methods(
        project: Project,
        datasets: list[str],
        metrics: dict[str, list[str]],
        auto_run: bool,
    ) -> dict[str, Any]:
        """
        Creates a new project in the database along with associated tests and methods.

        This method orchestrates the creation of:
        1. A new project
        2. A default test for the project
        3. Methods for each dataset in the project
        4. Updates relationships between all entities

        Args:
            project (Project): The project data to create.
            auto_run (bool): If True, automatically runs the created methods.

        Returns:
            dict: The created project serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the project in the database.
        """
        try:
            with ProjectController.database_transaction():
                # Step 1: Create the project
                result = _get_projects_collection().insert_one(
                    {**project.to_dict(), "test_ids": []}
                )

                if not result.acknowledged:
                    logger.error(f"Failed to create project {project.id}")
                    raise RuntimeError("Failed to create project")

                # Step 2: Create default test
                test_id = ProjectController._create_default_test(project)

                # Step 3: Duplicate existing tests from Project if any and combine with new test ID
                all_test_ids = [
                    str(test_id)
                ]  # Start with the newly created default test

                if project.test_ids:
                    duplicated_test_ids, _ = ProjectController._duplicate_tests(
                        project.test_ids, project, auto_run
                    )
                    all_test_ids.extend([str(tid) for tid in duplicated_test_ids])

                ProjectController._update_project_with_tests(
                    str(project.id), all_test_ids
                )
                total_tests = len(all_test_ids)
                logger.debug(
                    f"Added {total_tests} test IDs (1 new default + {len(project.test_ids) if project.test_ids else 0} duplicated) to project {project.id}"
                )

                # Step 4: Create methods for each dataset
                method_ids = ProjectController._create_methods_for_datasets(
                    project, datasets, metrics, test_id, auto_run
                )

                # Step 5: Update test with method IDs
                ProjectController._update_test_with_methods(test_id, method_ids)

                # Step 6: Cache dataset if needed
                for dataset in datasets:
                    split = (
                        project.parameters.get("dataset", {})
                        .get(dataset, {})
                        .get("split", "validation")
                    )
                    try:
                        dataset_loader = get_dataset_loader()
                        dataset_loader.load(
                            dataset_name=dataset, split=split, cache=True
                        )
                        logger.debug(
                            f"Cached dataset {dataset} for project {project.id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to cache dataset {dataset} for project {project.id}: {e}"
                        )

                # Step 7: Load model & tokenizer for the project (asynchronously, non-blocking) if not auto_run
                if not auto_run:
                    model_name = project.model_name
                    model_parameters = project.parameters.get("model", {}).get(
                        "from_pretrained", {}
                    )
                    tokenizer_parameters = project.parameters.get("tokenizer", {}).get(
                        "from_pretrained", {}
                    )

                    try:

                        def load_model_async():
                            try:
                                get_model_loader().load(
                                    model_name, model_parameters, tokenizer_parameters
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Background model loading failed for project {project.id}: {e}"
                                )

                        # Start loading in background thread
                        thread = threading.Thread(target=load_model_async, daemon=True)
                        thread.start()

                    except Exception as e:
                        logger.warning(
                            f"Failed to start background model loading for project {project.id}: {e}"
                        )

                updated_project = ProjectController.get_project_by_id(
                    str(project.id), include={"tests", "tests.methods"}
                )
                if not updated_project:
                    logger.error(
                        f"Project with ID {project.id} not found after creation."
                    )
                    raise RuntimeError(
                        f"Project with ID {project.id} not found after creation."
                    )

                return {"project": updated_project}
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise RuntimeError(f"Failed to create project: {e}")

    @staticmethod
    def update_project(project_id: str, project: ProjectUpdate) -> dict[str, Any]:
        """
        Updates an existing project in the database.

        Args:
            project_id (str): The ID of the project to update.
            project (ProjectUpdate): The updated project data.

        Returns:
            dict: The updated project serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the project in the database.
        """
        try:
            with ProjectController.database_transaction():
                # Fetch the existing project
                current_project = _get_projects_collection().find_one(
                    {"id": project_id}
                )
                if not current_project:
                    logger.warning(
                        f"Project with ID {project_id} not found for update."
                    )
                    raise ValueError(f"Project with ID {project_id} not found.")

                # Convert to Project model for type consistency
                current_project = Project(**current_project)

                # Update the project's test_ids if they have changed
                if project.test_ids and project.test_ids != current_project.test_ids:
                    # Remove old associations
                    old_test_ids = set(current_project.test_ids) - set(project.test_ids)
                    for old_test_id in old_test_ids:
                        _get_tests_collection().update_one(
                            {"id": old_test_id}, {"$unset": {"project_id": ""}}
                        )

                    for test_id in project.test_ids:
                        _get_tests_collection().update_one(
                            {"id": test_id}, {"$set": {"project_id": str(project_id)}}
                        )

                # Update the project
                result = _get_projects_collection().update_one(
                    {"id": project_id}, {"$set": project.to_dict()}
                )
                if result.matched_count == 0:
                    logger.error(f"Project with ID {project_id} not found.")
                    raise ValueError(f"Project with ID {project_id} not found.")
                updated_project = _get_projects_collection().find_one(
                    {"id": project_id}
                )

                return Project(**updated_project).to_dict()
        except Exception as e:
            logger.error(f"Error updating project with ID {project_id}: {e}")
            raise RuntimeError(f"Failed to update project with ID {project_id}: {e}")

    @staticmethod
    def delete_project(project_id: str):
        """
        Deletes a project from the database.

        Args:
            project_id (str): The ID of the project to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the project from the database.
        """
        try:
            with ProjectController.database_transaction():
                # Fetch the project to get its test_ids
                project = _get_projects_collection().find_one({"id": project_id})
                if not project:
                    logger.error(f"Project with ID {project_id} not found.")
                    raise ValueError(f"Project with ID {project_id} not found.")

                # Convert project to Project model for type consistency
                project_obj = Project(**project)

                # Remove project_id from the tests' project_ids array
                for test_id in project_obj.test_ids:
                    _get_tests_collection().update_one(
                        {"id": test_id}, {"$pull": {"project_ids": str(project_id)}}
                    )

                # Delete the project
                result = _get_projects_collection().delete_one({"id": project_id})
                if result.deleted_count == 0:
                    raise ValueError(f"Project with ID {project_id} not found.")

                return {
                    "message": f"Project with ID {project_id} deleted successfully."
                }
        except Exception as e:
            logger.error(f"Error deleting project with ID {project_id}: {e}")
            raise RuntimeError(f"Failed to delete project with ID {project_id}: {e}")
