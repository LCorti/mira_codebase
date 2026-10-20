from enum import Enum

from pydantic import Field

from app.config.registry import (
    get_database_service,
    get_dataset_loader,
    get_model_loader,
)
from app.controllers.logit_controller import LogitController
from app.controllers.result_controller import ResultController
from app.models.base_model import BaseMongoModel
from app.services.logging_service import LoggingService
from app.services.model_inference import run_analysis, run_diagnostics, run_inference

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")


class JobCancelledException(Exception):
    """Exception raised when a job is cancelled during execution."""

    pass  # noqa


def _get_methods_collection():
    """
    Returns the methods collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("methods")


def _get_results_collection():
    """
    Returns the results collection from the database service.
    """
    db_service = get_database_service()
    return db_service.get_collection("results")


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, value: str) -> "JobStatus":
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


class JobType(str, Enum):
    FULL = "FULL"
    INFERENCE = "INFERENCE"
    ANALYSIS = "ANALYSIS"
    DIAGNOSIS = "DIAGNOSIS"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, value: str) -> "JobType":
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


class Job(BaseMongoModel):
    type: JobType = Field(default=JobType.FULL)
    status: JobStatus = Field(default=JobStatus.PENDING)
    status_message: str = Field(default="")
    runtime: float = Field(default=0.0, ge=0)
    estimated_runtime: float = Field(default=0.0, ge=0)
    parameters: dict = Field(
        default={
            "task_type": "",
            "task_parameters": {},
            "model": {"from_pretrained": {}, "inference": {}},
            "tokenizer": {"from_pretrained": {}, "inference": {}},
            "dataset": {
                "split": "train",
                "input_column": "input",
                "reference_column": "reference",
                "num_samples": 15,
            },
            "metrics": {"METRIC_NAME": {}},
        }
    )

    method_id: str = Field(
        default="", description="ID of the method associated with this job"
    )
    result_id: str = Field(
        default="", description="ID of the result associated with this job"
    )
    tag_ids: list[str] = Field(
        default=[], description="List of tag IDs associated with this job"
    )

    def method(self):
        """
        Retrieve the method associated with this job.

        Returns:
            Method: The method object associated with this job.
        """
        if not self.method_id:
            return None

        from app.models.method import Method

        try:
            method = _get_methods_collection().find_one({"id": self.method_id})
            logger.debug(f"Retrieved method with ID {self.method_id} from database.")

            if not method:
                raise ValueError(f"Method with ID {self.method_id} not found.")

            return Method(**method)
        except Exception as e:
            logger.error(f"Error retrieving method with ID {self.method_id}: {e}")
            raise RuntimeError(f"Failed to retrieve method: {e}")

    def result(self):
        """
        Retrieve the result associated with this job.

        Returns:
            Result: The result object associated with this job.
        """
        if not self.result_id:
            return None

        from app.models.result import Result

        try:
            result = _get_results_collection().find_one({"id": self.result_id})

            if not result:
                raise ValueError(f"Result with ID {self.result_id} not found.")

            return Result(**result)
        except Exception as e:
            logger.error(f"Error retrieving result with ID {self.result_id}: {e}")
            return {}

    def tags(self):
        """
        Retrieve the tags associated with this job.

        Returns:
            List[Tag]: List of tag objects associated with this job.
        """
        from app.controllers.tag_controller import TagController
        from app.models.tag import Tag

        tags = []
        for tag_id in self.tag_ids:
            try:
                tag_data = TagController.get_tag_by_id(tag_id)
                if tag_data:
                    tags.append(Tag(**tag_data))
            except Exception as e:
                logger.error(f"Error retrieving tag with ID {tag_id}: {e}")

        return tags

    def to_dict(self, include: set[str] | None = None) -> dict:
        """
        Convert model to dictionary, lazily including only requested relations.

        Args:
            include: Set of relations to include (e.g., {'result'})
        """
        include = include or set()
        data = self.model_dump()

        # Only fetch and include result if explicitly requested
        if "result" in include:
            result = self.result()
            if result:
                data["result"] = result.to_dict()
                data.pop("result_id", None)

        if "log" in include:
            data["log"] = self.get_logs()

        if "tags" in include:
            data["tags"] = [tag.to_dict() for tag in self.tags()]

        return data

    def execute(self, job_queue=None):
        """
        Execute the job based on its type:
        - INFERENCE: Only run inference and create a result, skip analysis
        - ANALYSIS: Skip inference and grab results from associated result_id, then run analysis
        - FULL: Do both inference and analysis (normal behavior)

        Args:
            job_queue: Optional JobQueue instance for checking cancellation status

        Returns:
            self: The job instance with updated results.
        Raises:
            JobCancelledException: If the job was cancelled during execution
            RuntimeError: If there is an error loading the model, tokenizer, dataset, or if inference/analysis fails.
        """

        def check_cancellation():
            """Helper function to check if job should be cancelled."""
            if job_queue and job_queue.is_job_cancelled(self.id):
                logger.info(f"{self.id}: Job execution cancelled")
                raise JobCancelledException(f"Job {self.id} was cancelled")

        method = self.method()

        # Check cancellation before starting
        check_cancellation()

        # Create result for this job
        result_object = None
        diagnostic_only = self.parameters.get("task_parameters", {}).get(
            "diagnostic_only", False
        )
        diagnostic_type = self.parameters.get("task_parameters", {}).get(
            "diagnostic_type", "lxt"
        )

        if diagnostic_only:
            old_result = ResultController.get_result_by_id(self.result_id)
            from app.models.result import Result

            new_result = ResultController.create_result(
                Result(
                    diagnostics=old_result["diagnostics"],
                    summary=old_result["summary"],
                    results=old_result["results"],
                    parameters=self.parameters,
                    method_id=self.method_id,
                    job_id=self.id,
                )
            )
            if not new_result:
                logger.error("Failed to create result object for diagnostic-only job.")
                raise RuntimeError(
                    "Failed to create result object for diagnostic-only job."
                )
            self.result_id = new_result.get("id", "")
            result_object = Result(**new_result)
        elif (
            self.type in [JobType.INFERENCE, JobType.FULL, JobType.DIAGNOSIS]
            and not diagnostic_only
        ):
            from app.models.result import Result

            result_object = Result(
                parameters=self.parameters,
                method_id=self.method_id,
                job_id=self.id,
            )

            try:
                result_response = ResultController.create_result(result_object)
                if not result_response:
                    logger.error("Failed to create result object.")
                    raise RuntimeError("Failed to create result object.")
            except Exception as e:
                logger.error(f"Failed to create result object: {e}")
                raise RuntimeError(f"Failed to create result object: {e}")

        # Handle different job types
        if self.type in [JobType.INFERENCE, JobType.FULL]:
            # Run inference for INFERENCE and FULL job types
            dataset_name = method.dataset_name
            project = method.test().project()
            model_name = project.model_name
            logger.info(
                f"{self.id}: Running inference with model {model_name} and dataset {dataset_name}."
            )

            model_parameters = self.parameters.get("model", {}).get(
                "from_pretrained", {}
            )
            tokenizer_parameters = self.parameters.get("tokenizer", {}).get(
                "from_pretrained", {}
            )

            # Check cancellation before loading model
            check_cancellation()

            # Load model, tokenizer, and dataset
            try:
                model, tokenizer = get_model_loader().load(
                    model_name, model_parameters, tokenizer_parameters
                )
                dataset = get_dataset_loader().load(dataset_name)

                if not model or not tokenizer or not dataset:
                    raise ValueError("Model, tokenizer, or dataset not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to load model, tokenizer, or dataset: {e}")

            # Check cancellation before inference
            check_cancellation()

            # Run inference & save results
            try:
                # Run inference using the model, tokenizer, and dataset
                results, logits = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    parameters=self.parameters,
                    job_id=self.id,
                    job_queue=job_queue,  # Pass job_queue for cancellation checking
                )

                # Check cancellation after inference
                check_cancellation()

                for result, logit in zip(results, logits):
                    from app.models.logit import Logit

                    logit_object = Logit(
                        logit_str=logit,
                    )
                    try:
                        logit_response = LogitController.create_logit(logit_object)
                        if not logit_response:
                            logger.error("Failed to create logit object.")
                    except Exception as e:
                        logger.error(f"Failed to create logit object: {e}")

                    result["logit_id"] = (
                        logit_response.get("id") if logit_response else ""
                    )

                check_cancellation()

                result_object.results = (
                    results  # Update result object with inference results
                )

                try:
                    from app.models.result import ResultUpdate

                    result_response = ResultController.update_result(
                        result_object.id, ResultUpdate(**result_object.to_dict())
                    )
                    if not result_response:
                        logger.error("Failed to create result object.")
                        raise RuntimeError("Failed to create result object.")
                except Exception as e:
                    logger.error(f"Failed to create result object: {e}")
                    raise RuntimeError(f"Failed to create result object: {e}")

                logger.info(
                    f"{self.id}: Inference completed, {len(results)} results saved."
                )
            except JobCancelledException:
                raise  # Re-raise cancellation exception
            except Exception as e:
                raise RuntimeError(f"Failed to run inference: {e}")

        # Check cancellation before analysis
        check_cancellation()

        if self.type in [JobType.ANALYSIS]:
            # For ANALYSIS jobs, load existing result
            if not self.result_id:
                logger.error("result_id must be set for ANALYSIS jobs.")
                raise ValueError("result_id must be set for ANALYSIS jobs.")

            result_object = self.result()
            if not result_object:
                logger.error(
                    f"Result with ID {self.result_id} not found for ANALYSIS job."
                )
                raise ValueError(
                    f"Result with ID {self.result_id} not found for ANALYSIS job."
                )
            logger.info(
                f"{self.id}: Loaded existing result {self.result_id} for analysis."
            )

        # Run analysis for ANALYSIS and FULL job types
        if self.type in [JobType.ANALYSIS, JobType.FULL] and method.metrics:
            try:
                logger.info(
                    f"{self.id}: Running analysis using metrics: {method.metrics}."
                )

                # Analyze results using the method's metrics
                analyzed_result_object = run_analysis(
                    metrics=method.metrics,
                    result=result_object,
                    parameters=self.parameters,
                    job_id=self.id,
                    job_queue=job_queue,  # Pass job_queue for cancellation checking
                )

                # Check cancellation after analysis
                check_cancellation()

                # Update result object with analyzed_results
                summary = (
                    analyzed_result_object.summary
                    if analyzed_result_object.summary
                    else {}
                )
                results = (
                    analyzed_result_object.results
                    if analyzed_result_object.results
                    else {}
                )

                try:
                    _ = ResultController.update_result_attr(
                        result_object.id, summary, results
                    )
                except Exception as e:
                    logger.error(f"Failed to update result object with analysis: {e}")
                    raise RuntimeError(
                        f"Failed to update result object with analysis: {e}"
                    )

                logger.info(f"{self.id}: Analysis completed and results updated.")
            except JobCancelledException:
                raise  # Re-raise cancellation exception
            except Exception as e:
                logger.error(f"{self.id}: Failed to run analysis: {e}")
                raise RuntimeError(f"Failed to run analysis: {e}")

        # Run diagnostics for ANALYSIS and FULL job types
        if self.type in [JobType.DIAGNOSIS, JobType.ANALYSIS, JobType.FULL] and (
            method.diagnostics or diagnostic_only
        ):
            try:
                logger.info(
                    f"{self.id}: Running diagnostics using: {method.diagnostics}."
                ) if not diagnostic_only else logger.info(
                    f"{self.id}: Running {diagnostic_type} diagnostics."
                )

                model_name = self.method().test().project().model_name

                # Analyze results using the method's diagnostics
                diagnostics_result_object = run_diagnostics(
                    diagnostics=method.diagnostics,
                    result=result_object,
                    parameters=self.parameters,
                    job_id=self.id,
                    job_queue=job_queue,  # Pass job_queue for cancellation checking
                    model_name=model_name,
                )

                # Check cancellation before diagnostics
                check_cancellation()

                # Update result object with diagnostics
                result_diagnostics = (
                    diagnostics_result_object.diagnostics
                    if diagnostics_result_object.diagnostics
                    else {}
                )

                try:
                    _ = ResultController.update_diagnostics_attr(
                        result_object.id, result_diagnostics
                    )
                except Exception as e:
                    logger.error(
                        f"{self.id}: Failed to update diagnostics in result: {e}"
                    )
                    raise RuntimeError(f"Failed to update diagnostics in result: {e}")

                logger.info(f"{self.id}: Diagnostics completed, results updated.")
            except JobCancelledException:
                raise  # Re-raise cancellation exception
            except Exception as e:
                logger.error(f"{self.id}: Failed to run diagnostics: {e}")
                raise RuntimeError(f"Failed to run diagnostics: {e}")

        logger.info(f"{self.id}: Finished job.")
        return self

    def get_logs(self, include_timestamp: bool = True) -> list[dict[str, str]]:
        """
        Retrieve all log lines associated with this job ID.

        Args:
            include_timestamp: Whether to include timestamp information in results

        Returns:
            List of dictionaries containing log information for this job

        Example:
            job = Job(...)
            logs = job.get_logs()
            for log in logs:
                print(f"[{log['level']}] {log['message']}")
        """
        return LoggingService.get_job_logs(self.id, include_timestamp=include_timestamp)

    def get_logs_summary(self) -> dict[str, any]:
        """
        Get a summary of log entries for this job.

        Returns:
            Dictionary containing summary information about the job's logs
        """
        return LoggingService.get_job_logs_summary(self.id)


class JobUpdate(BaseMongoModel):
    type: JobType | None = None
    status: JobStatus | None = None
    status_message: str | None = None
    runtime: float | None = None
    estimated_runtime: float | None = None
    method_id: str | None = None
    result_id: str | None = None
