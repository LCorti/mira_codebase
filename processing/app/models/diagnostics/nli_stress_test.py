import json
from typing import Any

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")

DATA_PATH = "app/data/NLI_StressTest/stress_test.jsonl"

COMPETENCE_TESTS = ["antonym", "quant"]
DISTRACTION_TESTS = ["negation", "wordoverlap", "length_mismatch"]
NOISE_TESTS = ["dev_gram_contentword_swap_perturbed"]

JOB_ID = ""
JOB_QUEUE = None


class NLIStressTestDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the NLI stress test diagnostic .
        """
        super().__init__(
            name="nli_stress_test",
            description="Diagnostic for evaluating NLI stress tests.",
        )

    def generate_report(
        self, results_dict, competence_keys, distraction_keys, noise_keys
    ):
        """
        :param results_dict: model predictions on tests
        :param competence_keys: competence tests to be reported, includes antonymy and numerical reasoning
        :param distraction_keys: distraction tests to be reported, includes word overlap, negaion and length mismatch
        :param noise_keys: noise tests to be reported, includes content word perturbaion (Appendix C)
        """

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        report = "===================== COMPETENCE TESTS ======================\n"

        for each_key in competence_keys:
            acc = (
                results_dict[each_key]["accuracy"] if each_key in results_dict else 0.0
            )
            report += f"{each_key.upper()}: {acc!s}\n"

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        report += "\n===================== DISTRACTION TESTS ======================\n"

        for each_key in distraction_keys:
            acc = (
                results_dict[each_key]["accuracy"] if each_key in results_dict else 0.0
            )
            report += f"{each_key.upper()}: {acc!s}\n"

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        report += "\n===================== NOISE TESTS ===========================\n"

        for each_key in noise_keys:
            acc = (
                results_dict[each_key]["accuracy"] if each_key in results_dict else 0.0
            )
            report += f"{each_key.upper()}: {acc!s}\n"

        self._check_job_cancellation(JOB_ID, JOB_QUEUE)

        return report

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the script from the "Stress Test Evaluation for Natural Language Inference" paper (https://aclanthology.org/C18-1198/).

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        # Validate inputs
        if not parameters:
            parameters = {}
        if not results:
            raise ValueError("No results provided for diagnostic computation.")

        global JOB_ID, JOB_QUEUE
        JOB_ID = kwargs.get("job_id", "")
        JOB_QUEUE = kwargs.get("job_queue", "")

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        predictions = [res.get("prediction", []) for res in results]

        try:
            # Add prediction to data
            with open(DATA_PATH, "r") as inp:
                data = inp.readlines()
        except FileNotFoundError:
            logger.error(
                f"NLI Stress Test data file not found at {DATA_PATH}. Please ensure the file exists."
            )
            raise FileNotFoundError(
                f"Data file not found at {DATA_PATH}. Please ensure the file exists."
            )

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        # Merge predictions with data
        results_dict = {}
        for line, prediction in zip(data, predictions):
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )

            obj = json.loads(line)

            # Add prediction to object
            obj["prediction"] = prediction[0] if prediction else ""

            # Default dict entry
            if obj["source"] not in results_dict:
                results_dict[obj["source"]] = {
                    "total": 0.0,
                    "correct": 0.0,
                    "accuracy": 0.0,
                }

            # Count correct predictions
            if "prediction" in obj and obj["prediction"] == obj["gold_label"]:
                results_dict[obj["source"]]["correct"] += 1
            results_dict[obj["source"]]["total"] += 1

        for each_key, entry in results_dict.items():
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )
            entry["accuracy"] = round(
                entry["correct"] * 100.0 / entry["total"],
                2,
            )

        keys = sorted(results_dict.keys())

        competence_keys = []
        for each_test in COMPETENCE_TESTS:
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )
            for each_key in keys:
                if each_key.startswith(each_test):
                    competence_keys.append(each_key)

        distraction_keys = []
        for each_test in DISTRACTION_TESTS:
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )
            for each_key in keys:
                if each_key.startswith(each_test):
                    distraction_keys.append(each_key)

        noise_keys = []
        for each_test in NOISE_TESTS:
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )
            for each_key in keys:
                if each_key.startswith(each_test):
                    noise_keys.append(each_key)

        # Generate and return report
        return self.generate_report(
            results_dict, competence_keys, distraction_keys, noise_keys
        )
