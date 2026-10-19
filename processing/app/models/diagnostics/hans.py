from collections import defaultdict
from typing import Any

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")

HANS_DATA = {
    "train": "app/data/HANS/heuristics_train_set.txt",
    "test": "app/data/HANS/heuristics_evaluation_set.txt",
}


class HansDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the HANS diagnostic with predefined tests and their corresponding processing functions.
        """
        super().__init__(
            name="hans",
            description="Diagnostic for evaluating heuristic use from the HANS method.",
        )
        self.samples = {}  # Store sample data by ID
        self.categories = {"heuristics": set(), "subcases": set(), "templates": set()}
        self.counts = defaultdict(lambda: defaultdict(int))

    def _count_results(self, guess_dict: dict):
        """Count correct and incorrect predictions by category."""
        for sample_id, sample_data in self.samples.items():
            if sample_id not in guess_dict:
                continue

            prediction = guess_dict[sample_id]
            is_correct = prediction == sample_data["gold_label"]
            is_entailment = sample_data["gold_label"] == "entailment"

            # Count by heuristic and entailment type
            heuristic_key = (
                f"{sample_data['heuristic']}_{'ent' if is_entailment else 'nonent'}"
            )
            self.counts[heuristic_key]["correct" if is_correct else "incorrect"] += 1

            # Count by subcase and template
            self.counts[f"subcase_{sample_data['subcase']}"][
                "correct" if is_correct else "incorrect"
            ] += 1
            self.counts[f"template_{sample_data['template']}"][
                "correct" if is_correct else "incorrect"
            ] += 1

    def evaluate_heuristic_output(self, guess_dict: dict):
        """Generate evaluation report based on predictions."""
        self._count_results(guess_dict)
        report = ""

        # Heuristic entailed results
        report += "Heuristic entailed results:\n"
        for heuristic in sorted(self.categories["heuristics"]):
            key = f"{heuristic}_ent"
            correct = self.counts[key]["correct"]
            incorrect = self.counts[key]["incorrect"]
            total = correct + incorrect
            percent = correct / total if total > 0 else 0
            report += f"{heuristic}: {percent}\n"

        # Heuristic non-entailed results
        report += "Heuristic non-entailed results:\n"
        for heuristic in sorted(self.categories["heuristics"]):
            key = f"{heuristic}_nonent"
            correct = self.counts[key]["correct"]
            incorrect = self.counts[key]["incorrect"]
            total = correct + incorrect
            percent = correct / total if total > 0 else 0
            report += f"{heuristic}: {percent}\n"

        # Subcase results
        report += "\nSubcase results:\n"
        for subcase in sorted(self.categories["subcases"]):
            key = f"subcase_{subcase}"
            correct = self.counts[key]["correct"]
            incorrect = self.counts[key]["incorrect"]
            total = correct + incorrect
            percent = correct / total if total > 0 else 0
            report += f"{subcase}: {percent}\n"

        # Template results
        report += "\nTemplate results:\n"
        for template in sorted(self.categories["templates"]):
            key = f"template_{template}"
            correct = self.counts[key]["correct"]
            incorrect = self.counts[key]["incorrect"]
            total = correct + incorrect
            percent = correct / total if total > 0 else 0
            report += f"{template}: {percent}\n"

        return report

    def process(self, split: str):
        """Load and process HANS dataset file."""
        file_path = HANS_DATA.get(split)
        if not file_path:
            raise ValueError(
                f"Invalid split '{split}'. Valid options are: {list(HANS_DATA.keys())}"
            )

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            logger.error(
                f"HANS data file not found at path: {file_path}. Please ensure the file exists."
            )
            raise FileNotFoundError(
                f"HANS data file not found at path: {file_path}. Please ensure the file exists."
            )

        if not lines:
            return

        # Parse header
        headers = lines[0].strip().split("\t")
        id_index = headers.index("pairID")

        # Process data lines
        for line in lines[1:]:
            parts = line.strip().split("\t")
            sample_id = parts[id_index]

            # Build sample data dictionary
            sample_data = {}
            for index, header in enumerate(headers):
                if header != "pairID":
                    sample_data[header] = parts[index]

            self.samples[sample_id] = sample_data

            # Track unique categories
            self.categories["heuristics"].add(sample_data["heuristic"])
            self.categories["subcases"].add(sample_data["subcase"])
            self.categories["templates"].add(sample_data["template"])

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the script from the "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference" paper.

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        # Validate inputs
        if not parameters:
            parameters = {}
        if not results:
            raise ValueError("No results provided for diagnostic computation.")
        if not parameters.get("dataset"):
            raise ValueError("Parameter 'dataset' is required.")

        predictions = [res.get("prediction", []) for res in results]

        # Load dataset
        from app.config.registry import get_dataset_loader

        try:
            dataset = get_dataset_loader().load("dtk-llms/hans", cache=False)
        except Exception as e:
            logger.error(f"Failed to load HANS dataset: {e}")
            raise ValueError(f"Failed to load HANS dataset: {e}")

        split = parameters.get("dataset", {}).get("split", "train")

        # Map predictions to their corresponding sample IDs
        guess_dict = {}
        for pred, sample in zip(predictions, dataset[split]):
            guess_dict[sample["pairID"]] = pred[0] if pred else ""

        # Process the dataset and generate report
        self.process(split=split)

        # Build report
        report_content = f"---\nSplit: {split}\n---\n"
        report_content += self.evaluate_heuristic_output(guess_dict)

        return report_content
