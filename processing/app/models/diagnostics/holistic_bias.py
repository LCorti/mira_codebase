from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")

MANN_WHITNEY_U_P_THRESHOLD = 0.05

JOB_ID = ""


class HolisticBiasDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the Holistic Bias diagnostic with predefined tests and their corresponding processing functions.
        """
        super().__init__(
            name="holistic_bias",
            description="Diagnostic for evaluating heuristic use from the Holistic Bias method.",
        )
        self.sentences = []
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _calculate_perplexity(self, texts: list[str]) -> list[float]:
        """
        Calculate perplexity for a batch of texts using the loaded model.
        """
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # Calculate per-sample loss (instead of averaged loss)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            # Flatten for loss calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Calculate loss per token, then average per sequence
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else -100
            )
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=pad_token_id, reduction="none"
            )
            token_losses = loss_fct(shift_logits, shift_labels)

            # Reshape back to (batch_size, seq_len-1)
            token_losses = token_losses.view(inputs["input_ids"].size(0), -1)

            # Calculate mean loss per sequence, ignoring padded tokens
            attention_mask = inputs["attention_mask"][..., 1:]  # Shift attention mask
            seq_losses = (token_losses * attention_mask).sum(
                dim=1
            ) / attention_mask.sum(dim=1)

            # Convert to perplexities
            perplexities = torch.exp(seq_losses).cpu().tolist()

        return perplexities

    def evaluate_sentences(
        self, batch_size: int = 16, job_id: str = "", job_queue=None
    ) -> list[tuple]:
        """
        Evaluate perplexity for all sentences.

        Returns:
            A nested dictionary of perplexities indexed by axis, template, and descriptor.
        """
        binned_perplexities = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for i in range(0, len(self.sentences), batch_size):
            batch = self.sentences[i : i + batch_size]

            self._check_job_cancellation(job_id, job_queue)

            texts = [sentence_metadata["text"] for sentence_metadata in batch]
            perplexities = self._calculate_perplexity(texts)

            for sentence_metadata, ppl in zip(batch, perplexities):
                axis = sentence_metadata["axis"]
                template = sentence_metadata["template"]
                descriptor = sentence_metadata["descriptor"]
                binned_perplexities[axis][template][descriptor].append(ppl)

        return binned_perplexities

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the script from the ""I'm sorry to hear that": Finding New Biases in Language Models with a Holistic Descriptor Dataset" paper by Smith et al (2022).

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        # Validate inputs
        if not parameters:
            parameters = {}
        if not parameters.get("dataset"):
            logger.error("Parameter 'dataset' is required.")
            raise ValueError("Parameter 'dataset' is required.")
        if not parameters.get("model_name"):
            logger.error("Parameter 'model_name' is required.")
            raise ValueError("Parameter 'model_name' is required.")
        if not parameters.get("model_parameters"):
            logger.error("Parameter 'model_parameters' is required.")
            raise ValueError("Parameter 'model_parameters' is required.")
        if not parameters.get("tokenizer_parameters"):
            logger.error("Parameter 'tokenizer_parameters' is required.")
            raise ValueError("Parameter 'tokenizer_parameters' is required.")

        global JOB_ID
        JOB_ID = kwargs.get("job_id", "")

        # Load dataset
        from app.config.registry import get_dataset_loader

        try:
            dataset = get_dataset_loader().load("dtk-llms/HolisticBias", cache=False)
        except Exception as e:
            logger.error(f"Failed to load Holistic BIAS dataset: {e}")
            raise ValueError(f"Failed to load BIAS dataset: {e}")

        split = parameters.get("dataset", {}).get("split", "test")

        sample_list = []
        for item in dataset[split]:
            sample_list.append(
                {
                    "text": item["text"],
                    "axis": item["axis"],
                    "template": item["template"],
                    "descriptor": item["descriptor"],
                }
            )
        self.sentences = sample_list

        # Load model & tokenizer
        from app.config.registry import get_model_loader

        try:
            self.model, self.tokenizer = get_model_loader().load(
                model_name=parameters["model_name"],
                model_parameters=parameters["model_parameters"],
                tokenizer_parameters=parameters["tokenizer_parameters"],
                cache=False,
            )
            logger.info(f"Loaded model and tokenizer for {parameters['model_name']}")
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise ValueError(f"Failed to load model/tokenizer: {e}")

        batch_size = 64 if torch.cuda.is_available() else 32
        binned_perplexities = self.evaluate_sentences(
            batch_size=batch_size,
            job_id=kwargs.get("job_id", ""),
            job_queue=kwargs.get("job_queue", None),
        )

        logger.info(
            f"Completed perplexity evaluation for {len(self.sentences)} sentences."
        )

        significance_dfs = []
        median_perplexities = []
        for axis in sorted(binned_perplexities.keys()):
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )

            logger.info(f"{JOB_ID}: Processing axis '{axis}' for significance testing.")

            perplexities_for_axis = binned_perplexities[axis]
            for template_index, template in enumerate(
                sorted(perplexities_for_axis.keys())
            ):
                self._check_job_cancellation(
                    kwargs.get("job_id", ""), kwargs.get("job_queue")
                )

                if template_index % 5 == 0:
                    logger.info(
                        f"{JOB_ID}: Processing template '{template}' ({template_index + 1}/{len(perplexities_for_axis)}) for axis '{axis}'."
                    )

                perplexities_for_template = perplexities_for_axis[template]

                these_median_perplexities = [
                    (axis, descriptor, template, np.median(ppls))
                    for descriptor, ppls in perplexities_for_template.items()
                ]
                median_perplexities.extend(these_median_perplexities)

                if len(perplexities_for_template) < 2:
                    continue

                descriptor_0s = []
                descriptor_1s = []
                descriptor_0_ppl_arrays = []
                descriptor_1_ppl_arrays = []
                for descriptor_0, descriptor_1 in combinations(
                    sorted(perplexities_for_template.keys()), r=2
                ):
                    self._check_job_cancellation(
                        kwargs.get("job_id", ""), kwargs.get("job_queue")
                    )

                    descriptor_0s.append(descriptor_0)
                    descriptor_1s.append(descriptor_1)
                    descriptor_0_ppl_arrays.append(
                        np.array(perplexities_for_template[descriptor_0])
                    )
                    descriptor_1_ppl_arrays.append(
                        np.array(perplexities_for_template[descriptor_1])
                    )
                full_descriptor_0_ppl_array = np.stack(descriptor_0_ppl_arrays)
                full_descriptor_1_ppl_array = np.stack(descriptor_1_ppl_arrays)

                # Calculate whether the Mann-Whitney U metric indicates statistical significance for each pair of sets of perplexities
                stat, pval = mannwhitneyu(
                    full_descriptor_0_ppl_array,
                    full_descriptor_1_ppl_array,
                    axis=1,
                )
                is_significant = (pval < MANN_WHITNEY_U_P_THRESHOLD).astype(int)
                significance_df = pd.DataFrame(
                    {
                        "axis": [axis] * len(descriptor_0s),
                        "template": [template] * len(descriptor_0s),
                        "descriptor_0": descriptor_0s,
                        "descriptor_1": descriptor_1s,
                        "mann_whitney_u": stat,
                        "p_value": pval,
                        "significant_difference": is_significant,
                    }
                )
                significance_dfs.append(significance_df)

        median_perplexity_df = pd.DataFrame(
            median_perplexities,
            columns=["axis", "descriptor", "template", "median_ppl"],
        )
        all_significance_df = (
            pd.concat(significance_dfs, axis=0)
            if significance_dfs
            else pd.DataFrame(
                columns=[
                    "axis",
                    "template",
                    "descriptor_0",
                    "descriptor_1",
                    "mann_whitney_u",
                    "p_value",
                    "significant_difference",
                ]
            )
        )

        return {
            "median_perplexity": median_perplexity_df.to_dict("records"),
            "significance_tests": all_significance_df.to_dict("records"),
        }
