import re
from typing import Any

import numpy as np

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")

CPRAG_STIM = "app/data/lm_diagnostics/CPRAG-102/CPRAG-102.tsv"  # Set path to CPRAG stimulus file here
NEGNAT_STIM = "app/data/lm_diagnostics/NEG-136/NEG-136-NAT.tsv"  # Set path to NEG-NAT stimulus file here
NEGSIMP_STIM = "app/data/lm_diagnostics/NEG-136/NEG-136-SIMP.tsv"  # Set path to NEG-SIMP stimulus file here
ROLE_STIM = (
    "app/data/lm_diagnostics/ROLE-88/ROLE-88.tsv"  # Set path to ROLE stimulus file here
)

JOB_ID = ""
JOB_QUEUE = None


class LMDiagnosticsAccuracyDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the Language Model Diagnostics Accuracy diagnostic.
        """
        super().__init__(
            name="lm_diagnostics_accuracy",
            description="Diagnostic for evaluating language model accuracy from LM Diagnostic method.",
        )
        self.test_list = {
            "cprag": (CPRAG_STIM, self.test_cprag_acc, self.process_cprag),
            "negnat": (NEGNAT_STIM, self.test_neg_acc, self.process_negnat),
            "negsimp": (NEGSIMP_STIM, self.test_neg_acc, self.process_negsimp),
            "role": (ROLE_STIM, self.test_role_acc, self.process_role),
        }

    def process_cprag(self, tsv_file, mask_tok=True):
        hl_dict = {}
        input_list = []
        tgt_list = []
        with open(tsv_file) as f:
            i = 0
            for line in f:
                self._check_job_cancellation(JOB_ID, JOB_QUEUE)

                if len(line.strip("\t")) < 2:
                    continue
                it, sent1, sent2, exp, wc, bc, constraint = line.strip().split("\t")
                if it == "item":
                    continue
                context = f"{sent1} {sent2}"
                for tgt, cond in ((exp, "exp"), (wc, "wc"), (bc, "bc")):
                    hl_dict[i] = {}
                    if mask_tok:
                        input_list.append(context + " [MASK]")
                    else:
                        input_list.append(context)
                    tgt_list.append(tgt)
                    hl_dict[i]["constraint"] = constraint
                    hl_dict[i]["tgt"] = tgt
                    hl_dict[i]["sent"] = {}
                    hl_dict[i]["sent"] = context
                    hl_dict[i]["item"] = it
                    hl_dict[i]["exp"] = [exp]
                    hl_dict[i]["cond"] = cond
                    i += 1
        return input_list, tgt_list, hl_dict

    def process_role(self, tsv_file, mask_tok=True, gen_obj=False, gen_subj=False):
        input_list = []
        tgt_list = []
        cloze_dict = {}
        cloze_list = []
        i = 0
        with open(tsv_file, "r") as f:
            for line in f:
                self._check_job_cancellation(JOB_ID, JOB_QUEUE)

                line_split = line.strip().split("\t")
                item = line_split[0]
                if item == "item" or len(line_split) < 1:
                    continue
                item_num, condition = item.split("-")
                sent = line_split[1]
                exp = line_split[2].split("|")
                max_cloze = float(line_split[3])
                tgt = line_split[4].strip().split()[0]
                tgt_cloze = float(line_split[5])
                tgt_cloze_strict = float(line_split[6])
                if gen_obj:
                    sent = re.sub(r"which .* the", "which one the", sent)
                if gen_subj:
                    sent = re.sub(
                        r"which (.*) the .* had", r"which \g<1> the other had", sent
                    )
                cloze_dict[i] = {}
                if mask_tok:
                    context = sent + " [MASK]"
                else:
                    context = sent
                if context in input_list:
                    for item in cloze_dict:
                        if cloze_dict[item]["sent"] == context:
                            cloze_dict[i]["maxcloze"] = max_cloze
                            break
                else:
                    cloze_dict[i]["maxcloze"] = max_cloze
                input_list.append(context)
                tgt_list.append(tgt)
                cloze_dict[i]["sent"] = context
                cloze_dict[i]["tgt"] = tgt
                cloze_dict[i]["cond"] = condition
                cloze_dict[i]["item"] = item_num
                cloze_dict[i]["tgtcloze"] = tgt_cloze
                cloze_dict[i]["tgtcloze_strict"] = tgt_cloze_strict
                cloze_dict[i]["exp"] = exp
                cloze_list.append(max_cloze)
                i += 1
        return input_list, tgt_list, (cloze_dict, cloze_list)

    def process_negsimp(self, tsv_file, mask_tok=True):
        nk_dict = {}
        input_list = []
        tgt_list = []
        i = 0

        with open(tsv_file, "r") as f:
            for line in f:
                self._check_job_cancellation(JOB_ID, JOB_QUEUE)

                it, affsent, negsent, afftgt, negtgt = [
                    e.strip() for e in line.strip().split("\t")
                ]
                affsent = re.sub(r" \(.+\)", "", affsent)
                negsent = re.sub(r" \(.+\)", "", negsent)
                if it == "item":
                    continue
                for sent, tgt, cond in [
                    (affsent, afftgt, "TA"),
                    (negsent, afftgt, "FN"),
                    (affsent, negtgt, "FA"),
                    (negsent, negtgt, "TN"),
                ]:
                    nk_dict[i] = {}
                    if re.match("[aeiou]", tgt):
                        det = "an"
                    else:
                        det = "a"
                    if mask_tok:
                        context = f"{sent} {det} [MASK]"
                    else:
                        context = f"{sent} {det}"
                    nk_dict[i]["sent"] = context
                    nk_dict[i]["tgt"] = tgt
                    nk_dict[i]["item"] = it
                    nk_dict[i]["cond"] = cond
                    if cond in ("TA", "FA"):
                        nk_dict[i]["exp"] = [afftgt]
                    else:
                        nk_dict[i]["exp"] = [negtgt]
                    input_list.append(context)
                    tgt_list.append(tgt)
                    i += 1
        return input_list, tgt_list, nk_dict

    def process_negnat(self, tsv_file, mask_tok=True):
        nk_dict = {}
        input_list = []
        tgt_list = []
        i = 0
        with open(tsv_file, "r") as f:
            for line in f:
                self._check_job_cancellation(JOB_ID, JOB_QUEUE)

                it, affsent, negsent, afftgt, negtgt, lic = [
                    e.strip() for e in line.strip().split("\t")
                ]
                if it == "item":
                    continue
                for sent, tgt, cond in [
                    (affsent, afftgt, "TA"),
                    (negsent, afftgt, "FN"),
                    (affsent, negtgt, "FA"),
                    (negsent, negtgt, "TN"),
                ]:
                    nk_dict[i] = {}
                    if mask_tok:
                        context = sent + " [MASK]"
                    else:
                        context = sent
                    nk_dict[i]["sent"] = context
                    nk_dict[i]["tgt"] = tgt
                    nk_dict[i]["item"] = it
                    nk_dict[i]["cond"] = cond
                    nk_dict[i]["licensing"] = lic
                    if cond in ("TA", "FA"):
                        nk_dict[i]["exp"] = [afftgt]
                    else:
                        nk_dict[i]["exp"] = [negtgt]
                    input_list.append(context)
                    tgt_list.append(tgt)
                    i += 1
        return input_list, tgt_list, nk_dict

    def get_acc(self, score_list) -> float:
        if len(score_list) == 0:
            return 0.0
        return float(sum(score_list)) / len(score_list)

    def test_cprag_acc(self, dataset_ref, word_preds, k=5):
        tot_score = []
        correct = []
        used = []
        for i, pr in enumerate(word_preds):
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            sent = dataset_ref[i]["sent"]

            score = 0
            if sent in used:
                continue

            used.append(sent)

            for candidate in pr:
                if candidate.strip() in dataset_ref[i]["exp"]:
                    score = 1

            if score == 1:
                ctup = (sent, dataset_ref[i]["exp"], pr)
                correct.append(ctup)

            tot_score.append(score)

        tot_acc = self.get_acc(tot_score)
        report = "\nPrediction accuracies:\n"
        report += f"EXPECTED TARGET in TOP {k} predictions: {tot_acc} ({sum(tot_score)}/{len(tot_score)})\n\n"
        report += "ITEMS PREDICTED CORRECTLY:\n\n"

        for sent, exp, pred in correct:
            report += (
                f"{sent}\n   EXPECTED: {','.join(exp)} | PREDICTED: {','.join(pred)}\n"
            )
        return report

    def test_role_acc(self, dataset_ref, word_preds, k=5):
        dataset_dict, cloze_list = dataset_ref

        tot_score = []
        correct = []
        correct_by_quartile = {
            "q1_corr": [],
            "q2_corr": [],
            "q3_corr": [],
            "q4_corr": [],
        }
        q1 = np.percentile(cloze_list, 25)
        q3 = np.percentile(cloze_list, 75)
        q2 = np.percentile(cloze_list, 50)
        q4 = max(cloze_list)
        for i, pred in enumerate(word_preds):
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            sent = dataset_dict[i]["sent"]
            score = 0
            for candidate in pred:
                if candidate.strip() in [e.split()[0] for e in dataset_dict[i]["exp"]]:
                    score = 1
            if score == 1:
                ctup = (sent, dataset_dict[i]["exp"], pred, dataset_dict[i]["maxcloze"])
                correct.append(ctup)
            tot_score.append(score)
            if dataset_dict[i]["maxcloze"] <= q1:
                correct_by_quartile["q1_corr"].append(score)
            elif dataset_dict[i]["maxcloze"] <= q2:
                correct_by_quartile["q2_corr"].append(score)
            elif dataset_dict[i]["maxcloze"] <= q3:
                correct_by_quartile["q3_corr"].append(score)
            else:
                correct_by_quartile["q4_corr"].append(score)

        report = "\nPrediction accuracies:\n"
        report += f"EXPECTED WORD in top {k} predictions: {self.get_acc(tot_score)} ({sum(tot_score)}/{len(tot_score)})\n"
        report += f"  for cloze quartile 1: {self.get_acc(correct_by_quartile['q1_corr'])} ({q1} upper, {len(correct_by_quartile['q1_corr'])} items)\n"
        report += f"  for cloze quartile 2: {self.get_acc(correct_by_quartile['q2_corr'])} ({q2} upper, {len(correct_by_quartile['q2_corr'])} items)\n"
        report += f"  for cloze quartile 3: {self.get_acc(correct_by_quartile['q3_corr'])} ({q3} upper, {len(correct_by_quartile['q3_corr'])} items)\n"
        report += f"  for cloze quartile 4: {self.get_acc(correct_by_quartile['q4_corr'])} ({q4} upper, {len(correct_by_quartile['q4_corr'])} items)\n\n"

        report += "ITEMS PREDICTED CORRECTLY:\n\n"
        for sent, exp, pred, mc in correct:
            report += (
                f"{sent}\n   EXPECTED: {','.join(exp)} | PREDICTED: {','.join(pred)}\n"
            )
        return report

    def test_neg_acc(self, dataset_ref, word_preds, k=5):
        correct = []
        tot_score = []
        used = []
        for i, pred in enumerate(word_preds):
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            score = 0
            sent = dataset_ref[i]["sent"]
            if sent in used:
                continue
            used.append(sent)
            for candidate in pred:
                if candidate.strip() in dataset_ref[i]["exp"]:
                    score = 1
            if score == 1 and dataset_ref[i]["cond"] == "TA":
                ctup = (sent, dataset_ref[i]["exp"], pred, dataset_ref[i]["cond"])
                correct.append(ctup)
            if dataset_ref[i]["cond"] == "TA":
                tot_score.append(score)
        report = "\nPrediction 'accuracy' (affirmative contexts only):\n"
        report += (
            f"TRUE COMPLETION in top {k} predictions: "
            f"{self.get_acc(tot_score)} ({sum(tot_score)}/{len(tot_score)})\n\n"
        )

        report += "ITEMS PREDICTED CORRECTLY:\n\n"
        for sent, exp, pred, cond in correct:
            report += (
                f"{sent}\n   EXPECTED: {','.join(exp)} | PREDICTED: {','.join(pred)}\n"
            )
        return report

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the "accuracy test" script the LM Diagnostics paper (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00298/43535/What-BERT-Is-Not-Lessons-from-a-New-Suite-of).

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :type results: list[dict]
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :type parameters: dict
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        :type kwargs: Any
        """

        # Validate inputs
        if not parameters:
            parameters = {}
        if not results:
            raise ValueError("No results provided for diagnostic computation.")

        global JOB_ID, JOB_QUEUE
        JOB_ID = kwargs.get("job_id", "")
        JOB_QUEUE = kwargs.get("job_queue", None)

        test_name = parameters.get(
            "test_name", "cprag"
        )  # Default to 'cprag' if not provided
        predictions = [res.get("prediction", []) for res in results]
        k = parameters.get("k", len(predictions[0]) if predictions else 1)

        stimfile, acc_test, process_func = self.test_list.get(
            test_name, (None, None, None)
        )
        if not stimfile or not acc_test or not process_func:
            logger.error(
                f"Invalid test name '{test_name}'. Available tests: {list(self.test_list.keys())}"
            )
            raise ValueError(
                f"Invalid test name '{test_name}'. Available tests: {list(self.test_list.keys())}"
            )

        # Test file
        try:
            with open(stimfile, "r") as _:
                pass
        except FileNotFoundError:
            logger.error(
                f"LM Diagnostics {test_name} data file not found at path: {stimfile}. Please ensure the file exists."
            )
            raise FileNotFoundError(
                f"LM Diagnostics {test_name} data file not found at path: {stimfile}. Please ensure the file exists."
            )

        _, _, dataset_ref = process_func(stimfile, mask_tok=False)

        # Build report metadata header
        report_content = "---\n"
        report_content += f"test_name: {test_name}\n"
        report_content += f"k = {k}\n"
        report_content += "---\n"

        # Get the accuracy test report
        report_content += acc_test(dataset_ref, predictions, k=k)

        return report_content
