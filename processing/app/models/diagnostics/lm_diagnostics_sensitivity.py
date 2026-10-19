import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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


class LMDiagnosticsSensitivityDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the Language Model Diagnosis Sensitivity metric.
        """
        super().__init__(
            name="lm_diagnostics_sensitivity",
            description="Diagnostic for evaluating language model sensitivity from LM Diagnostic method.",
        )
        self.metric = None
        self.test_list = {
            "cprag": (CPRAG_STIM, self.cprag_sensitivity_test, self.process_cprag),
            "negnat": (NEGNAT_STIM, self.neg_sensitivity_test, self.process_negnat),
            "negsimp": (NEGSIMP_STIM, self.neg_sensitivity_test, self.process_negsimp),
            "role": (ROLE_STIM, self.role_sensitivity_test, self.process_role),
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
                cloze_dict[i]["tgt_cloze_strict"] = tgt_cloze_strict
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

    def convert_to_experiment_grouping(self, data_dict):
        cond_dict = {}
        for k in data_dict:
            it = data_dict[k]["item"]
            co = data_dict[k]["cond"]

            if it not in cond_dict:
                cond_dict[it] = {}

            cond_dict[it][co] = {}
            for key_cat in [
                "sent",
                "tgt",
                "tgtcloze",
                "constraint",
                "licensing",
                "maxcloze",
                "tgtprob",
                "toppreds",
                "topprobs",
            ]:
                if key_cat not in data_dict[k]:
                    continue
                cond_dict[it][co][key_cat] = data_dict[k][key_cat]
        return cond_dict

    def cprag_sensitivity_test(self, dataset_ref, target_probs):
        for i, prob in enumerate(target_probs):
            dataset_ref[i]["tgtprob"] = prob

        cond_dict = self.convert_to_experiment_grouping(dataset_ref)

        thresh = 0.01
        exp_top = {"H": [], "L": []}
        exp_top_thresh = {"H": [], "L": []}
        all_probs = {"H": [], "L": []}
        for it in cond_dict:
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            # Check if all required conditions have tgtprob
            if not all(
                cont in cond_dict[it] and "tgtprob" in cond_dict[it][cont]
                for cont in ["exp", "wc", "bc"]
            ):
                logger.warning(f"Missing tgtprob for item {it}, skipping")
                continue
            exp_prob, wc_prob, bc_prob = [
                cond_dict[it][cont]["tgtprob"] for cont in ["exp", "wc", "bc"]
            ]

            cons = cond_dict[it]["exp"]["constraint"]
            all_probs[cons].append((exp_prob, wc_prob, bc_prob))
            if (exp_prob > wc_prob) and (exp_prob > bc_prob):
                exp_top[cons].append(1)
                if (
                    abs(exp_prob - wc_prob) > thresh
                    and abs(exp_prob - bc_prob) > thresh
                ):
                    exp_top_thresh[cons].append(1)
                else:
                    exp_top_thresh[cons].append(0)
            else:
                exp_top[cons].append(0)
                exp_top_thresh[cons].append(0)

        report = "\nSensitivity results:\n\n"

        report += f"Expected word more probable than two inappropriate targets: {self.get_acc(exp_top['H'] + exp_top['L'])} ({sum(exp_top['H'] + exp_top['L'])}/{len(exp_top['H'] + exp_top['L'])})\n"
        report += f"  for high-constraint items: {self.get_acc(exp_top['H'])} ({sum(exp_top['H'])}/{len(exp_top['H'])})\n"
        report += f"  for low-constraint items: {self.get_acc(exp_top['L'])} ({sum(exp_top['L'])}/{len(exp_top['L'])})\n\n"
        report += f"Expected word more probable than two inappropriate targets -- difference threshold {thresh}: {self.get_acc(exp_top_thresh['H'] + exp_top_thresh['L'])} ({sum(exp_top_thresh['H'] + exp_top_thresh['L'])}/{len(exp_top_thresh['H'] + exp_top_thresh['L'])})\n"
        report += f"  for high-constraint items: {self.get_acc(exp_top_thresh['H'])} ({sum(exp_top_thresh['H'])}/{len(exp_top_thresh['H'])})\n"
        report += f"  for low-constraint items: {self.get_acc(exp_top_thresh['L'])} ({sum(exp_top_thresh['L'])}/{len(exp_top_thresh['L'])})\n"

        return report

    def neg_sensitivity_test(self, dataset_ref, target_probs):
        for i, prob in enumerate(target_probs):
            dataset_ref[i]["tgtprob"] = prob

        cond_dict = self.convert_to_experiment_grouping(dataset_ref)

        thresh = 0.01
        pref_true = {"aff": [], "neg": []}
        pref_true_l = {"aff": [], "neg": []}
        pref_true_u = {"aff": [], "neg": []}
        pref_true_thresh = {"aff": [], "neg": []}
        lic = None
        for it in cond_dict:
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            # Check if all required conditions have tgtprob
            if not all(
                cont in cond_dict[it] and "tgtprob" in cond_dict[it][cont]
                for cont in ["TA", "FA", "TN", "FN"]
            ):
                logger.warning(f"Missing tgtprob for item {it}, skipping")
                continue

            if "licensing" in cond_dict[it]["TA"]:
                lic = cond_dict[it]["TA"]["licensing"]

            for true_cond, false_cond, pol in [
                ("TA", "FA", "aff"),
                ("TN", "FN", "neg"),
            ]:
                true_prob, false_prob = (
                    cond_dict[it][true_cond]["tgtprob"],
                    cond_dict[it][false_cond]["tgtprob"],
                )

                if true_prob > false_prob:
                    score = 1
                else:
                    score = 0

                pref_true[pol].append(score)

                if lic:
                    if lic == "Y":
                        pref_true_l[pol].append(score)
                    elif lic == "N":
                        pref_true_u[pol].append(score)

                if (true_prob > false_prob) and (abs(true_prob - false_prob) > thresh):
                    pref_true_thresh[pol].append(1)
                else:
                    pref_true_thresh[pol].append(0)

        report = "\nSensitivity results:\n\n"
        report += f"True completion more probable than false: {self.get_acc(pref_true['aff'] + pref_true['neg'])} ({sum(pref_true['aff'] + pref_true['neg'])}/{len(pref_true['aff'] + pref_true['neg'])})\n"
        report += f"   in affirmative contexts: {self.get_acc(pref_true['aff'])} ({sum(pref_true['aff'])}/{len(pref_true['aff'])})\n"
        report += f"   in negative contexts: {self.get_acc(pref_true['neg'])} ({sum(pref_true['neg'])}/{len(pref_true['neg'])})\n"
        report += f"True completion more probable than false -- difference threshold {thresh}: {self.get_acc(pref_true_thresh['aff'] + pref_true_thresh['neg'])} ({sum(pref_true_thresh['aff'] + pref_true_thresh['neg'])}/{len(pref_true_thresh['aff'] + pref_true_thresh['neg'])})\n"
        report += f"   in affirmative contexts: {self.get_acc(pref_true_thresh['aff'])} ({sum(pref_true_thresh['aff'])}/{len(pref_true_thresh['aff'])})\n"
        report += f"   in negative contexts: {self.get_acc(pref_true_thresh['neg'])} ({sum(pref_true_thresh['neg'])}/{len(pref_true_thresh['neg'])})\n"
        if lic:
            report += f"True completion more probable in NATURAL sentences: {self.get_acc(pref_true_l['aff'] + pref_true_l['neg'])} ({sum(pref_true_l['aff'] + pref_true_l['neg'])}/{len(pref_true_l['aff'] + pref_true_l['neg'])})\n"
            report += f"   in affirmative contexts: {self.get_acc(pref_true_l['aff'])} ({sum(pref_true_l['aff'])}/{len(pref_true_l['aff'])})\n"
            report += f"   in negative contexts: {self.get_acc(pref_true_l['neg'])} ({sum(pref_true_l['neg'])}/{len(pref_true_l['neg'])})\n"
            report += f"True completion more probable in LESS NATURAL sentences: {self.get_acc(pref_true_u['aff'] + pref_true_u['neg'])} ({sum(pref_true_u['aff'] + pref_true_u['neg'])}/{len(pref_true_u['aff'] + pref_true_u['neg'])})\n"
            report += f"   in affirmative contexts: {self.get_acc(pref_true_u['aff'])} ({sum(pref_true_u['aff'])}/{len(pref_true_u['aff'])})\n"
            report += f"   in negative contexts: {self.get_acc(pref_true_u['neg'])} ({sum(pref_true_u['neg'])}/{len(pref_true_u['neg'])})\n"
        report += "\n\n"

        return report

    def role_sensitivity_test(self, dataset_ref, target_probs):
        dataset_dict, _ = dataset_ref
        for i, prob in enumerate(target_probs):
            dataset_dict[i]["tgtprob"] = prob

        cond_dict = self.convert_to_experiment_grouping(dataset_dict)

        thresh = 0.01
        good_top = []
        good_top_thresh = []
        prob_pairs = []
        cloze_pairs = []
        for it in cond_dict:
            self._check_job_cancellation(JOB_ID, JOB_QUEUE)

            # Check if all required conditions have tgtprob
            if not all(
                cont in cond_dict[it] and "tgtprob" in cond_dict[it][cont]
                for cont in ["a", "b"]
            ):
                logger.warning(f"Missing tgtprob for item {it}, skipping")
                continue

            a_prob, b_prob = (
                cond_dict[it]["a"]["tgtprob"],
                cond_dict[it]["b"]["tgtprob"],
            )
            if a_prob > b_prob:
                good_top.append(1)
            else:
                good_top.append(0)
            if (a_prob > b_prob) and (abs(a_prob - b_prob) > thresh):
                good_top_thresh.append(1)
            else:
                good_top_thresh.append(0)
            prob_pairs.append((a_prob, b_prob))
            cloze_pairs.append(
                (cond_dict[it]["a"]["tgtcloze"], cond_dict[it]["b"]["tgtcloze"])
            )
        prob_diffs = [e[0] - e[1] for e in prob_pairs]
        cloze_diffs = [e[0] - e[1] for e in cloze_pairs]

        report = "\nSensitivity results:\n\n"

        report += f"Good completion more probable than role reversal: {self.get_acc(good_top)} ({sum(good_top)}/{len(good_top)})\n\n"
        report += f"Good completion more probable than role reversal -- difference threshold {thresh}: {self.get_acc(good_top_thresh)} ({sum(good_top_thresh)}/{len(good_top_thresh)})\n\n"
        report += f"AVG PROB DIFF: {np.average(prob_diffs)}\n"
        report += f"AVG CLOZE DIFF: {np.average(cloze_diffs)}\n"

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
        references = [res.get("reference", "")[0] for res in results]

        from app.controllers.logit_controller import LogitController

        logits = [
            LogitController.get_logit_tensor_by_id(res.get("logit_id", ""))
            for res in results
        ]

        # Try to get tokenizer from kwargs or registry
        tokenizer = None
        from app.config.registry import get_model_loader

        try:
            _, tokenizer = get_model_loader().get_model_tokenizer()
        except Exception as e:
            logger.warning(f"Could not get tokenizer from model registry: {e}")
            model_name = parameters.get("model_name", None)
            _, tokenizer = get_model_loader().load(model_name=model_name, cache=False)

        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided either as parameter or available in model registry for LM-Diagnostics metric computation."
            )

        target_probs = []
        for logit, target in zip(logits, references):
            if (
                logit is None
                or not isinstance(logit, torch.Tensor)
                or not target
                or not isinstance(target, str)
            ):
                target_probs.append(0.0)
                continue

            probs = F.softmax(logit[0], dim=-1)

            target_token_id = tokenizer.encode(target, add_special_tokens=False)
            if len(target_token_id) > 0:
                target_id = target_token_id[0]  # Use first token ID if multiple
                target_prob = probs[target_id].item()
            else:
                target_prob = 0.0
            target_probs.append(target_prob)

        stimfile, sens_test, process_func = self.test_list.get(
            test_name, (None, None, None)
        )
        if not stimfile or not sens_test or not process_func:
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
        report_content += "---\n"

        # Get the sensitivity test report
        report_content += sens_test(dataset_ref, target_probs)

        return report_content
