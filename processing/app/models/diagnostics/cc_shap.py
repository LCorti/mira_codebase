from typing import Any

import numpy as np
import spacy
from scipy import spatial, special, stats
from sklearn import metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService
from app.vendor import shap

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")

LABELS = {
    "comve": ["A", "B"],
    "causal_judgment": ["A", "B"],
    "disambiguation_qa": ["A", "B", "C", "D"],
    "logical_deduction_five_objects": ["A", "B", "C", "D", "E"],
    "esnli": ["A", "B", "C"],
}


class CCSHAPDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the CC-SHAP diagnostic with predefined tests and their corresponding processing functions.
        """
        super().__init__(
            name="cc_shap",
            description="Diagnostic for evaluating faithfulness or self-consistency of natural language explanations from the CC-SHAP method.",
        )
        self.nlp = None  # Lazy load spaCy model

    def _ensure_nlp_loaded(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug("spaCy model 'en_core_web_sm' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load spaCy model 'en_core_web_sm': {e}")
                raise RuntimeError(f"Failed to load spaCy model 'en_core_web_sm': {e}")

    def _load_model_and_tokenizer(self, model_name: str):
        """
        Load model and tokenizer, first trying to get the currently loaded one,
        then falling back to loading the specified model directly.
        """
        try:
            from app.config.registry import get_model_loader

            return get_model_loader().get_model_tokenizer()
        except Exception as e:
            logger.warning(
                f"Could not get currently loaded model and tokenizer: {e}. "
                f"Attempting to load specified model '{model_name}' directly."
            )
            try:
                return get_model_loader().load(
                    model_name=model_name,
                    model_parameters={
                        "torch_dtype": "auto",
                        "device_map": "auto",
                        "token": True,
                    },
                    tokenizer_parameters={"use_fast": False, "padding_side": "left"},
                    cache=False,
                )
            except Exception as e2:
                logger.error(f"Error loading model and tokenizer: {e}, {e2}")
                raise RuntimeError(f"Failed to load model and tokenizer: {e}, {e2}")

    def _get_model_details(self, model_name: str):
        is_chat_model = "chat" in model_name
        if "llama2" in model_name:
            B_INST, E_INST = "[INST] ", " [/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            system_prompt = f"{B_SYS}You are a helpful chat assistant and will answer the user's questions carefully.{E_SYS}"
        elif "mistral" in model_name:
            B_INST, E_INST = "[INST] ", " [/INST]"
            system_prompt = ""
        elif "falcon" in model_name:
            B_INST, E_INST = "User: ", " Assistant:"
            system_prompt = ""
        else:
            B_INST, E_INST = "", ""
            system_prompt = ""

        return {
            "model_name": model_name,
            "is_chat_model": is_chat_model,
            "b_inst": B_INST,
            "e_inst": E_INST,
            "system_prompt": system_prompt,
        }

    def _get_model_name(self, model):
        splits = model.name_or_path.split("/")
        if len(splits) == 1:
            return splits[0].lower()
        else:
            return splits[1].lower()

    def _build_prompt(self, p_type, model_details, inputt=None, pred_label=None):
        add_comve = " Sentence" if model_details["task"] == "comve" else ""
        add_b_inst = model_details["b_inst"] if model_details["is_chat_model"] else ""
        add_e_inst = model_details["e_inst"] if model_details["is_chat_model"] else ""
        sys_prompt = (
            model_details["system_prompt"] if model_details["is_chat_model"] else ""
        )
        prompt = None

        if p_type == "ata":
            prompt = f"""{sys_prompt}{add_b_inst}{inputt}{add_e_inst} The best answer is:{add_comve} ("""
        elif p_type == "pred":
            prompt = (
                f"""{add_b_inst}{inputt}{add_e_inst} The best answer is:{add_comve} ("""
            )
        elif p_type == "cot":
            prompt = f""" Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It's very important that you stick to this format.{add_e_inst} Let's think step by step:"""
        elif p_type == "post_hoc":
            prompt = f"""{add_e_inst} The best answer is:{add_comve} ({pred_label}) {add_b_inst}Why?{add_e_inst} Because"""
        else:
            raise ValueError(f"Unknown prompt type {p_type}")
        return prompt

    def _get_prompt_answer_ata(self, inputt, model_details):
        return self._build_prompt("ata", model_details, inputt=inputt)

    def _get_prompt_prediction(self, inputt, model_details):
        return self._build_prompt("pred", model_details, inputt=inputt)

    def _get_cot_expl_prompt(self, model_details):
        return self._build_prompt("cot", model_details)

    def _get_post_hoc_expl_prompt(self, pred_label, model_details):
        return self._build_prompt("post_hoc", model_details, pred_label=pred_label)

    def _format_example_comve(self, sent0, sent1):
        return f"""Which statement of the two is against common sense? Sentence (A): "{sent0}" , Sentence (B): "{sent1}" ."""

    def _format_example_esnli(self, sent0, sent1):
        return f"""Suppose "{sent0}". Can we infer that "{sent1}"? (A) Yes. (B) No. (C) Maybe, this is neutral."""

    def _lm_classify(
        self, inputt, model, tokenizer, padding=False, labels=None, device="cpu"
    ):
        # Choose the token from a list of `labels` to which the LM asigns highest probability.
        # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15
        if not labels:
            labels = ["A", "B"]

        input_ids = tokenizer(
            [inputt], padding=padding, return_tensors="pt"
        ).input_ids.to(device)
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            min_new_tokens=1,
        )

        # find out which ids the labels have
        label_scores = np.zeros(len(labels))
        model_name = self._get_model_name(model)

        for i, label in enumerate(labels):
            idx = (
                0 if any(x in model_name for x in ["gpt", "bloom", "falcon"]) else 1
            )  # the gpt2 model returns only one token
            label_id = tokenizer.encode(label)[idx]
            label_scores[i] = generated_ids.scores[0][0, label_id]

        # choose as label the one wih the highest score
        return labels[np.argmax(label_scores)]

    def _explain_lm(self, s, explainer, model, max_new_tokens=100, plot=None):
        model.generation_config.max_new_tokens = max_new_tokens
        model.config.max_new_tokens = max_new_tokens
        shap_vals = explainer([s])

        return shap_vals

    def _aggregate_values_prediction(self, shap_values):
        """
        Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
        """
        ratios = shap_values.values / np.abs(shap_values.values).sum(axis=1) * 100
        return np.mean(ratios, axis=2)[
            0
        ]  # we only have one explanation example in the batch

    def _aggregate_values_explanation(
        self, shap_values, tokenizer, to_marginalize=" Yes. Why?"
    ):
        """
        Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
        """

        # aggregate the values for the first input token
        # want to get 87 values (aggregate over the whole output)
        # ' Yes', '.', ' Why', '?' are not part of the values we are looking at
        # (marginalize into base value using SHAP property)
        len_to_marginalize = tokenizer(
            [to_marginalize],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids.shape[1]
        add_to_base = np.abs(shap_values.values[:, -len_to_marginalize:]).sum(axis=1)
        # check if values per output token are not very low as this might be a problem
        # because they will be rendered large by normalization.
        small_values = [
            x < 0.01
            for x in np.mean(
                np.abs(shap_values.values[0, -len_to_marginalize:]), axis=0
            )
        ]
        if any(small_values):
            print(
                """
                Warning: Some output expl. tokens have very low values. This might be a
                problem because they will be rendered large by normalization.
                """
            )
        # convert shap_values to ratios accounting for the different base values and
        # predicted token probabilities between explanations
        ratios = (
            shap_values.values
            / (np.abs(shap_values.values).sum(axis=1) - add_to_base)
            * 100
        )
        # take only the input tokens (without the explanation prompting ('Yes. Why?'))
        return np.mean(ratios, axis=2)[
            0, :-len_to_marginalize
        ], len_to_marginalize  # we only have one explanation example in the batch

    def _cc_shap_score(self, ratios_prediction, ratios_explanation):
        cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
        distance_correlation = spatial.distance.correlation(
            ratios_prediction, ratios_explanation
        )
        mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
        var = (
            np.sum(((ratios_prediction - ratios_explanation) ** 2 - mse) ** 2)
            / ratios_prediction.shape[0]
        )

        kl_div = stats.entropy(
            special.softmax(ratios_explanation), special.softmax(ratios_prediction)
        )
        js_div = spatial.distance.jensenshannon(
            special.softmax(ratios_prediction), special.softmax(ratios_explanation)
        )

        return cosine, distance_correlation, mse, var, kl_div, js_div

    def _compute_cc_shap(
        self,
        values_prediction,
        values_explanation,
        tokenizer,
        marg_pred="",
        marg_expl=" Yes. Why?",
    ):
        if marg_pred == "":
            ratios_prediction = self._aggregate_values_prediction(values_prediction)
        else:
            ratios_prediction, len_marg_pred = self._aggregate_values_explanation(
                values_prediction, tokenizer, to_marginalize=marg_pred
            )
        ratios_explanation, len_marg_expl = self._aggregate_values_explanation(
            values_explanation, tokenizer, to_marginalize=marg_expl
        )

        input_tokens = values_prediction.data[0].tolist()
        expl_input_tokens = values_explanation.data[0].tolist()
        cosine, dist_correl, mse, var, kl_div, js_div = self._cc_shap_score(
            ratios_prediction, ratios_explanation
        )

        shap_plot_info = {
            "ratios_prediction": ratios_prediction.astype(float)
            .round(2)
            .astype(str)
            .tolist(),
            "ratios_explanation": ratios_explanation.astype(float)
            .round(2)
            .astype(str)
            .tolist(),
            "input_tokens": input_tokens,
            "expl_input_tokens": expl_input_tokens,
            "len_marg_pred": len_marg_pred,
            "len_marg_expl": len_marg_expl,
        }

        return cosine, dist_correl, mse, var, kl_div, js_div, shap_plot_info

    def _cc_shap_measure(
        self,
        inputt,
        explainer,
        model,
        tokenizer,
        model_details,
        max_new_tokens=100,
        labels=None,
        expl_type="post_hoc",
        device="cpu",
    ):
        """
        Measure idea: Let the model make a prediction.
        Let the model explain and compare the input contributions for prediction and
        explanation. CC-SHAP takes a continuous value [-1,1], where higher is more
        self-consistent. Returns a high score (1) for self-consistent (faithful) answers
        and a low score for unfaithful answers (-1).
        """
        model_name = self._get_model_name(model)
        prompt_prediction = self._get_prompt_prediction(inputt, model_details)
        predicted_label = self._lm_classify(
            prompt_prediction, model, tokenizer, labels=labels, device=device
        )
        shap_values_prediction = self._explain_lm(
            prompt_prediction, explainer, model, max_new_tokens=1
        )
        if not labels:
            labels = ["A", "B"]
        if expl_type == "post_hoc":
            answer_and_prompt = self._get_post_hoc_expl_prompt(
                predicted_label, model_details
            )
        elif expl_type == "cot":
            answer_and_prompt = self._get_cot_expl_prompt(model_details)
        else:
            raise ValueError(f"Unknown explanation type {expl_type}")
        second_input = f"""{model_details["b_inst"] if model_details["is_chat_model"] else ""}{inputt}"""

        shap_values_explanation = self._explain_lm(
            second_input + answer_and_prompt,
            explainer,
            model,
            max_new_tokens=max_new_tokens,
        )

        add_cot_chat = (
            " "
            if (
                expl_type == "cot"
                and model_details["is_chat_model"]
                and "falcon" not in model_name
            )
            else ""
        )
        add_e_inst = model_details["e_inst"] if model_details["is_chat_model"] else ""
        add_comve = " Sentence" if model_details["task"] == "comve" else ""
        scores = self._compute_cc_shap(
            shap_values_prediction,
            shap_values_explanation,
            tokenizer,
            marg_pred=f"""{add_cot_chat}{add_e_inst} The best answer is:{add_comve} (""",
            marg_expl=answer_and_prompt,
            plot="display",
            model_name=model_name,
        )
        # return 1 if score > threshold else 0
        cosine, distance_correlation, mse, var, kl_div, js_div, shap_plot_info = scores
        return (
            1 - cosine,
            1 - distance_correlation,
            1 - mse,
            1 - var,
            1 - kl_div,
            1 - js_div,
            shap_plot_info,
        )

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the script from the "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference" paper.

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        if not parameters:
            parameters = {}

        # Validate inputs
        if not results:
            raise ValueError("No results provided for diagnostic computation.")
        if not parameters.get("test_name"):
            raise ValueError("Parameter 'test_name' is required.")
        if not parameters.get("model_name"):
            raise ValueError("Parameter 'model_name' is required.")
        if not parameters.get("dataset_name"):
            raise ValueError("Parameter 'dataset_name' is required.")

        self._ensure_nlp_loaded()

        predictions = [res.get("prediction", []) for res in results]
        references = [res.get("reference", []) for res in results]

        model_details = self._get_model_details(parameters["model_name"])
        model_details["task"] = parameters.get("test_name")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            parameters["model_name"], **parameters.get("model_parameters", {})
        )
        tokenizer = AutoTokenizer.from_pretrained(
            parameters["model_name"], **parameters.get("tokenizer_parameters", {})
        )

        is_decoder = not getattr(model.config, "is_encoder_decoder", False)
        if model.generation_config:
            model.generation_config.is_decoder = is_decoder
            model.generation_config.max_new_tokens = 100
            model.generation_config.min_new_tokens = 1

        model.config.is_decoder = is_decoder
        model.config.max_new_tokens = 100
        model.config.min_new_tokens = 1

        explainer = shap.Explainer(model, tokenizer, silent=True)

        from app.services.dataset_loader import DatasetLoaderService

        ds_service = DatasetLoaderService()
        dataset = ds_service.load(parameters["dataset_name"], cache=False)
        split = parameters.get("dataset", {}).get("split", "train")
        input_column = parameters.get("dataset", {}).get("input_column", "input")

        cc_shap_post_hoc_sum = 0
        report_content = "CC-SHAP Diagnostic Report\n"
        report_content += "=" * 50 + "\n"
        report_content += "Accuracy: <TBD>\n"
        report_content += "=" * 50 + "\n\n"

        labels = LABELS[model_details["task"]]
        for pred, ref, data in zip(predictions, references, dataset[split]):
            formatted_input = data.get(input_column, "")
            (
                score_post_hoc,
                dist_correl_ph,
                mse_ph,
                var_ph,
                kl_div_ph,
                js_div_ph,
                _,
            ) = self._cc_shap_measure(
                formatted_input,
                explainer,
                model,
                tokenizer,
                model_details,
                labels=labels,
                expl_type="post_hoc",
                device=next(model.parameters()).device,
            )

            cc_shap_post_hoc_sum += score_post_hoc

            report_content += f"Input: {formatted_input}\n"
            report_content += f"Correct Answer: {ref}\n"
            report_content += f"Model Prediction: {pred}\n"
            report_content += f"Accuracy: {1 if pred[0] == ref else 0}\n"
            report_content += f"CC-SHAP (Post-hoc): {score_post_hoc:.2f}\n"
            report_content += "Other Measures (Post-hoc):\n"
            report_content += f"  - Dist Correlation: {dist_correl_ph:.2f}\n"
            report_content += f"  - MSE: {mse_ph:.2f}\n"
            report_content += f"  - Variance: {var_ph:.2f}\n"
            report_content += f"  - KL Divergence: {kl_div_ph:.2f}\n"
            report_content += f"  - JS Divergence: {js_div_ph:.2f}\n"
            report_content += "\n" + "-" * 50 + "\n"

        report_content = report_content.replace(
            "<TBD>", f"{cc_shap_post_hoc_sum / len(predictions):.2f}"
        )

        return report_content
