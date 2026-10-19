from typing import Any

import numpy as np
import torch
from captum.attr import LLMAttribution, ShapleyValueSampling, TextTokenInput
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")


class EncoderDecoderAttributionWrapper(torch.nn.Module):
    """
    Wrapper for encoder-decoder models (like T5) for attribution.
    Maps input tokens to output logits for a specific target token position.
    """

    def __init__(self, model, target_ids, target_token_idx=0):
        super().__init__()
        self.model = model
        self.target_ids = target_ids
        self.target_token_idx = target_token_idx
        self.device = next(model.parameters()).device
        self.target_ids = self.target_ids.to(self.device)

    def set_target_token_idx(self, idx):
        self.target_token_idx = idx

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Re-encode inputs for each forward pass (required for Shapley value computation)
        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.model.config.pad_token_id

        # Build decoder input: [decoder_start_token] + target_tokens[0:target_token_idx]
        decoder_start = torch.full(
            (batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device
        )

        if (
            self.target_token_idx > 0
        ):  # Append previous target tokens to match autoregressive generation
            prev_tokens = self.target_ids[:, : self.target_token_idx].expand(
                batch_size, -1
            )
            decoder_input_ids = torch.cat([decoder_start, prev_tokens], dim=1)
        else:
            decoder_input_ids = decoder_start

        outputs = self.model(
            encoder_outputs=(encoder_outputs.last_hidden_state,),
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        # Get logits for the position we're predicting
        logits = outputs.logits[:, -1, :]  # Last position's logits
        target_token_id = self.target_ids[0, self.target_token_idx]

        # Return log-probability of the target token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[:, target_token_id].unsqueeze(-1)


class ShapDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the SHAP diagnostic.
        """
        super().__init__(
            name="shap",
            description="Diagnostic for evaluating SHapley Additive exPlanations (SHAP).",
        )

    def _preprocess_model_parameters(self, parameters):
        """
        Preprocess model parameters to convert string representations to actual PyTorch types.
        """
        processed = parameters.copy()

        # Remove invalid parameters
        invalid_params = ["architecture"]
        for param in invalid_params:
            processed.pop(param, None)

        # Handle torch_dtype conversion
        if "torch_dtype" in processed:
            dtype_value = processed["torch_dtype"]
            if isinstance(dtype_value, str):
                dtype_map = {
                    "float16": torch.float16,
                    "torch.float16": torch.float16,
                    "float32": torch.float32,
                    "torch.float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                    "torch.bfloat16": torch.bfloat16,
                    "float": torch.float32,
                    "half": torch.float16,
                    "auto": "auto",
                }
                processed["torch_dtype"] = dtype_map.get(dtype_value, torch.float32)

        # Handle device_map - ensure it's properly formatted
        if (
            "device_map" in processed
            and isinstance(processed["device_map"], str)
            and processed["device_map"].lower() in ["auto", "balanced", "sequential"]
        ):
            processed["device_map"] = processed["device_map"].lower()

        return processed

    def load_model_and_tokenizer(
        self,
        model_name,
        model_parameters: dict | None = None,
        tokenizer_parameters: dict | None = None,
    ):
        model = None
        tokenizer = None
        model_type = None

        if not model_parameters:
            model_parameters = {}

        if not tokenizer_parameters:
            tokenizer_parameters = {}

        # Preprocess parameters to convert string representations to proper types
        processed_model_params = self._preprocess_model_parameters(model_parameters)
        processed_tokenizer_params = tokenizer_parameters.copy()

        # Temp fix
        processed_tokenizer_params = {"max_length": 512, "truncation": True}

        try:
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name, **processed_tokenizer_params
                )
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name, **processed_model_params
                )
                model_type = "encoder-decoder"
            elif "qwen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **processed_tokenizer_params
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, **processed_model_params
                )
                model_type = "decoder-only"
            else:
                raise ValueError("Unsupported model for this script.")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer for {model_name}: {e}")
            raise ValueError(f"Failed to load model or tokenizer for {model_name}: {e}")

        return model, tokenizer, model_type

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> list[dict]:
        """
        Run SHAP through CAPTUM and return the evaluation results.

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        if not parameters:
            parameters = {}

        if not parameters.get("model_name"):
            logger.error(
                "Model name must be specified in parameters with key 'model_name'."
            )
            raise ValueError(
                "Model name must be specified in parameters with key 'model_name'."
            )
        if not parameters.get("dataset_name"):
            logger.error("Parameter 'dataset_name' is required.")
            raise ValueError("Parameter 'dataset_name' is required.")
        if not parameters.get("dataset"):
            logger.error("Parameter 'dataset' is required.")
            raise ValueError("Parameter 'dataset' is required.")

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        model_name = parameters.get("model_name")
        model, tokenizer, model_type = self.load_model_and_tokenizer(
            model_name,
            model_parameters=parameters.get("model_parameters", {}),
            tokenizer_parameters=parameters.get("tokenizer_parameters", {}),
        )

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        model.eval()

        # Load dataset
        from app.config.registry import get_dataset_loader

        try:
            dataset = get_dataset_loader().load(parameters["dataset_name"], cache=False)
        except Exception as e:
            logger.error(f"Failed to load {parameters['dataset_name']} dataset: {e}")
            raise ValueError(
                f"Failed to load {parameters['dataset_name']} dataset: {e}"
            )

        split = parameters.get("dataset", {}).get("split", "test")
        input_column = parameters.get("dataset", {}).get("input_column", "input")
        num_samples = parameters.get("dataset", {}).get("num_samples", None)
        num_samples = (
            int(num_samples)
            if num_samples is not None
            and isinstance(num_samples, (int, str))
            and str(num_samples).isdigit()
            else None
        )

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        # Use diagnostic_sample input if provided, otherwise use the whole dataset
        diagnostic_sample = parameters.get("diagnostic_sample", {})
        if diagnostic_sample and diagnostic_sample.get("input") is not None:
            inputs_to_process = [diagnostic_sample["input"]]
            results_to_process = [
                r for r in results if r["input"] == diagnostic_sample["input"]
            ]
        else:
            inputs_to_process = dataset[split][input_column]
            results_to_process = results

        attribution_results = []
        for i, (input, output) in enumerate(zip(inputs_to_process, results_to_process)):
            if num_samples is not None and i >= num_samples:
                break

            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )

            logger.info(
                f"{kwargs.get('job_id', '')}: SHAP - Processing sample {i + 1}/{min(len(inputs_to_process), num_samples) if num_samples else len(inputs_to_process)}"
            )

            prediction = output["prediction"]
            if isinstance(prediction, list) and len(prediction) == 1:
                prediction = prediction[0]
            elif isinstance(prediction, list):
                logger.warning(
                    f"Prediction is a list with {len(prediction)} elements. Using first element."
                )
                prediction = prediction[0] if prediction else ""

            if not isinstance(prediction, str):
                logger.error(
                    f"Prediction must be a string, got {type(prediction)}: {prediction}"
                )
                raise TypeError(
                    f"Prediction must be a string, got {type(prediction)}: {prediction}"
                )

            enc_input = tokenizer(input, return_tensors="pt")
            input_tokens = [tokenizer.decode([t]) for t in enc_input.input_ids[0]]

            if model_type == "encoder-decoder":
                # Tokenize once and move tensors to device early
                input_ids = enc_input.input_ids.to(model.device)
                attention_mask = enc_input.attention_mask.to(model.device)

                try:
                    target_ids = tokenizer(
                        prediction, return_tensors="pt", add_special_tokens=False
                    ).input_ids
                except Exception as e:
                    logger.error(f"Failed to tokenize prediction '{prediction}': {e}")
                    raise ValueError(
                        f"Failed to tokenize prediction '{prediction}': {e}"
                    )
                target_tokens = [tokenizer.decode([t]) for t in target_ids[0]]

                attribution_matrix = []

                # Get n_samples and perturbations_per_eval for Shapley value computation
                n_samples = parameters.get("shap_n_samples", 25)
                perturbations_per_eval = parameters.get(
                    "shap_perturbations_per_eval", 1
                )

                # Create wrapper
                wrapped_model = EncoderDecoderAttributionWrapper(
                    model=model, target_ids=target_ids, target_token_idx=0
                )
                shapley = ShapleyValueSampling(wrapped_model)

                for target_idx in range(target_ids.shape[1]):
                    self._check_job_cancellation(
                        kwargs.get("job_id", ""), kwargs.get("job_queue")
                    )

                    logger.info(
                        f"{kwargs.get('job_id', '')}: SHAP - Computing attributions for target token index {target_idx} ({target_tokens[target_idx]})"
                    )

                    wrapped_model.set_target_token_idx(target_idx)
                    try:
                        attributions = shapley.attribute(
                            input_ids,
                            additional_forward_args=(attention_mask,),
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            show_progress=True,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to compute attributions for target token index {target_idx}: {e}"
                        )
                        raise ValueError(
                            f"Failed to compute attributions for target token index {target_idx}: {e}"
                        )

                    attribution_matrix.append(attributions[0].detach().cpu().numpy())

                attribution_matrix = np.array(
                    attribution_matrix
                )  # shape: (target_seq_len, input_seq_len)
            else:
                # Get n_samples and perturbations_per_eval for Shapley value computation
                n_samples = parameters.get("shap_n_samples", 25)
                perturbations_per_eval = parameters.get(
                    "shap_perturbations_per_eval", 1
                )

                shapley = ShapleyValueSampling(model)
                llm_attr = LLMAttribution(shapley, tokenizer)

                # Skip special tokens (EOS, BOS, PAD, etc.)
                skip_tokens = []
                if tokenizer.eos_token_id is not None:
                    skip_tokens.append(tokenizer.eos_token_id)
                if tokenizer.bos_token_id is not None:
                    skip_tokens.append(tokenizer.bos_token_id)
                if tokenizer.pad_token_id is not None:
                    skip_tokens.append(tokenizer.pad_token_id)

                inp = TextTokenInput(input, tokenizer, skip_tokens=skip_tokens)

                try:
                    attr_res = llm_attr.attribute(
                        inp,
                        target=prediction,
                        n_samples=n_samples,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=True,
                    )
                except Exception as e:
                    logger.error(
                        f"{kwargs.get('job_id', '')}: Failed to compute attributions for prediction '{prediction}': {e}"
                    )
                    raise ValueError(
                        f"Failed to compute attributions for prediction '{prediction}': {e}"
                    )
                attribution_matrix = (
                    attr_res.token_attr.detach().cpu().numpy()
                )  # shape: (output_seq_len, input_seq_len)

                # Use input tokens from attribution result (already filtered by Captum)
                input_tokens = [
                    token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()
                    for token in attr_res.input_tokens
                ]

                # Clean target tokens to remove tokenization artifacts
                # Ġ (U+0120) represents space in GPT-2/Qwen tokenizers
                # Ċ (U+010A) represents newline
                # ▁ (U+2581) represents space in SentencePiece tokenizers
                target_tokens = [
                    token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()
                    for token in attr_res.output_tokens
                ]

            try:
                attribution_matrix_2d = attribution_matrix
                # Ensure attribution_matrix is 2D
                if len(attribution_matrix.shape) == 1:
                    # If 1D, reshape to (1, input_seq_len)
                    attribution_matrix_2d = attribution_matrix.reshape(1, -1)
                elif len(attribution_matrix.shape) > 2:
                    # If more than 2D, squeeze extra dimensions
                    attribution_matrix_2d = attribution_matrix.squeeze()

                attribution_results.append(
                    {
                        "input": input,
                        "prediction": prediction,
                        "attribution_matrix": [
                            {
                                "generated_token": output_token,
                                "input_attributions": [
                                    [
                                        input_token,
                                        round(
                                            float(
                                                np.asarray(
                                                    attribution_matrix_2d[j, i],
                                                    dtype=np.float16,
                                                ).item()
                                            ),
                                            4,
                                        ),
                                    ]
                                    for i, input_token in enumerate(input_tokens)
                                ],
                            }
                            for j, output_token in enumerate(target_tokens)
                        ],
                    }
                )
            except Exception as e:
                logger.error(f"Failed to prepare attribution results: {e}")
                raise ValueError(f"Failed to prepare attribution results: {e}")

        logger.info(
            f"{kwargs.get('job_id', '')}: SHAP - Completed processing all samples."
        )

        return attribution_results
