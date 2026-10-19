from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)
from transformers.models.qwen2 import modeling_qwen2

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService
from app.vendor.lxt.efficient import monkey_patch
from app.vendor.lxt.efficient.models.longt5 import LongT5ForConditionalGeneration

logger = LoggingService.setup_module_logger(__name__, log_filename="job.log")


class LxtDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the LXT diagnostic with predefined tests and their corresponding processing functions.
        """
        super().__init__(
            name="lxt",
            description="Diagnostic for evaluating heuristic use from the LXT method.",
        )

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> list[dict]:
        """
        Run the script from the "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers" paper.

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        if not parameters:
            parameters = {}
        if not parameters.get("model_name"):
            logger.error("Parameter 'model_name' is required.")
            raise ValueError("Parameter 'model_name' is required.")
        if not parameters.get("dataset_name"):
            logger.error("Parameter 'dataset_name' is required.")
            raise ValueError("Parameter 'dataset_name' is required.")
        if not parameters.get("dataset"):
            logger.error("Parameter 'dataset' is required.")
            raise ValueError("Parameter 'dataset' is required.")

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

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

        task_parameters = parameters.get("task_parameters", {})
        system_message = task_parameters.get("system_message", "")

        # Get inference parameters from model configuration
        model_inference_params = parameters.get("model", {}).get("inference", {})

        decoding_strategy = model_inference_params.get("decoding_strategy", "greedy")
        # repetition_penalty = model_inference_params.get("repetition_penalty", 1.0)
        # max_new_tokens = model_inference_params.get("max_new_tokens", 256)
        # min_length = model_inference_params.get("min_length", 2)
        # top_p = model_inference_params.get("top_p", 0.95)
        # do_sample = model_inference_params.get("do_sample", False)

        self._check_job_cancellation(kwargs.get("job_id", ""), kwargs.get("job_queue"))

        model = None
        tokenizer = None
        model_name = parameters["model_name"]
        model_type = None
        max_input_length = 0
        max_output_length = 0
        try:
            logger.info(f"Loading model and tokenizer for {model_name}...")
            if "t5" in model_name.lower():
                model = LongT5ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    use_safetensors=False,
                    torch_dtype=torch.bfloat16,
                )
                model_type = "t5"
                max_input_length = 16384
                max_output_length = 1024
            elif "qwen" in model_name.lower():
                monkey_patch(modeling_qwen2, verbose=True)
                model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.bfloat16
                )
                model_type = "qwen"
                max_input_length = 32768
                max_output_length = 32768
            else:
                logger.error(
                    "Unsupported model for LXT diagnostic. Supported models are T5 and Qwen."
                )
                raise ValueError(
                    "Unsupported model for LXT diagnostic. Supported models are T5 and Qwen."
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

        # deactivate gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = False

        # Use diagnostic_sample input if provided, otherwise use the whole dataset
        diagnostic_sample = parameters.get("diagnostic_sample", {})
        inputs_to_process = []
        if diagnostic_sample and diagnostic_sample.get("input"):
            inputs_to_process = [diagnostic_sample["input"]]
        else:
            inputs_to_process = dataset[split][input_column]

        num_samples = parameters.get("dataset", {}).get("num_samples", None)
        num_samples = (
            int(num_samples)
            if num_samples is not None
            and isinstance(num_samples, (int, str))
            and str(num_samples).isdigit()
            else None
        )

        # Temp fix
        max_input_length = 4096

        attribution_results = []
        for i, input in enumerate(inputs_to_process):
            self._check_job_cancellation(
                kwargs.get("job_id", ""), kwargs.get("job_queue")
            )

            if num_samples is not None and i >= num_samples:
                break
            logger.info(
                f"{kwargs.get('job_id', '')}: LXT - Processing input: {i + 1}/{num_samples if num_samples is not None else len(inputs_to_process)}"
            )

            enc_prompt = input
            dec_prompt = "Answer: "
            if system_message and system_message != "<input>":
                dec_prompt = (
                    system_message.split("<input>")[-1].strip()
                )  # Get decoder prompt part after <input>; empty string if not present
                enc_prompt = (
                    system_message.replace("<input>", input)
                    if system_message and "<input>" in system_message
                    else f"{system_message} {input}"
                )  # Check for placeholder & replace, otherwise append

            enc_input_ids = tokenizer(
                enc_prompt,
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(model.device)
            enc_input_embeds = model.get_input_embeddings()(enc_input_ids)

            dec_input_ids = None
            if model_type == "t5":
                dec_input_ids = tokenizer(
                    dec_prompt,
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(model.device)
                dec_input_ids = model._shift_right(dec_input_ids)
                # initial_dec_length = dec_input_ids.shape[-1]

            # Add starting token
            if (
                model_type == "t5"
                and dec_input_ids is not None
                and dec_input_ids[0, 0] != model.config.decoder_start_token_id
            ):
                start_token = torch.tensor(
                    [[model.config.decoder_start_token_id]], device=model.device
                )
                dec_input_ids = torch.cat([start_token, dec_input_ids], dim=-1)

            max_new_tokens = 256
            init_enc_length = enc_input_ids.shape[-1]
            input_ids_seq_length = (
                dec_input_ids.shape[-1]
                if dec_input_ids is not None
                else enc_input_ids.shape[-1]
            )
            min_length_absolute = input_ids_seq_length + 2
            max_length = max_new_tokens + input_ids_seq_length
            max_length = min(max_output_length, max_length)
            eos_token_id = tokenizer.eos_token_id

            logits_processor = LogitsProcessorList(
                [
                    RepetitionPenaltyLogitsProcessor(penalty=1.0),
                    MinLengthLogitsProcessor(
                        min_length=min_length_absolute, eos_token_id=eos_token_id
                    ),
                    ForcedEOSTokenLogitsProcessor(
                        max_length=max_length, eos_token_id=eos_token_id
                    ),
                ]
            )
            # Use do_sample parameter instead of decoding_strategy if available
            if decoding_strategy == "sample":
                logits_processor.append(TopPLogitsWarper(top_p=0.95))

            # initial_enc_length = enc_input_ids.shape[-1]

            original_enc_tokens = [
                tokenizer.decode([token_id], skip_special_tokens=False).strip()
                or tokenizer.convert_ids_to_tokens([token_id])[0]
                for token_id in enc_input_ids[0]
            ]
            relevance = torch.zeros(len(original_enc_tokens))

            curr_len = (
                dec_input_ids.shape[-1]
                if dec_input_ids is not None
                else enc_input_ids.shape[-1]
            )
            relevance_scores = []
            next_token = None

            while curr_len <= max_length and next_token != eos_token_id:
                self._check_job_cancellation(
                    kwargs.get("job_id", ""), kwargs.get("job_queue")
                )

                # logger.info(f"{kwargs.get('job_id', '')}: curr_len={curr_len}, next_token={next_token}, eos_token_id={eos_token_id}")

                dec_input_embeds = None
                current_input_embeds = None

                if model_type == "t5":
                    dec_input_embeds = model.get_input_embeddings()(dec_input_ids)
                    current_input_embeds = enc_input_embeds.requires_grad_()

                    output_logits = model(
                        inputs_embeds=current_input_embeds,
                        decoder_inputs_embeds=dec_input_embeds.requires_grad_(),
                        use_cache=False,
                    ).logits

                    raw_next_logits = output_logits[0, -1, :].cpu()
                    dec_input_ids_copy = dec_input_ids.cpu()
                    next_token_scores = logits_processor(
                        dec_input_ids_copy, raw_next_logits.unsqueeze(0)
                    ).to(model.device)

                elif model_type == "qwen":
                    current_input_embeds = model.get_input_embeddings()(
                        enc_input_ids
                    ).requires_grad_()
                    output_logits = model(
                        inputs_embeds=current_input_embeds,
                        use_cache=False,
                    ).logits

                    raw_next_logits = output_logits[0, -1, :].cpu()
                    input_ids_copy = enc_input_ids.cpu()
                    next_token_scores = logits_processor(
                        input_ids_copy, raw_next_logits.unsqueeze(0)
                    ).to(model.device)
                else:
                    raise ValueError("Unsupported model type for LXT diagnostic.")

                k = 1
                if decoding_strategy == "sample":
                    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                    topk_indices = torch.multinomial(probs, num_samples=k)
                    max_raw_logits = output_logits[0, -1, topk_indices[0].item()]
                else:  # greedy decoding
                    _, topk_indices = torch.topk(next_token_scores, k=k, dim=-1)
                    max_raw_logits, _ = torch.max(output_logits[0, -1, :], dim=-1)

                max_raw_logits.backward()

                relevance = (
                    (current_input_embeds.grad * current_input_embeds)
                    .float()
                    .sum(-1)
                    .detach()
                    .cpu()[0]
                )

                # Normalize relevance between [-1, 1] for plotting
                relevance = relevance / relevance.abs().max()

                # Clear out the gradients after computing current relevance score
                model.zero_grad()
                current_input_embeds.grad = None
                if dec_input_embeds is not None:
                    dec_input_embeds.grad = None

                curr_len += 1
                if model_type == "t5":
                    dec_input_ids = torch.cat([dec_input_ids, topk_indices], dim=-1)
                elif model_type == "qwen":
                    enc_input_ids = torch.cat([enc_input_ids, topk_indices], dim=-1)
                else:
                    raise ValueError("Unsupported model type for LXT diagnostic.")
                next_token = topk_indices[0].item()
                decoded_token = tokenizer.decode(topk_indices[0])
                # logger.info(f"LXT: Generated token: '{decoded_token}' (ID: {next_token})")

                relevance_half = relevance.half()
                tokens_relevance = list(
                    zip(
                        original_enc_tokens,
                        [round(float(x), 4) for x in relevance_half],
                    )
                )
                relevance_scores.append(
                    {
                        "generated_token": decoded_token,
                        "token_relevance": tokens_relevance,
                    }
                )

            prediction = ""
            if model_type == "t5":
                prediction = tokenizer.decode(
                    dec_input_ids[0], skip_special_tokens=True
                )
            elif model_type == "qwen":
                prediction = tokenizer.decode(
                    enc_input_ids[0][init_enc_length:], skip_special_tokens=True
                )

                if prediction.startswith("user"):  # trim leading 'user' if present
                    prediction = prediction.replace("user", "", 1).strip()
            else:
                raise ValueError("Unsupported model type for LXT diagnostic.")

            # logger.info(f"LXT: Final prediction: {prediction}")

            # Store the result in SHAP-compatible format
            attribution_results.append(
                {
                    "input": enc_prompt,
                    "prediction": prediction,
                    "attribution_matrix": relevance_scores,
                }
            )

        return attribution_results
