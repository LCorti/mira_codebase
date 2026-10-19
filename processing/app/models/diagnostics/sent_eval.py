from typing import Any

import numpy as np
import torch

from app.models.diagnostics.base_diagnostic import Diagnostic
from app.services.logging_service import LoggingService
from app.vendor import senteval

logger = LoggingService.setup_module_logger(__name__, log_filename="diagnostic.log")

PATH_TO_DATA = "app/data/SentEval"

PARAMS_SENTEVAL = {"task_path": PATH_TO_DATA, "usepytorch": False, "kfold": 2}
PARAMS_SENTEVAL["classifier"] = {
    "nhid": 0,
    "optim": "rmsprop",
    "batch_size": 128,
    "tenacity": 2,
    "epoch_size": 2,
}

DEFAULT_TASKS = ["CR", "MRPC"]

TASK_MAP = {
    "MR": "Sentiment Analysis of Movie Reviews",
    "CR": "Sentiment Analysis of Product Reviews",
    "SUBJ": "Subjectivity Status",
    "MPQA": "Opinion Polarity",
    # "SST2": "Stanford Sentiment Treebank (binary)", # 67k
    "SST5": "Stanford Sentiment Treebank (5-point)",
    "TREC": "Question-type Classification",
    "SICKEntailment": "Semantic Entailment",
    # "SNLI": "Stanford Natural Language Inference", # 550k
    "MRPC": "Microsoft Research Paraphrase Corpus",
    # "STS12": "Semantic Text Similarity (2012 ed.)", # not included
    # "STS13": "Semantic Text Similarity (2013 ed.)", # not included
    # "STS14": "Semantic Text Similarity (2014 ed.)", # not included
    # "STS15": "Semantic Text Similarity (2015 ed.)", # not included
    # "STS16": "Semantic Text Similarity (2016 ed.)", # not included
    "STSBenchmark": "Semantic Textual Similarity Benchmark",
    "SICKRelatedness": "Semantic Relatedness",
    # These are all very long, disable for user study
    # "Length": "Length Prediction",
    # "WordContent": "Word Content Analysis",
    # "Depth": "Tree Depth Prediction",
    # "TopConstituents": "Top Constituents Prediction",
    # "BigramShift": "Word Order Analysis",
    # "Tense": "Verb Tense Prediction",
    # "SubjNumber": "Subject Number Prediction",
    # "ObjNumber": "Object Number Prediction",
    # "OddManOut": "Semantic Odd Man Out",
    # "CoordinationInversion": "Coordination Inversion"
}

MODEL_NAME = None
MODEL_PARAMETERS = {}
TOKENIZER_PARAMETERS = {}


def prepare(params, samples):
    params.encoder = ModelLoader(MODEL_NAME, MODEL_PARAMETERS, TOKENIZER_PARAMETERS)


def batcher(params, batch):
    sentences = [" ".join(sent) if sent != [] else "." for sent in batch]
    return params.encoder.encode(sentences)


class ModelLoader:
    def __init__(
        self,
        model_name: str,
        model_parameters: dict | None = None,
        tokenizer_parameters: dict | None = None,
    ):
        """
        Initialize model and tokenizer.

        Args:
            :param model_name: Name of the pre-trained FLAN-T5 model
            :type model_name: str
            :param model_parameters: Additional parameters for model loading
            :type model_parameters: dict
            :param tokenizer_parameters: Additional parameters for tokenizer loading
            :type tokenizer_parameters: dict
        """

        from app.config.registry import get_model_loader

        if not model_parameters:
            model_parameters = {}
        if not tokenizer_parameters:
            tokenizer_parameters = {}

        # Temp fix
        tokenizer_parameters = {"max_length": 4096, "truncation": True}
        self.model, self.tokenizer = get_model_loader().load(
            model_name, model_parameters, tokenizer_parameters, cache=False
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=4):
        """
        Encode sentences using model encoder with aggressive memory optimizations.
        Supports both encoder-decoder models (FLAN-T5) and decoder-only models (Qwen2, GPT).
        """
        embeddings = []

        # Check if model has encoder (encoder-decoder) or is decoder-only
        has_encoder = hasattr(self.model, "encoder")

        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)

                # Get hidden states based on model architecture
                if has_encoder:
                    # Encoder-decoder model (e.g., FLAN-T5)
                    last_hidden_state = self.model.encoder(**inputs).last_hidden_state
                else:
                    # Decoder-only model (e.g., Qwen2, GPT)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]

                # Use mean pooling over the sequence dimension
                # last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_embeddings = last_hidden_state * attention_mask
                summed_embeddings = masked_embeddings.sum(dim=1)
                seq_lengths = attention_mask.sum(dim=1)
                mean_embeddings = summed_embeddings / seq_lengths

                embeddings.append(mean_embeddings.cpu().float().numpy())

                # Delete intermediate tensors to free memory
                del inputs, last_hidden_state, attention_mask, masked_embeddings
                del summed_embeddings, seq_lengths, mean_embeddings

                # Clear GPU cache after each batch to prevent memory buildup
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return np.vstack(embeddings)


class SentEvalDiagnostic(Diagnostic):
    def __init__(self):
        """
        Initialize the SentEval diagnostic.
        """
        super().__init__(
            name="sent_eval",
            description="Diagnostic for evaluating sentence embeddings.",
        )

    def compute(
        self, results: list[dict], parameters: dict | None = None, **kwargs: Any
    ) -> dict:
        """
        Run the tasks from SentEval on the provided sentence embeddings (https://github.com/facebookresearch/SentEval).

        :param results: List of objects per prediction containing 'input', 'prediction', 'reference', and 'logit_id' keys.
        :param parameters: Additional parameters for the diagnostic computation, such as 'test' to specify which test to run.
        :param kwargs: Additional keyword arguments, including 'tokenizer' for decoding token IDs.
        """

        if not parameters:
            parameters = {}

        model_name = parameters.get("model_name")
        if not model_name:
            logger.error(
                "Model name must be specified in parameters with key 'model_name'."
            )
            raise ValueError(
                "Model name must be specified in parameters with key 'model_name'."
            )

        model_parameters = parameters.get("model_parameters", {})
        tokenizer_parameters = parameters.get("tokenizer_parameters", {})

        global MODEL_NAME, MODEL_PARAMETERS, TOKENIZER_PARAMETERS
        MODEL_NAME = model_name
        MODEL_PARAMETERS = model_parameters
        TOKENIZER_PARAMETERS = tokenizer_parameters

        tasks = parameters.get("tasks", DEFAULT_TASKS)
        if not all(task in TASK_MAP for task in tasks):
            logger.error(
                f"Invalid tasks specified. Available tasks are: {list(TASK_MAP.keys())}"
            )
            raise ValueError(
                f"Invalid tasks specified. Available tasks are: {list(TASK_MAP.keys())}"
            )
        sent_eval_params = parameters.get("senteval_params", PARAMS_SENTEVAL)

        se = senteval.engine.SE(sent_eval_params, batcher, prepare)
        eval_results = se.eval(
            tasks
        )  # Dict with task names as keys and results as values

        report = ""
        for task, result in eval_results.items():
            # Header
            report += "=" * 50
            report += f"\n{task}: {TASK_MAP[task]}"
            # Parse individual fiels
            for k, v in result.items():
                report += f"\n\t- {k}: {v.item() if isinstance(v, np.generic) else v}"
            report += "\n"
        return report
