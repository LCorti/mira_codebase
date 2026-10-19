from typing import Any

import evaluate

from app.models.metrics.base_metric import Metric


class BertScoreMetric(Metric):
    def __init__(self):
        """
        Initialize the BertScoreMetric.
        """
        super().__init__(
            name="bertscore",
            description="Computes the BERTScore metric for model predictions.",
        )
        self.metric = evaluate.load("bertscore")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | list[list[str]],
        lang: str = "en",
        model_type: str | None = None,
        num_layers: int | None = None,
        verbose: bool = False,
        idf: bool | dict = False,
        device: str | None = None,
        batch_size: int = 64,
        nthreads: int = 4,
        all_layers: bool = False,
        rescale_with_baseline: bool = False,
        baseline_path: str | None = None,
        use_fast_tokenizer: bool = False,
        **kwargs: Any,
    ) -> dict[str, float] | dict[str, list[float]]:
        """
        Compute the BERTScore metric with native parameter support.
        See more details about BERTScore at: https://huggingface.co/spaces/evaluate-metric/bertscore

        :param predictions: List of predictions to score. Each prediction should be a string.
        :type predictions: List[str]
        :param references: List of reference for each prediction or a list of several references per prediction.
        :type references: Union[List[str], List[List[str]]]
        :param lang: Language of the sentences; required (e.g. 'en'). Either lang or model_type must be specified. Defaults to "en".
        :type lang: str, optional
        :param model_type: BERT specification, default using the suggested model for the target language. Either lang or model_type must be specified.
        :type model_type: str, optional
        :param num_layers: The layer of representation to use, default using the number of layers tuned on WMT16 correlation data.
        :type num_layers: int, optional
        :param verbose: Turn on intermediate status update. Defaults to False.
        :type verbose: bool, optional
        :param idf: Use idf weighting; can also be a precomputed idf_dict. Defaults to False.
        :type idf: Union[bool, dict], optional
        :param device: On which the contextual embedding model will be allocated on. If None, uses cuda:0 if available.
        :type device: str, optional
        :param batch_size: BERTScore processing batch size. Defaults to 64.
        :type batch_size: int, optional
        :param nthreads: Number of threads. Defaults to 4.
        :type nthreads: int, optional
        :param all_layers: Use all layers for representation. Defaults to False.
        :type all_layers: bool, optional
        :param rescale_with_baseline: Rescale BERTScore with pre-computed baseline. Defaults to False.
        :type rescale_with_baseline: bool, optional
        :param baseline_path: Customized baseline file path.
        :type baseline_path: str, optional
        :param use_fast_tokenizer: Use fast tokenizer parameter passed to HF tokenizer. Defaults to False.
        :type use_fast_tokenizer: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Computed BERTScore metric values including precision, recall, f1, and hashcode.
        :rtype: Union[dict[str, float], dict[str, List[float]]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        # Validate that either lang or model_type is specified
        if lang is None and model_type is None:
            raise ValueError(
                "Either 'lang' (e.g. 'en') or 'model_type' (e.g. 'bert-base-uncased') must be specified."
            )

        # Validate batch_size parameter
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        # Validate nthreads parameter
        if not isinstance(nthreads, int) or nthreads < 1:
            raise ValueError("nthreads must be a positive integer.")

        try:
            # Prepare arguments for the metric computation
            compute_args = {
                "predictions": predictions,
                "references": references,
                "verbose": verbose,
                "idf": idf,
                "batch_size": batch_size,
                "nthreads": nthreads,
                "all_layers": all_layers,
                "rescale_with_baseline": rescale_with_baseline,
                "use_fast_tokenizer": use_fast_tokenizer,
                **kwargs,
            }

            # Add optional parameters only if provided
            if lang is not None:
                compute_args["lang"] = lang
            if model_type is not None:
                compute_args["model_type"] = model_type
            if num_layers is not None:
                compute_args["num_layers"] = num_layers
            if device is not None:
                compute_args["device"] = device
            if baseline_path is not None:
                compute_args["baseline_path"] = baseline_path

            result = self.metric.compute(**compute_args)
            return {"bertscore": result}
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
