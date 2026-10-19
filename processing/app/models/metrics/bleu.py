import evaluate
from typing import List, Union, Any

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename='metrics.log')

class BleuMetric(Metric):
    def __init__(self):
        """
        Initialize the BleuMetric.
        """
        super().__init__(name="bleu", description="Computes the BLEU metric for model predictions.")
        self.metric = evaluate.load("bleu")
    
    def compute(self, 
                predictions: List[str], 
                references: Union[List[str], List[List[str]]], 
                tokenizer=None,
                max_order: int = 4,
                smooth: bool = False,
                **kwargs: Any) -> Union[dict[str, float], dict[str, List[float]]]:
        """
        Compute the BLEU metric with native parameter support.
        See more details about BLEU at: https://huggingface.co/spaces/evaluate-metric/bleu
        
        :param predictions: List of predictions to score. Each prediction should be a string with tokens separated by spaces.
        :type predictions: List[str]
        :param references: List of reference for each prediction or a list of several references per prediction.
        :type references: Union[List[str], List[List[str]]]
        :param tokenizer: Approach used for tokenizing predictions and references. The default tokenizer is tokenizer_13a, a minimal tokenization approach that is equivalent to mteval-v13a, used by WMT. This can be replaced by any function that takes a string as input and returns a list of tokens as output.
        :type tokenizer: callable, optional
        :param max_order: Maximum n-gram order to use when computing BLEU score. Defaults to 4.
        :type max_order: int, optional
        :param smooth: Whether or not to apply Lin et al. 2004 smoothing. Defaults to False.
        :type smooth: bool, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any
        
        :return: Computed BLEU metric values including bleu score, precisions, brevity penalty, length ratio, translation length, and reference length.
        :rtype: Union[dict[str, float], dict[str, List[float]]]
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        
        # Validate max_order parameter
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError("max_order must be a positive integer.")
        
        # Validate smooth parameter
        if not isinstance(smooth, bool):
            raise ValueError("smooth must be a boolean value.")
        
        try:
            # Prepare arguments for the metric computation
            compute_args = {
                "predictions": predictions,
                "references": references,
                "max_order": max_order,
                "smooth": smooth,
                **kwargs
            }
            
            # Add tokenizer only if provided
            if tokenizer is not None:
                compute_args["tokenizer"] = tokenizer
            
            result = self.metric.compute(**compute_args)
            return {"bleu": result}
        except Exception as e:
            logger.error(f"Error computing BLEU metric: {e}")
            return {}
