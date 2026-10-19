from abc import ABC
from typing import List, Any, Union

class Metric(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metric = None  # Subclasses must set this

    def compute(self, predictions: List[Any], references: Union[List[str], List[int], List[List[str]]], **kwargs: Any) -> Union[dict[str, float], dict[str, List[float]]]:
        """
        Compute the metric based on the provided outputs and references.
        
        :param predictions: List of model outputs (predictions).
        :type predictions: List[str]
        :param references: List of reference values to compare against.
        :type references: Union[List[str], List[int]]
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any
        
        :return: Computed metric value. Can be aggregated (dict[str, float]) or per-sample (dict[str, List[float]]).
        :rtype: Union[dict[str, float], dict[str, List[float]]]
        """
        
        if self.metric is None: 
            raise NotImplementedError("Subclasses must set self.metric in __init__")
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        
        try:
            return self.metric.compute(predictions=predictions, references=references, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to compute {self.name} metric: {str(e)}") from e
    
    def parse_output(self, computed_metric: dict, num_samples: int) -> List[dict]:
        """
        Parse the computed metric output into a standardized per-sample format.
        
        :param computed_metric: The raw output from the metric computation.
        :type computed_metric: dict
        :param num_samples: The number of samples in the dataset.
        :type num_samples: int
        
        :return: List of metric dictionaries, one per sample.
        :rtype: List[dict]
        """
        parsed_metrics = []
        
        # Check if the metric output contains per-sample values (lists) or aggregated values (floats)
        sample_metrics = {}
        aggregated_metrics = {}
        
        for metric_name, metric_value in computed_metric.items():
            if isinstance(metric_value, list):
                # Per-sample metric values
                if len(metric_value) != num_samples:
                    raise ValueError(f"Metric {metric_name} has {len(metric_value)} values but expected {num_samples}")
                sample_metrics[metric_name] = metric_value
            elif isinstance(metric_value, float) or isinstance(metric_value, int):
                # Aggregated metric value (same for all samples)
                aggregated_metrics[metric_name] = metric_value
            else:
                raise ValueError(f"Unsupported metric value type for {metric_name}: {type(metric_value)}. Expected list or float.")
        
        # Create per-sample metric dictionaries
        for i in range(num_samples):
            sample_dict = {}
            
            # Add per-sample metrics
            for metric_name, values in sample_metrics.items():
                sample_dict[metric_name] = values[i]
            
            # Add aggregated metrics (same value for all samples)
            for metric_name, value in aggregated_metrics.items():
                sample_dict[metric_name] = value
            
            parsed_metrics.append(sample_dict)
        
        return parsed_metrics
    
    def __str__(self):
        return f"{self.name} Metric: {self.description}"
    
    def __repr__(self):
        return f"Metric(name={self.name}, description={self.description})"