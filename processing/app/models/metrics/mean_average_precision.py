from sklearn.metrics import average_precision_score
import numpy as np
from typing import List, Union, Any, Optional

from app.services.logging_service import LoggingService

from app.models.metrics.base_metric import Metric

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")

class MeanAveragePrecisionMetric(Metric):
    def __init__(self):
        """
        Initialize the MeanAveragePrecisionMetric.
        Mean Average Precision (MAP) is the mean of Average Precision scores across multiple classes or queries.
        For multi-class problems, it computes the AP for each class and then takes the mean.
        For information retrieval, it computes AP for each query and takes the mean.
        """
        super().__init__(name="mean_average_precision", description="Computes the Mean Average Precision (MAP) metric for multi-class classification or information retrieval tasks.")
        # Note: We don't use evaluate.load() here since MAP is not available in HuggingFace evaluate
        self.metric = None
    
    def compute(self, 
                predictions: Union[List[List[float]], List[float]], 
                references: Union[List[List[int]], List[int]], 
                average: str = "macro",
                sample_weight: Optional[List[float]] = None,
                **kwargs: Any) -> dict[str, float]:
        """
        Compute the Mean Average Precision metric.

        For retrieval (single query):
        - predictions should be a 1D array of relevance scores (higher = more relevant)
        - references should be a 1D array of true relevance labels (1 = relevant, 0 = not relevant)
        - Example: predictions = [0.9, 0.8, 0.7], references = [1, 0, 1]
        - This ranks 3 documents by relevance score, where docs 1 and 3 are actually relevant

        For retrieval (multiple queries):
        - predictions should be a 2D array where each row is the scores for one query's results
        - references should be a 2D array where each row is the relevance labels for one query
        - Example: predictions = [[0.9, 0.8], [0.7, 0.6]], references = [[1, 0], [1, 1]]
        - Query 1: 2 docs with scores [0.9, 0.8], relevance [1, 0] (first doc relevant)
        - Query 2: 2 docs with scores [0.7, 0.6], relevance [1, 1] (both docs relevant)
        
        For multi-class classification:
        - predictions should be a 2D array where each row contains the scores for all classes
        - references should be a 1D array of true class labels
        - Example: predictions = [[0.1, 0.9], [0.8, 0.2]], references = [1, 0]
        
        :param predictions: Prediction scores. For multi-class: shape (n_samples, n_classes). For binary: shape (n_samples,).
        :type predictions: List[List[float]] or List[float]
        :param references: True labels. For multi-class: shape (n_samples,). For multi-label: shape (n_samples, n_classes).
        :type references: List[List[int]] or List[int]
        :param average: The averaging strategy. Options: 'macro', 'micro', 'samples', 'weighted'.
        :type average: str, optional
        :param sample_weight: Sample weights. Defaults to None.
        :type sample_weight: List[float], optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any
        
        :return: Computed Mean Average Precision score.
        :rtype: dict[str, float]
        """
        # Validate inputs
        if not predictions or not references:
            logger.error("Predictions and references must not be empty.")
            raise ValueError("Predictions and references must not be empty.")
        
        # Convert to numpy arrays
        y_score = np.array(predictions)
        y_true = np.array(references)

        # Validate average parameter
        valid_averages = ['macro', 'micro', 'samples', 'weighted']
        if average not in valid_averages:
            logger.error(f"Invalid average: {average}. Valid averages are: {valid_averages}")
            raise ValueError(f"Invalid average: {average}. Valid averages are: {valid_averages}")
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            if len(sample_weight) != len(predictions):
                logger.error("Sample weights must have the same length as predictions and references.")
                raise ValueError("Sample weights must have the same length as predictions and references.")
            if any(w < 0 for w in sample_weight):
                logger.error("Sample weights must be non-negative.")
                raise ValueError("Sample weights must be non-negative.")
            sample_weight = np.array(sample_weight)
        
        try:
            # Handle different input formats
            if y_score.ndim == 1:
                # Binary classification case
                if y_true.ndim != 1:
                    raise ValueError("For binary classification, references should be 1D")
                if len(y_score) != len(y_true):
                    raise ValueError("Predictions and references must have the same length.")
                
                # Compute single AP score
                ap_score = average_precision_score(
                    y_true=y_true,
                    y_score=y_score,
                    sample_weight=sample_weight
                )
                return {"mean_average_precision": float(ap_score)}
            elif y_score.ndim == 2:
                # Multi-class or multi-label case
                if y_true.ndim == 1:
                    # Multi-class case: convert to one-hot encoding
                    n_classes = y_score.shape[1]
                    if len(y_score) != len(y_true):
                        logger.error("Predictions and references must have the same number of samples for multi-class classification.")
                        raise ValueError("Predictions and references must have the same number of samples.")
                    
                    # Convert class labels to one-hot encoding
                    y_true_onehot = np.zeros((len(y_true), n_classes))
                    for i, label in enumerate(y_true):
                        if 0 <= label < n_classes:
                            y_true_onehot[i, label] = 1
                        else:
                            logger.error(f"Label {label} is out of range for {n_classes} classes")
                            raise ValueError(f"Label {label} is out of range for {n_classes} classes")
                    y_true = y_true_onehot
                elif y_true.ndim == 2:
                    # Multi-label case: already in correct format
                    if y_score.shape != y_true.shape:
                        logger.error("Predictions and references must have the same shape for multi-label classification.")
                        raise ValueError("Predictions and references must have the same shape for multi-label classification.")
                else:
                    logger.error("References must be 1D (multi-class) or 2D (multi-label)")
                    raise ValueError("References must be 1D (multi-class) or 2D (multi-label)")
                
                # Compute AP for each class/label and then take the mean
                ap_scores = []
                n_classes = y_score.shape[1]
                
                for class_idx in range(n_classes):
                    y_true_class = y_true[:, class_idx]
                    y_score_class = y_score[:, class_idx]
                    
                    # Skip classes with no positive samples
                    if np.sum(y_true_class) == 0:
                        continue
                    
                    ap = average_precision_score(
                        y_true=y_true_class,
                        y_score=y_score_class,
                        sample_weight=sample_weight
                    )
                    ap_scores.append(ap)
                
                if not ap_scores:
                    logger.error("No valid classes found (all classes have zero positive samples)")
                    raise ValueError("No valid classes found (all classes have zero positive samples)")
                
                # Apply averaging strategy
                if average == 'macro':
                    map_score = np.mean(ap_scores)
                elif average == 'micro':
                    # For micro-averaging, we flatten all predictions and references
                    y_true_flat = y_true.ravel()
                    y_score_flat = y_score.ravel()
                    map_score = average_precision_score(
                        y_true=y_true_flat,
                        y_score=y_score_flat,
                        sample_weight=None  # Sample weights don't make sense for micro-averaging in this context
                    )
                elif average == 'weighted':
                    # Weight by the number of positive samples in each class
                    weights = [np.sum(y_true[:, i]) for i in range(n_classes) if np.sum(y_true[:, i]) > 0]
                    map_score = np.average(ap_scores, weights=weights)
                elif average == 'samples':
                    # Average precision per sample (only makes sense for multi-label)
                    if y_true.ndim != 2:
                        logger.error("'samples' average only supported for multi-label classification")
                        raise ValueError("'samples' average only supported for multi-label classification")
                    sample_aps = []
                    for sample_idx in range(len(y_true)):
                        sample_true = y_true[sample_idx]
                        sample_score = y_score[sample_idx]
                        if np.sum(sample_true) > 0:  # Skip samples with no positive labels
                            ap = average_precision_score(
                                y_true=sample_true,
                                y_score=sample_score
                            )
                            sample_aps.append(ap)
                    map_score = np.mean(sample_aps) if sample_aps else 0.0
                
                return {"mean_average_precision": float(map_score)}
            else:
                logger.error("Predictions must be 1D (binary) or 2D (multi-class/multi-label)")
                raise ValueError("Predictions must be 1D (binary) or 2D (multi-class/multi-label)")
        except Exception as e:
            logger.error(f"Error computing Mean Average Precision: {str(e)}")
            raise RuntimeError(f"Failed to compute {self.name} metric: {str(e)}") from e
