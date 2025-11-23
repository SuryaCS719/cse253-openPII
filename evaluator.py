"""
OpenPII Watcher: Evaluation Framework
Implements precision, recall, F1 score with micro/macro averaging
Supports ground truth comparison and baseline evaluation
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for PII detection"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predicted: int
    total_actual: int


@dataclass
class GroundTruth:
    """Ground truth labels for a test sample"""
    sample_id: str
    text: str
    labels: Dict[str, Set[str]]  # pii_type -> set of actual values


class PIIEvaluator:
    """
    Comprehensive evaluation framework for PII detection
    Supports multiple averaging strategies and baseline comparison
    """
    
    def __init__(self):
        self.results_history = []
    
    def calculate_metrics(
        self,
        predicted: Set[str],
        actual: Set[str]
    ) -> EvaluationMetrics:
        """
        Calculate precision, recall, F1 for a single PII type
        
        Args:
            predicted: Set of predicted PII values
            actual: Set of actual PII values (ground truth)
        
        Returns:
            EvaluationMetrics object
        """
        true_positives = len(predicted & actual)
        false_positives = len(predicted - actual)
        false_negatives = len(actual - predicted)
        
        total_predicted = len(predicted)
        total_actual = len(actual)
        
        # Precision: What fraction of predictions were correct?
        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        
        # Recall: What fraction of actual PII was detected?
        recall = true_positives / total_actual if total_actual > 0 else 0.0
        
        # F1: Harmonic mean of precision and recall
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_predicted=total_predicted,
            total_actual=total_actual
        )
    
    def evaluate_single_sample(
        self,
        predicted_results: Dict[str, Set[str]],
        ground_truth: GroundTruth
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate detection results for a single sample
        
        Args:
            predicted_results: Dict of pii_type -> detected values
            ground_truth: GroundTruth object with actual labels
        
        Returns:
            Dict of pii_type -> EvaluationMetrics
        """
        metrics_by_type = {}
        
        for pii_type in ground_truth.labels.keys():
            predicted = predicted_results.get(pii_type, set())
            actual = ground_truth.labels[pii_type]
            
            metrics = self.calculate_metrics(predicted, actual)
            metrics_by_type[pii_type] = metrics
        
        return metrics_by_type
    
    def micro_average(
        self,
        all_metrics: List[Dict[str, EvaluationMetrics]]
    ) -> Dict[str, EvaluationMetrics]:
        """
        Micro-averaging: Pool all instances together before calculating metrics
        Each instance is weighted equally
        
        Good for: Overall system performance across all PII types
        """
        # Aggregate counts by PII type
        aggregated = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'total_predicted': 0, 'total_actual': 0
        })
        
        for sample_metrics in all_metrics:
            for pii_type, metrics in sample_metrics.items():
                aggregated[pii_type]['tp'] += metrics.true_positives
                aggregated[pii_type]['fp'] += metrics.false_positives
                aggregated[pii_type]['fn'] += metrics.false_negatives
                aggregated[pii_type]['total_predicted'] += metrics.total_predicted
                aggregated[pii_type]['total_actual'] += metrics.total_actual
        
        # Calculate metrics from aggregated counts
        micro_metrics = {}
        for pii_type, counts in aggregated.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            micro_metrics[pii_type] = EvaluationMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                total_predicted=counts['total_predicted'],
                total_actual=counts['total_actual']
            )
        
        return micro_metrics
    
    def macro_average(
        self,
        all_metrics: List[Dict[str, EvaluationMetrics]]
    ) -> Dict[str, EvaluationMetrics]:
        """
        Macro-averaging: Calculate metrics per sample, then average
        Each sample is weighted equally
        
        Good for: Understanding performance across different document types
        """
        # Collect metrics by PII type across samples
        by_type = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
        
        for sample_metrics in all_metrics:
            for pii_type, metrics in sample_metrics.items():
                by_type[pii_type]['precision'].append(metrics.precision)
                by_type[pii_type]['recall'].append(metrics.recall)
                by_type[pii_type]['f1'].append(metrics.f1_score)
        
        # Average across samples
        macro_metrics = {}
        for pii_type, values in by_type.items():
            n = len(values['precision'])
            avg_precision = sum(values['precision']) / n if n > 0 else 0.0
            avg_recall = sum(values['recall']) / n if n > 0 else 0.0
            avg_f1 = sum(values['f1']) / n if n > 0 else 0.0
            
            macro_metrics[pii_type] = EvaluationMetrics(
                precision=avg_precision,
                recall=avg_recall,
                f1_score=avg_f1,
                true_positives=0,  # Not meaningful in macro average
                false_positives=0,
                false_negatives=0,
                total_predicted=0,
                total_actual=0
            )
        
        return macro_metrics
    
    def evaluate_dataset(
        self,
        detector,
        test_dataset: List[GroundTruth]
    ) -> Dict:
        """
        Evaluate detector on full test dataset
        
        Returns comprehensive evaluation report with micro/macro metrics
        """
        all_sample_metrics = []
        
        for ground_truth in test_dataset:
            # Get detector predictions
            detected = detector.detect_all(ground_truth.text)
            predicted_values = {
                pii_type: set(match.value for match in matches)
                for pii_type, matches in detected.items()
            }
            
            # Evaluate this sample
            sample_metrics = self.evaluate_single_sample(predicted_values, ground_truth)
            all_sample_metrics.append(sample_metrics)
        
        # Calculate aggregate metrics
        micro_metrics = self.micro_average(all_sample_metrics)
        macro_metrics = self.macro_average(all_sample_metrics)
        
        return {
            'micro_average': micro_metrics,
            'macro_average': macro_metrics,
            'per_sample': all_sample_metrics,
            'num_samples': len(test_dataset)
        }
    
    def format_metrics_table(self, metrics: Dict[str, EvaluationMetrics]) -> str:
        """Format metrics as ASCII table for display"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{'PII Type':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'TP':>6} {'FP':>6} {'FN':>6}")
        lines.append("=" * 80)
        
        for pii_type, m in metrics.items():
            lines.append(
                f"{pii_type:<15} {m.precision:>12.3f} {m.recall:>12.3f} {m.f1_score:>12.3f} "
                f"{m.true_positives:>6} {m.false_positives:>6} {m.false_negatives:>6}"
            )
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def compare_to_baseline(
        self,
        our_metrics: Dict[str, EvaluationMetrics],
        baseline_metrics: Dict[str, EvaluationMetrics]
    ) -> Dict:
        """
        Compare our approach to baseline
        
        Baseline: Simple regex patterns without optimizations
        """
        comparison = {}
        
        for pii_type in our_metrics.keys():
            our = our_metrics[pii_type]
            baseline = baseline_metrics.get(pii_type)
            
            if baseline:
                comparison[pii_type] = {
                    'our_f1': our.f1_score,
                    'baseline_f1': baseline.f1_score,
                    'improvement': our.f1_score - baseline.f1_score,
                    'improvement_pct': ((our.f1_score - baseline.f1_score) / baseline.f1_score * 100) if baseline.f1_score > 0 else 0
                }
        
        return comparison
    
    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file"""
        
        def convert_to_dict(obj):
            """Recursively convert objects to dictionaries"""
            if isinstance(obj, EvaluationMetrics):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        serializable = convert_to_dict(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Example usage
    evaluator = PIIEvaluator()
    
    # Simulate some predictions and ground truth
    predicted = {'john@example.com', 'mary@test.com', 'wrong@email.com'}
    actual = {'john@example.com', 'mary@test.com', 'missed@email.com'}
    
    metrics = evaluator.calculate_metrics(predicted, actual)
    
    print("=== Example Metrics Calculation ===")
    print(f"Predicted: {predicted}")
    print(f"Actual: {actual}")
    print(f"\nResults:")
    print(f"  Precision: {metrics.precision:.3f} ({metrics.true_positives}/{metrics.total_predicted} correct)")
    print(f"  Recall: {metrics.recall:.3f} ({metrics.true_positives}/{metrics.total_actual} detected)")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, FN: {metrics.false_negatives}")

