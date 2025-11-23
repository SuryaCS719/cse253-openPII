"""
OpenPII Watcher: Experiment Runner
Run comprehensive experiments and generate evaluation results
"""

from pii_detector import PIIDetector
from test_data_generator import TestDataGenerator
from evaluator import PIIEvaluator, GroundTruth
import json
import matplotlib.pyplot as plt
import numpy as np


class BaselineDetector:
    """Simple baseline detector with basic patterns for comparison"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\S+@\S+\.\S+',  # Very basic email pattern
            'phone': r'\d{3}-\d{3}-\d{4}',  # Only dash-separated format
            'name': r'[A-Z][a-z]+ [A-Z][a-z]+',  # Only simple First Last
        }
        
        import re
        self.compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self.patterns.items()
        }
    
    def detect_all(self, text):
        results = {}
        for pii_type in ['email', 'phone', 'name', 'address', 'ssn', 'credit_card']:
            if pii_type in self.compiled_patterns:
                matches = []
                for match in self.compiled_patterns[pii_type].finditer(text):
                    from pii_detector import PIIMatch
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                results[pii_type] = matches
            else:
                results[pii_type] = []
        return results


def run_experiments():
    """Run comprehensive experiments"""
    
    print("=" * 80)
    print("OpenPII Watcher - Experimental Evaluation")
    print("=" * 80)
    print()
    
    # Initialize components
    print("[1/6] Initializing components...")
    detector = PIIDetector()
    baseline_detector = BaselineDetector()
    generator = TestDataGenerator()
    evaluator = PIIEvaluator()
    
    # Generate test dataset
    print("[2/6] Generating synthetic test dataset...")
    dataset = generator.generate_test_dataset(num_samples=20)
    print(f"  Generated {len(dataset)} test samples")
    
    # Save dataset
    generator.save_dataset(dataset, 'test_dataset.json')
    
    # Evaluate main detector
    print("[3/6] Evaluating main detection system...")
    main_results = evaluator.evaluate_dataset(detector, dataset)
    
    # Evaluate baseline
    print("[4/6] Evaluating baseline system...")
    baseline_results = evaluator.evaluate_dataset(baseline_detector, dataset)
    
    # Save results
    print("[5/6] Saving results...")
    evaluator.save_results(main_results, 'main_results.json')
    evaluator.save_results(baseline_results, 'baseline_results.json')
    
    # Print metrics
    print("\n" + "=" * 80)
    print("MAIN SYSTEM - MICRO AVERAGE METRICS")
    print(evaluator.format_metrics_table(main_results['micro_average']))
    
    print("\n" + "=" * 80)
    print("MAIN SYSTEM - MACRO AVERAGE METRICS")
    print(evaluator.format_metrics_table(main_results['macro_average']))
    
    print("\n" + "=" * 80)
    print("BASELINE SYSTEM - MICRO AVERAGE METRICS")
    print(evaluator.format_metrics_table(baseline_results['micro_average']))
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Our System vs Baseline")
    print("=" * 80)
    
    comparison = evaluator.compare_to_baseline(
        main_results['micro_average'],
        baseline_results['micro_average']
    )
    
    for pii_type, comp in comparison.items():
        print(f"\n{pii_type.upper()}:")
        print(f"  Our F1: {comp['our_f1']:.3f}")
        print(f"  Baseline F1: {comp['baseline_f1']:.3f}")
        print(f"  Improvement: {comp['improvement']:+.3f} ({comp['improvement_pct']:+.1f}%)")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    generate_visualizations(main_results, baseline_results)
    
    print("\n" + "=" * 80)
    print("Experiments completed successfully!")
    print("=" * 80)
    
    return main_results, baseline_results


def generate_visualizations(main_results, baseline_results):
    """Generate visualization plots"""
    
    # Extract metrics
    pii_types = list(main_results['micro_average'].keys())
    main_f1 = [main_results['micro_average'][t].f1_score for t in pii_types]
    baseline_f1 = [baseline_results['micro_average'][t].f1_score for t in pii_types]
    main_precision = [main_results['micro_average'][t].precision for t in pii_types]
    main_recall = [main_results['micro_average'][t].recall for t in pii_types]
    
    # Clean up PII type names for display
    display_names = [t.replace('_', ' ').title() for t in pii_types]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OpenPII Watcher: Detection Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: F1-Score Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(pii_types))
    width = 0.35
    
    ax1.bar(x - width/2, main_f1, width, label='Our System', color='#2563eb', alpha=0.8)
    ax1.bar(x + width/2, baseline_f1, width, label='Baseline', color='#94a3b8', alpha=0.8)
    
    ax1.set_xlabel('PII Type', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score Comparison: Our System vs Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Plot 2: Precision vs Recall
    ax2 = axes[0, 1]
    
    ax2.scatter(main_precision, main_recall, s=200, c='#2563eb', alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(display_names):
        ax2.annotate(name, (main_precision[i], main_recall[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Precision', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Precision vs Recall Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
    ax2.legend()
    
    # Plot 3: Detailed Metrics Heatmap
    ax3 = axes[1, 0]
    
    metrics_matrix = np.array([main_precision, main_recall, main_f1])
    
    im = ax3.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax3.set_xticks(np.arange(len(pii_types)))
    ax3.set_yticks(np.arange(3))
    ax3.set_xticklabels(display_names, rotation=45, ha='right')
    ax3.set_yticklabels(['Precision', 'Recall', 'F1-Score'])
    ax3.set_title('Performance Metrics Heatmap')
    
    # Add text annotations
    for i in range(3):
        for j in range(len(pii_types)):
            text = ax3.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='Score')
    
    # Plot 4: Improvement over Baseline
    ax4 = axes[1, 1]
    
    improvements = [(main_f1[i] - baseline_f1[i]) * 100 for i in range(len(pii_types))]
    colors = ['#10b981' if imp > 0 else '#ef4444' for imp in improvements]
    
    ax4.barh(display_names, improvements, color=colors, alpha=0.7)
    ax4.set_xlabel('F1-Score Improvement (%)', fontweight='bold')
    ax4.set_title('Improvement over Baseline')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (name, value) in enumerate(zip(display_names, improvements)):
        ax4.text(value, i, f' {value:+.1f}%', va='center', 
                fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig('detection_performance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: detection_performance.png")
    
    # Create second figure: Detailed Analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('OpenPII Watcher: Detailed Analysis', fontsize=16, fontweight='bold')
    
    # Plot 5: True Positives vs False Positives
    ax5 = axes2[0]
    
    tp = [main_results['micro_average'][t].true_positives for t in pii_types]
    fp = [main_results['micro_average'][t].false_positives for t in pii_types]
    fn = [main_results['micro_average'][t].false_negatives for t in pii_types]
    
    x = np.arange(len(pii_types))
    width = 0.25
    
    ax5.bar(x - width, tp, width, label='True Positives', color='#10b981', alpha=0.8)
    ax5.bar(x, fp, width, label='False Positives', color='#ef4444', alpha=0.8)
    ax5.bar(x + width, fn, width, label='False Negatives', color='#f59e0b', alpha=0.8)
    
    ax5.set_xlabel('PII Type', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('Detection Results Breakdown')
    ax5.set_xticks(x)
    ax5.set_xticklabels(display_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Overall System Summary
    ax6 = axes2[1]
    
    # Calculate overall metrics
    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [overall_precision, overall_recall, overall_f1]
    
    bars = ax6.bar(metrics_names, metrics_values, color=['#3b82f6', '#8b5cf6', '#ec4899'], alpha=0.7)
    
    ax6.set_ylabel('Score', fontweight='bold')
    ax6.set_title('Overall System Performance')
    ax6.set_ylim(0, 1.0)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: detailed_analysis.png")
    
    plt.close('all')


if __name__ == "__main__":
    run_experiments()

