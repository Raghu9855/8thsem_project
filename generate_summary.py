import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = 'outputs'
csv_path = os.path.join(OUTPUTS_DIR, 'reports', 'final_metrics_comparison.csv')

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. You must run evaluate.py completely at least once.")
    exit(1)

print(f"Loading existing metrics from: {csv_path}")
df = pd.read_csv(csv_path)

metrics_to_plot = {
    'Accuracy': 'master_accuracy_comparison.png',
    'Precision': 'master_precision_comparison.png',
    'Sensitivity (Recall)': 'master_recall_comparison.png',
    'ROC-AUC': 'master_performance_comparison.png'
}

for metric, filename in metrics_to_plot.items():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Experiment', y=metric, hue='Model', palette='viridis')
    if metric == 'ROC-AUC':
        plt.axhline(0.60, color='red', linestyle='--', alpha=0.6, label='Cross-Domain Goal (0.60)')
        plt.axhline(0.80, color='green', linestyle='--', alpha=0.6, label='Intra-Domain Goal (0.80)')
    plt.title(f'Final Model Performance Comparison ({metric})')
    plt.ylim(0.0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', filename))
    plt.close()
    print(f"Generated: outputs/plots/{filename}")

print("\n[SUCCESS] All 4 master performance charts have been instantly generated!")
