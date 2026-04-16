import os
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SummaryEngine')

def parse_logs():
    reports_dir = 'outputs/reports'
    results = []
    
    if not os.path.exists(reports_dir):
        logger.error(f"Reports directory {reports_dir} not found.")
        return
        
    for file in os.listdir(reports_dir):
        if file.endswith('.log'):
            path = os.path.join(reports_dir, file)
            with open(path, 'r') as f:
                content = f.read()
                
            # Extract final AUC using regex
            match = re.search(r"FINAL TEST AUC \((.*?)\): ([\d\.]+)", content)
            if match:
                pair = match.group(1)
                auc = float(match.group(2))
                
                # Extract specifics
                sens = re.search(r"FINAL TEST SENSITIVITY: ([\d\.]+)", content)
                spec = re.search(r"FINAL TEST SPECIFICITY: ([\d\.]+)", content)
                
                results.append({
                    'Experiment': pair,
                    'ROC-AUC': auc,
                    'Sensitivity': float(sens.group(1)) if sens else 'N/A',
                    'Specificity': float(spec.group(1)) if spec else 'N/A'
                })
                
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("   FINAL RESEARCH MATRIX SUMMARY (FOR MENTOR)")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)
        
        # SAVE CSV
        df.to_csv('outputs/reports/final_summary.csv', index=False)
        logger.info("Summary saved to outputs/reports/final_summary.csv")
        
        # GENERATE PLOT
        try:
            import matplotlib.pyplot as plt
            os.makedirs('outputs/plots', exist_ok=True)
            plt.figure(figsize=(12, 6))
            df.set_index('Experiment')[['ROC-AUC', 'Sensitivity', 'Specificity']].plot(kind='bar', ax=plt.gca())
            plt.title('Clinical Generalization Matrix: Final Performance Metrics')
            plt.ylabel('Score (0.0 - 1.0)')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=15)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plot_path = 'outputs/plots/research_summary.png'
            plt.savefig(plot_path)
            logger.info(f"Comparison chart saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
    else:
        logger.warning("No completed experiments found in logs.")

if __name__ == "__main__":
    parse_logs()
