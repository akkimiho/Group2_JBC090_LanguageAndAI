import json
import pandas as pd
from pathlib import Path

def generate_scientific_summary(results_dir='results'):
    """Aggregates all JSON results into a single clean DataFrame."""
    path = Path(results_dir)
    tasks = ['extrovert_introvert', 'sensing_intuitive', 'feeling_thinking', 'judging_perceiving']
    models = ['logistic', 'svm', 'rf', 'lgbm']
    
    summary_rows = []

    for task in tasks:
        # 1. Load Content Results
        content_path = path / f"{task}_content_cv.json"
        # 2. Load Stylo Depolluted (Main Hypothesis)
        stylo_dep_path = path / f"{task}_stylometric_cv.json"
        # 3. Load RQ3 (Demographics)
        rq3_path = path / f"{task}_rq3_cv.json"
        
        for model in models:
            row = {'Task': task.replace('_', ' ').title(), 'Model': model.upper()}
            
            # Extract Content F1
            if content_path.exists():
                with open(content_path, 'r') as f:
                    data = json.load(f)
                    row['Content F1'] = data['models'][model]['mean_macro_f1']
            
            # Extract Stylometric (Depolluted) F1
            if stylo_dep_path.exists():
                with open(stylo_dep_path, 'r') as f:
                    data = json.load(f)
                    row['Stylometric F1'] = data['models'][model]['mean_macro_f1']
            
            # Extract RQ3 F1
            if rq3_path.exists():
                with open(rq3_path, 'r') as f:
                    data = json.load(f)
                    row['RQ3 (Demo) F1'] = data['models'][model]['mean_macro_f1']
            
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    
    # Print the master table
    print("\n" + "="*80)
    print("MASTER PERFORMANCE TABLE (MACRO-F1)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV for your own use
    df.to_csv(path / 'final_results_summary.csv', index=False)
    print(f"\nSummary saved to {path}/final_results_summary.csv")
    return df

if __name__ == "__main__":
    generate_scientific_summary()