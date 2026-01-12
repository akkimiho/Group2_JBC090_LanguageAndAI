import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def add_value_labels(ax, spacing=5, rotation=0):
    """Add annotated value labels with highlight boxes on top of bars."""
    for rect in ax.patches:
        y_value = rect.get_height()
        # Skip labels for empty bars or those that are actually 0
        if y_value <= 0 or pd.isna(y_value):
            continue
            
        x_value = rect.get_x() + rect.get_width() / 2
        label = f"{y_value:.3f}"
        
        ax.annotate(
            label,                      
            (x_value, y_value),         
            xytext=(0, spacing),        
            textcoords="offset points", 
            ha='center',                
            va='bottom',
            fontsize=8,
            fontweight='bold',
            rotation=rotation,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.7)
        )

def generate_research_visuals(results_dir='results', output_dir='visuals'):
    results_path = Path(results_dir)
    visuals_path = Path(output_dir)
    visuals_path.mkdir(exist_ok=True, parents=True)

    tasks = ['extrovert_introvert', 'sensing_intuitive', 'feeling_thinking', 'judging_perceiving']
    task_labels = {
        'extrovert_introvert': 'Extrovert/Introvert (EI)', 
        'sensing_intuitive': 'Sensing/Intuitive (SN)', 
        'feeling_thinking': 'Feeling/Thinking (FT)', 
        'judging_perceiving': 'Judging/Perceiving (JP)'
    }
    models = ['logistic', 'svm', 'rf', 'lgbm']
    
    all_data = []

    # 1. Scrape data from all possible experiment JSONs
    for task in tasks:
        files = {
            'Content Features': results_path / f"{task}_content_cv.json",
            'Stylometric (Raw)': results_path / f"{task}_stylometric_raw_cv.json",
            'Stylometric (Depolluted)': results_path / f"{task}_stylometric_cv.json",
            'Demographic Combined': results_path / f"{task}_rq3_cv.json"
        }

        for exp_name, file_path in files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for m_key in models:
                        if 'models' in data and m_key in data['models']:
                            m_data = data['models'][m_key]
                            all_data.append({
                                'Task': task_labels[task],
                                'Condition': exp_name,
                                'Algorithm': m_key.upper(),
                                'Macro-F1': m_data.get('mean_macro_f1', 0),
                                'Accuracy': m_data.get('mean_accuracy', 0)
                            })

    if not all_data:
        print("Error: No data found. Ensure result JSONs exist in the results folder.")
        return

    df = pd.DataFrame(all_data)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- VISUAL 1: H1 - Stylometric Robustness (Raw vs. Depolluted) ---
    plt.figure(figsize=(14, 8))
    h1_df = df[df['Condition'].str.contains('Stylometric')]
    ax1 = sns.barplot(data=h1_df, x='Task', y='Macro-F1', hue='Condition', palette='coolwarm')
    add_value_labels(ax1)
    plt.title('Hypothesis 1: Robustness of Stylometric Signal (Raw vs. Depolluted)', fontsize=16, pad=20)
    plt.ylabel('Predictive Performance (Macro-F1)', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.savefig(visuals_path / 'h1_robustness_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- VISUAL 2: H2 - Content vs. Stylometric Signal Strength ---
    plt.figure(figsize=(14, 8))
    h2_df = df[df['Condition'].isin(['Content Features', 'Stylometric (Depolluted)'])]
    ax2 = sns.barplot(data=h2_df, x='Task', y='Macro-F1', hue='Condition', palette='magma')
    add_value_labels(ax2)
    plt.title('Hypothesis 2: Content Signal (What) vs. Stylometric Signal (How)', fontsize=16, pad=20)
    plt.ylabel('Predictive Performance (Macro-F1)', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.savefig(visuals_path / 'h2_content_vs_style.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- VISUAL 3: H3 - Demographic Information Gain ---
    plt.figure(figsize=(14, 8))
    h3_df = df[df['Condition'].isin(['Stylometric (Depolluted)', 'Demographic Combined'])]
    ax3 = sns.barplot(data=h3_df, x='Task', y='Macro-F1', hue='Condition', palette='viridis')
    add_value_labels(ax3)
    plt.title('Hypothesis 3: Demographic Prediction Gain (RQ3 vs. Style Baseline)', fontsize=16, pad=20)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    plt.ylabel('Predictive Performance (Macro-F1)', fontweight='bold')
    plt.ylim(0, 1.2)
    plt.savefig(visuals_path / 'h3_demographic_gain.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- VISUAL 4: Model Comparison (Algorithm Performance across all Conditions) ---
    plt.figure(figsize=(14, 8))
    # Filter only for the primary depolluted stylometric task for a clear algorithm comparison
    algo_df = df[df['Condition'] == 'Stylometric (Depolluted)']
    ax4 = sns.barplot(data=algo_df, x='Task', y='Macro-F1', hue='Algorithm', palette='Set2')
    add_value_labels(ax4)
    plt.title('Algorithm Comparison: Performance of Models on Depolluted Stylometry', fontsize=16, pad=20)
    plt.ylabel('Macro-F1 Score', fontweight='bold')
    plt.ylim(0, 0.9)
    plt.savefig(visuals_path / 'algorithm_performance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- VISUAL 5: Master Algorithm Reliability (Aggregated Heatmap) ---
    plt.figure(figsize=(12, 10))
    # We pivot to see how models performed across every single task and condition
    heat_df = df.pivot_table(index=['Algorithm'], columns=['Task'], values='Macro-F1', aggfunc='mean')
    sns.heatmap(heat_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=0.5)
    plt.title('Algorithm Reliability: Mean Performance across Personality Dimensions', fontsize=16, pad=20)
    plt.savefig(visuals_path / 'algorithm_reliability_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- VISUAL 6: Condition Impact (Average model performance per Task) ---
    plt.figure(figsize=(14, 8))
    ax6 = sns.barplot(data=df, x='Condition', y='Macro-F1', hue='Task', palette='muted', capsize=.1)
    add_value_labels(ax6)
    plt.title('Condition Impact: Average Prediction Across Models for each Step', fontsize=16, pad=20)
    plt.xticks(rotation=15)
    plt.savefig(visuals_path / 'condition_impact_overview.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Comprehensive research visuals saved in: {visuals_path.absolute()}")

if __name__ == "__main__":
    generate_research_visuals()