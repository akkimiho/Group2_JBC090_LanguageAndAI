import pandas as pd
import numpy as np
from pathlib import Path
import json

from feature_extraction import FeatureExtractor
from model_trainer_file import ExperimentRunner
from hyperparameter_tuning import HyperparameterTuner

# --- JSON UTILITIES ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int_)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def deep_convert_keys(data):
    if isinstance(data, dict): return {str(k): deep_convert_keys(v) for k, v in data.items()}
    elif isinstance(data, list): return [deep_convert_keys(i) for i in data]
    else: return data

def safe_json_dump(data, filepath):
    clean_data = deep_convert_keys(data)
    with open(filepath, 'w') as f:
        json.dump(clean_data, f, indent=2, cls=NumpyEncoder)

# --- MAIN EXPERIMENT ---
class PersonalityExperiment:
    def __init__(self, data_dir='data/clean', output_dir='results'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.tasks = {
            # 'extrovert_introvert': 'extrovert', 
            'sensing_intuitive': 'sensing',
            'feeling_thinking': 'feeling', 
            'judging_perceiving': 'judging'
        }
        self.model_types = ['logistic', 'svm', 'rf', 'lgbm']

    def load_data(self, task_name, depolluted=True):
        label_col = self.tasks[task_name]
        text_col = 'post_depolluted' if depolluted else 'post_tokens'
        file_path = self.data_dir / (f'{task_name}_depolluted_text.csv' if depolluted else f'{task_name}_tokens_raw.csv')
        df = pd.read_csv(file_path).dropna(subset=[text_col, label_col])
        
        # Standardize IDs to strings and strip whitespace
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].values
        ids = df['auhtor_ID'].astype(str).str.strip().values
        return texts, labels, ids

    def load_demographics_rq3(self):
        raw_path = self.data_dir.parent / 'raw'
        try:
            gender_df = pd.read_csv(raw_path / 'gender.csv') 
            age_df = pd.read_csv(raw_path / 'birth_year.csv')
            nat_df = pd.read_csv(raw_path / 'nationality.csv')

            for df in [gender_df, age_df, nat_df]:
                df['auhtor_ID'] = df['auhtor_ID'].astype(str).str.strip()

            # Merge demographics together first
            merged = gender_df.merge(age_df, on='auhtor_ID', how='outer').merge(nat_df, on='auhtor_ID', how='outer')
            merged['gender_enc'] = merged['female'].fillna(0)
            merged['age_val'] = pd.to_numeric(merged['birth_year'], errors='coerce').fillna(0)
            nat_dummies = pd.get_dummies(merged['nationality'], prefix='nat').astype(int)
            
            return pd.concat([merged[['auhtor_ID', 'gender_enc', 'age_val']], nat_dummies], axis=1)
        except Exception as e:
            print(f"RQ3 Demographics loading failed: {e}")
            return None

    def run_task_experiment(self, task_name, feat_type, depolluted=True, tune=False, n_trials=2):
        print(f"\nTask: {task_name} | Feature: {feat_type} | Depolluted: {depolluted}")
        texts, labels, _ = self.load_data(task_name, depolluted)
        X = FeatureExtractor(feature_type=feat_type, max_features=5000).fit_transform(texts)
        if hasattr(X, "toarray"): X = X.toarray()
        
        if tune:
            tuner = HyperparameterTuner(model_type='logistic', n_trials=n_trials)
            return tuner.tune_all(X, labels)
        
        runner = ExperimentRunner(output_dir=str(self.output_dir))
        res = runner.run_experiment(X, labels, f"{task_name}_{feat_type}", self.model_types)
        safe_json_dump(res, self.output_dir / f"{task_name}_{feat_type}_cv.json")
        return res
    
    def run_rq3_experiment(self, task_name):
        print(f"\n--- Running RQ3 (Demographics + Stylometric) for {task_name} ---")
        texts, labels, author_ids = self.load_data(task_name, depolluted=True)
        
        fe = FeatureExtractor(feature_type='stylometric')
        X_stylo = fe.fit_transform(texts)
        if hasattr(X_stylo, "toarray"): X_stylo = X_stylo.toarray()
        
        df_stylo = pd.DataFrame(X_stylo)
        df_stylo['auhtor_ID'] = author_ids
        df_stylo['target_label'] = labels 
        
        df_demo = self.load_demographics_rq3()
        if df_demo is None: return None
            
        # --- DYNAMIC MERGE STRATEGY ---
        overlap = set(df_stylo['auhtor_ID']).intersection(set(df_demo['auhtor_ID']))
        
        
        
        if len(overlap) > 0:
            print(f"Match found! Overlap count: {len(overlap)}. Using INNER merge.")
            merged = df_stylo.merge(df_demo, on='auhtor_ID', how='inner')
        else:
            print(f"No match found. IDs likely mismatched. Using LEFT merge to preserve data.")
            merged = df_stylo.merge(df_demo, on='auhtor_ID', how='left')

        # 4. Final Feature Preparation (Fill missing with 0)
        y_final = merged['target_label'].values
        X_final = np.ascontiguousarray(
            merged.drop(columns=['auhtor_ID', 'target_label']).fillna(0).values
        )
        
        # 5. Run combined experiment
        runner = ExperimentRunner(output_dir=str(self.output_dir))
        res = runner.run_experiment(X_final, y_final, f"{task_name}_rq3", self.model_types)
        safe_json_dump(res, self.output_dir / f"{task_name}_rq3_cv.json")
        return res

    def run_full_pipeline(self):
        print(f"\nStarting full pipeline for remaining tasks: {list(self.tasks.keys())}")
        for task in self.tasks.keys():
            print(f"\n{'='*40}\nPIPELINE: {task.upper()}\n{'='*40}")
            # 1. Content Baseline
            self.run_task_experiment(task, 'content', True)
            # 2. Stylometric Raw (for comparison)
            self.run_task_experiment(task, 'stylometric', False)
            # 3. Stylometric Depolluted (Main Hypothesis)
            self.run_task_experiment(task, 'stylometric', True)
            # 4. RQ3 Combined Analysis
            self.run_rq3_experiment(task)
            # 5. Final Hyperparameter Tuning
            self.run_task_experiment(task, 'stylometric', True, True, 4)

if __name__ == "__main__":
    exp = PersonalityExperiment()
    
    # Run pipeline for remaining tasks
    exp.run_full_pipeline()
    
    print("\n--- ALL TASKS COMPLETE ---")