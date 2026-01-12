"""Quick start script for testing the pipeline on a single task.

This script demonstrates how to use the modules for a quick experiment.
Run this first to ensure everything works before running the full pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from feature_extraction import FeatureExtractor, StylometricFeatures
from model_trainer_file import ModelTrainer
from hyperparameter_tuning import HyperparameterTuner


def quick_test():
    """Run a quick test on a small subset of data."""
    
    print("="*70)
    print("QUICK START TEST")
    print("="*70)
    
    # 1. Load sample data
    print("\n1. Loading sample data...")
    data_file = Path('data/clean/extrovert_introvert_depolluted_text.csv')
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure you have run data preprocessing first.")
        return
    df = pd.read_csv(data_file)
    df = df.dropna(subset=['post_depolluted', 'extrovert'])
    
    # Use a small subset for testing
    df = df.sample(min(500, len(df)), random_state=42)
    
    texts = df['post_depolluted'].astype(str).tolist()
    labels = df['extrovert'].values
    
    print(f"   Loaded {len(texts)} samples")
    print(f"   Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 2. Test feature extraction
    print("\n2. Testing feature extraction...")
    
    # Test stylometric features
    print("   Extracting stylometric features...")
    stylo_extractor = StylometricFeatures()
    stylo_features = stylo_extractor.extract_batch(texts[:5])
    print(f"   Stylometric features shape: {stylo_features.shape}")
    print(f"   Sample features:\n{stylo_features.head()}")
    
    # Test content features
    print("\n   Extracting content-based features...")
    content_extractor = FeatureExtractor(feature_type='content', max_features=1000)
    X_content = content_extractor.fit_transform(texts)
    print(f"   Content features shape: {X_content.shape}")
    
    # 3. Test model training
    print("\n3. Testing model training...")
    
    # Use stylometric features for quick test
    extractor = FeatureExtractor(feature_type='stylometric')
    X = extractor.fit_transform(texts)
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Train a simple model
    print("   Training Logistic Regression with 3-fold CV...")
    trainer = ModelTrainer(model_type='logistic')
    cv_results = trainer.cross_validate(X, labels, n_splits=3)
    
    print(f"\n   Results:")
    print(f"   Mean Macro-F1: {cv_results['mean_macro_f1']:.4f} (+/- {cv_results['std_macro_f1']:.4f})")
    print(f"   Mean Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    if 'mean_auc' in cv_results:
        print(f"   Mean AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
    
    # 4. Test hyperparameter tuning (optional, takes longer)
    print("\n4. Testing hyperparameter tuning (5 trials only)...")
    response = input("   Run hyperparameter tuning test? (y/n): ")
    
    if response.lower() == 'y':
        tuner = HyperparameterTuner(
            model_type='logistic',
            n_trials=5,
            cv_splits=3,
            random_state=42
        )
        
        best_params = tuner.optimize(X, labels)
        print(f"\n   Best parameters: {best_params}")
        print(f"   Best score: {tuner.study.best_value:.4f}")
    else:
        print("   Skipping hyperparameter tuning test.")
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE!")
    print("="*70)
    print("\nAll modules are working correctly!")
    print("You can now run the full pipeline using main_experiment.py")
    print("\nNext steps:")
    print("  1. For full pipeline: python main_experiment.py")
    print("  2. For custom experiments: see README.md for examples")


def test_all_tasks():
    """Test that all data files are accessible."""
    
    print("\n" + "="*70)
    print("CHECKING DATA FILES")
    print("="*70 + "\n")
    
    tasks = {
        'extrovert_introvert': 'extrovert',
        'sensing_intuitive': 'sensing',
        'feeling_thinking': 'feeling',
        'judging_perceiving': 'judging'
    }
    
    data_dir = Path('data/clean')
    all_ok = True
    
    for task_name, label_col in tasks.items():
        depolluted_file = data_dir / f'{task_name}_depolluted_text.csv'
        raw_file = data_dir / f'{task_name}_tokens_raw.csv'
        
        print(f"Task: {task_name}")
        
        if depolluted_file.exists():
            df = pd.read_csv(depolluted_file)
            print(f"  ✓ Depolluted file: {len(df)} samples")
        else:
            print(f"  ✗ Depolluted file not found: {depolluted_file}")
            all_ok = False
        
        if raw_file.exists():
            df = pd.read_csv(raw_file)
            print(f"  ✓ Raw file: {len(df)} samples")
        else:
            print(f"  ✗ Raw file not found: {raw_file}")
            all_ok = False
        
        print()
    
    if all_ok:
        print("✓ All data files found and accessible!")
    else:
        print("✗ Some data files are missing. Please check your data directory.")
    
    return all_ok


def main():
    """Main function."""
    
    print("\n" + "#"*70)
    print(" "*20 + "PERSONALITY PREDICTION")
    print(" "*15 + "Quick Start & Testing Script")
    print("#"*70)
    
    # First check data files
    if not test_all_tasks():
        print("\nPlease ensure all preprocessed data files are in data/clean/")
        return
    
    # Then run quick test
    print("\n")
    quick_test()


if __name__ == "__main__":
    main()
