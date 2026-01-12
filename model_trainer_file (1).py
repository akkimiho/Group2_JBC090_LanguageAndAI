"""Model training and evaluation module for personality prediction.

This module implements model training, cross-validation, and evaluation
according to the research proposal specifications.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate personality prediction models."""
    
    def __init__(self, model_type='logistic', random_state=42):
        """Initialize model trainer.
        
        Parameters
        ----------
        model_type : str
            Type of model: 'logistic', 'svm', 'rf', 'lgbm'
        random_state : int
            Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler(with_mean=False)  # For sparse matrices
        self.models = []
        self.results = []
        
    def _get_model(self, class_weight=None):
        """Get model instance based on type.
        
        Parameters
        ----------
        class_weight : dict or str, optional
            Class weights for handling imbalance
            
        Returns
        -------
        Sklearn-compatible model
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.random_state,
                solver='saga',
                penalty='l2'
            )
        elif self.model_type == 'svm':
            return LinearSVC(
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.random_state,
                dual=False
            )
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                class_weight=class_weight,
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'lgbm':
            return LGBMClassifier(
                class_weight=class_weight,
                n_estimators=100,
                random_state=self.random_state,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def cross_validate(self, X, y, n_splits=5, use_class_weight=True):
        """Perform stratified k-fold cross-validation.
        
        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix
        y : array-like
            Target labels
        n_splits : int
            Number of CV folds
        use_class_weight : bool
            Whether to use class weighting
            
        Returns
        -------
        dict
            Dictionary containing CV results
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        fold_results = []

        # Ensure X and y are in a format that supports integer indexing before the loop
        # This prevents the KeyError for DataFrames and the AttributeError for NumPy
        X_proc = X.values if hasattr(X, 'iloc') else X
        y_proc = y.values if hasattr(y, 'iloc') else y

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y_proc), 1):
            print(f"  Fold {fold}/{n_splits}...")
            
            # Now this indexing will work perfectly for any data type
            X_train, X_val = X_proc[train_idx], X_proc[val_idx]
            y_train, y_val = y_proc[train_idx], y_proc[val_idx]
                    
            # Handle class imbalance
            class_weight = None
            if use_class_weight:
                classes = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight = {classes[i]: weights[i] for i in range(len(classes))}
            
            # Train model
            model = self._get_model(class_weight=class_weight)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            fold_result = {
                'fold': fold,
                'accuracy': accuracy_score(y_val, y_pred),
                'macro_f1': f1_score(y_val, y_pred, average='macro'),
                'weighted_f1': f1_score(y_val, y_pred, average='weighted')
            }
            
            # Add AUC if binary classification
            if len(np.unique(y)) == 2:
                if hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_val)
                else:
                    y_score = model.predict_proba(X_val)[:, 1]
                fold_result['auc'] = roc_auc_score(y_val, y_score)
            
            fold_results.append(fold_result)
            self.models.append(model)
        
        # Aggregate results
        cv_results = {
            'folds': fold_results,
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_results]),
            'mean_macro_f1': np.mean([f['macro_f1'] for f in fold_results]),
            'std_macro_f1': np.std([f['macro_f1'] for f in fold_results]),
            'mean_weighted_f1': np.mean([f['weighted_f1'] for f in fold_results]),
            'std_weighted_f1': np.std([f['weighted_f1'] for f in fold_results])
        }
        
        if 'auc' in fold_results[0]:
            cv_results['mean_auc'] = np.mean([f['auc'] for f in fold_results])
            cv_results['std_auc'] = np.std([f['auc'] for f in fold_results])
        
        return cv_results
    
    def train_final_model(self, X_train, y_train, use_class_weight=True):
        """Train final model on full training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix
            Training features
        y_train : array-like
            Training labels
        use_class_weight : bool
            Whether to use class weighting
            
        Returns
        -------
        Trained model
        """
        class_weight = None
        if use_class_weight:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {classes[i]: weights[i] for i in range(len(classes))}
        
        model = self._get_model(class_weight=class_weight)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model on test set.
        
        Parameters
        ----------
        model : trained model
            Model to evaluate
        X_test : array-like or sparse matrix
            Test features
        y_test : array-like
            Test labels
            
        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_test)) == 2:
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            else:
                y_score = model.predict_proba(X_test)[:, 1]
            results['auc'] = roc_auc_score(y_test, y_score)
        
        return results
    
    def get_feature_importance(self, model, feature_names=None, top_n=20):
        """Extract feature importance from model.
        
        Parameters
        ----------
        model : trained model
            Model to extract importance from
        feature_names : list, optional
            Names of features
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            DataFrame with feature importances
        """
        if hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        else:
            print("Model does not support feature importance extraction")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class ExperimentRunner:
    """Run complete experiments with multiple configurations."""
    
    def __init__(self, output_dir='results'):
        """Initialize experiment runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        self.output_dir = output_dir
        self.all_results = []
        
    def run_experiment(self, X, y, experiment_name, model_types, n_splits=5):
        """Run experiment with multiple model types.
        
        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix
        y : array-like
            Target labels
        experiment_name : str
            Name of the experiment
        model_types : list
            List of model types to test
        n_splits : int
            Number of CV folds
            
        Returns
        -------
        dict
            Dictionary containing all experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        experiment_results = {
            'experiment_name': experiment_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'models': {}
        }
        
        for model_type in model_types:
            print(f"\nTraining {model_type.upper()}...")
            
            trainer = ModelTrainer(model_type=model_type)
            cv_results = trainer.cross_validate(X, y, n_splits=n_splits)
            
            experiment_results['models'][model_type] = cv_results
            
            # Print summary
            print(f"  Mean Macro-F1: {cv_results['mean_macro_f1']:.4f} (+/- {cv_results['std_macro_f1']:.4f})")
            print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
            if 'mean_auc' in cv_results:
                print(f"  Mean AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
        
        self.all_results.append(experiment_results)
        
        return experiment_results
    
    def save_results(self, filename='experiment_results.json'):
        """Save all experiment results to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def print_summary(self):
        """Print summary of all experiments."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")
        
        for exp in self.all_results:
            print(f"Experiment: {exp['experiment_name']}")
            print(f"  Samples: {exp['n_samples']}, Features: {exp['n_features']}")
            print(f"  Class Distribution: {exp['class_distribution']}")
            print("\n  Model Performance (Macro-F1):")
            
            for model_name, results in exp['models'].items():
                f1 = results['mean_macro_f1']
                std = results['std_macro_f1']
                print(f"    {model_name:10s}: {f1:.4f} (+/- {std:.4f})")
            print()


if __name__ == "__main__":
    # Test with dummy data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    # Test single model
    trainer = ModelTrainer(model_type='logistic')
    cv_results = trainer.cross_validate(X, y, n_splits=5)
    
    print("\nCross-Validation Results:")
    print(f"Mean Macro-F1: {cv_results['mean_macro_f1']:.4f}")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f}")
    print(f"Mean AUC: {cv_results['mean_auc']:.4f}")
