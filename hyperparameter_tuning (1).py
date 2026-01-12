import optuna
from optuna.samplers import TPESampler
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna with fast iteration limits."""
    
    def __init__(self, model_type, n_trials=4, cv_splits=5, random_state=42):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_params = None
        self.study = None
        
    def _objective_logistic(self, trial, X, y):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['saga', 'liblinear']),
            'max_iter': 1000, # Kept at 1000 for speed
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        # Solver compatibility
        if params['penalty'] == 'l1':
            params['solver'] = 'liblinear' if params['solver'] == 'liblinear' else 'saga'
        elif params['solver'] == 'liblinear':
            params['penalty'] = 'l2'
        
        model = LogisticRegression(**params)
        scorer = make_scorer(f1_score, average='macro')
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        return cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1).mean()
    
    def _objective_svm(self, trial, X, y):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'max_iter': 1000, # Kept at 1000 for speed
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'dual': True  # Required for 'hinge' compatibility
        }
        model = LinearSVC(**params)
        scorer = make_scorer(f1_score, average='macro')
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        return cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1).mean()
    
    def _objective_rf(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200), # Lowered range for speed
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': 1
        }
        model = RandomForestClassifier(**params)
        scorer = make_scorer(f1_score, average='macro')
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        return cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1).mean()
    
    def _objective_lgbm(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.3, log=True),
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': 1,
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        scorer = make_scorer(f1_score, average='macro')
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        return cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1).mean()

    def optimize(self, X, y):
        if self.model_type == 'logistic': objective_func = lambda t: self._objective_logistic(t, X, y)
        elif self.model_type == 'svm': objective_func = lambda t: self._objective_svm(t, X, y)
        elif self.model_type == 'rf': objective_func = lambda t: self._objective_rf(t, X, y)
        elif self.model_type == 'lgbm': objective_func = lambda t: self._objective_lgbm(t, X, y)
        
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        self.study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=True)
        return self.study.best_params

    def tune_all(self, X, y):
        results = {}
        for m_type in ['logistic', 'svm', 'rf', 'lgbm']:
            self.model_type = m_type
            results[m_type] = self.optimize(X, y)
        return results
