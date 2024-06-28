import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from typing import Optional, Dict, Any
import copy

class ModelTrainer:
    def __init__(self, model: BaseEstimator, 
                 use_class_weights: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.model = model
        self.use_class_weights = use_class_weights
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = self._init_logging()

    def _init_logging(self):
        logger = logging.getLogger("cellcanvas.model_trainer")
        logger.setLevel(logging.DEBUG)
        return logger

    def compute_class_weights(self, y):
        """Compute class weights for imbalanced datasets."""
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return dict(zip(np.unique(y), class_weights))

    def preprocess_data(self, X, y):
        """Preprocess the input data and labels."""
        # Subclasses should implement preprocessing
        return X, y

    def train(self, X, y, **kwargs):
        """Train the model on the given data."""
        self.logger.info("Starting model training")
        X, y = self.preprocess_data(X, y)
        
        if self.use_class_weights:
            class_weights = self.compute_class_weights(y)
            self.model.set_params(class_weight=class_weights)
        
        self.model.set_params(n_jobs=self.n_jobs, random_state=self.random_state)
        self.model.set_params(**kwargs)
        
        self.model.fit(X, y)
        self.logger.info("Model training completed")

    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation on the model."""
        self.logger.info(f"Performing {cv}-fold cross-validation")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        self.logger.info(f"Cross-validation scores: mean={scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores

    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model from a file."""
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(enumerate(self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(enumerate(self.model.coef_[0]))
        else:
            self.logger.warning("Model does not provide feature importances")
            return None

    def set_model_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        self.logger.info(f"Model parameters updated: {params}")

    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return self.model.get_params()

    def hyperparameter_tuning(self, X, y, param_grid: Dict[str, Any], cv=5, 
                              scoring='accuracy', search_method: str = 'grid', 
                              n_iter: int = 10):
        """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV."""
        self.logger.info("Starting hyperparameter tuning")
        if search_method == 'grid':
            search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        elif search_method == 'random':
            search = RandomizedSearchCV(self.model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            raise ValueError("search_method must be 'grid' or 'random'")
        
        search.fit(X, y)
        self.logger.info(f"Best parameters found: {search.best_params_}")
        self.model = search.best_estimator_
        return search.best_params_, search.best_score_

    def train_with_early_stopping(self, X, y, patience: int = 10, **kwargs):
        """Train the model with early stopping."""
        self.logger.info("Starting model training with early stopping")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        X_train, y_train = self.preprocess_data(X_train, y_train)
        X_val, y_val = self.preprocess_data(X_val, y_val)

        if self.use_class_weights:
            class_weights = self.compute_class_weights(y_train)
            self.model.set_params(class_weight=class_weights)
        
        self.model.set_params(n_jobs=self.n_jobs, random_state=self.random_state)
        self.model.set_params(**kwargs)

        best_score = -np.inf
        epochs_no_improve = 0
        best_model = None
        for epoch in range(1, 1000):
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)

            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(self.model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                self.logger.info(f"Early stopping after {epoch} epochs")
                break
        
        self.model = best_model
        self.logger.info("Model training with early stopping completed")
