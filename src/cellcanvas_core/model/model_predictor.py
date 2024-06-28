import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator
from typing import Union, Tuple, Optional
import logging

class ModelPredictor:
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.logger = self._init_logging()

    def _init_logging(self):
        logger = logging.getLogger("cellcanvas.model_predictor")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def predict(self, X: Union[np.ndarray, da.Array]) -> np.ndarray:
        """Make predictions on input data."""
        self.logger.info("Starting prediction")
        if isinstance(X, da.Array):
            predictions = X.map_overlap(self._predict_chunk, depth=1, boundary='reflect')
            return predictions.compute()
        else:
            return self._predict_chunk(X)

    def _predict_chunk(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on a single chunk of data."""
        self.logger.debug(f"Predicting chunk of shape {X.shape}")
        return self.model.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1])

    def predict_proba(self, X: Union[np.ndarray, da.Array]) -> np.ndarray:
        """Make probability predictions on input data."""
        self.logger.info("Starting probability prediction")
        if isinstance(X, da.Array):
            probas = X.map_overlap(self._predict_proba_chunk, depth=1, boundary='reflect')
            return probas.compute()
        else:
            return self._predict_proba_chunk(X)

    def _predict_proba_chunk(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions on a single chunk of data."""
        self.logger.debug(f"Predicting probabilities for chunk of shape {X.shape}")
        return self.model.predict_proba(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1] + (-1,))

    def predict_large_volume(self, X: Union[np.ndarray, da.Array], 
                             chunk_size: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
        """Predict on a large volume using chunking."""
        self.logger.info(f"Starting large volume prediction with chunk size {chunk_size}")
        if isinstance(X, np.ndarray):
            X = da.from_array(X, chunks=chunk_size + (X.shape[-1],))
        predictions = self.predict(X)
        return predictions

    def predict_with_uncertainty(self, X: Union[np.ndarray, da.Array], 
                                 n_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation using Monte Carlo dropout."""
        self.logger.info(f"Starting prediction with uncertainty estimation, {n_iterations} iterations")
        predictions = []
        for i in range(n_iterations):
            self.logger.debug(f"Iteration {i+1}/{n_iterations}")
            pred = self.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        return mean_prediction, uncertainty

    def predict_with_threshold(self, X: Union[np.ndarray, da.Array], 
                               threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions based on a probability threshold."""
        self.logger.info(f"Making binary predictions with threshold {threshold}")
        probas = self.predict_proba(X)
        return (probas > threshold).astype(int)

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Calculate feature importance if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            self.logger.info("Getting feature importances from feature_importances_")
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.logger.info("Getting feature importances from coef_")
            return np.abs(self.model.coef_[0])
        else:
            self.logger.warning("Model does not provide feature importances")
            return None
