from unittest.mock import create_autospec

import numpy as np
from sklearn.base import BaseEstimator


class MockDataManager:
    def get_training_data(self):
        features = np.random.random((10, 10, 10, 5))  # Random 4D array
        labels = np.random.randint(0, 2, (10, 10, 10, 1))  # Random binary labels
        return features, labels


class MockSegmentationModel(BaseEstimator):
    def fit(self, X, y):
        pass  # Mock fitting logic

    def predict(self, X):
        return np.random.randint(0, 2, X.shape[0])  # Random binary predictions


def get_mock_data_manager():
    return MockDataManager()


def get_mock_segmentation_model():
    return create_autospec(MockSegmentationModel)
