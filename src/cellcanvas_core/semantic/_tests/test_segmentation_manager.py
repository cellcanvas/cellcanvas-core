import unittest
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import numpy.testing as npt
from sklearn.exceptions import NotFittedError

from cellcanvas_core.semantic._tests.mocks import (
    get_mock_data_manager,
    get_mock_segmentation_model,
)
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


class TestSemanticSegmentationManager(unittest.TestCase):
    def setUp(self):
        self.data_manager = get_mock_data_manager()
        self.model = get_mock_segmentation_model()
        self.manager = SemanticSegmentationManager(self.data_manager, self.model)

    def test_fit(self):
        # Mock training data
        features = da.random.random((10, 10, 10, 5), chunks=(10, 10, 10, 5))
        labels = da.random.randint(0, 2, (10, 10, 10, 1), chunks=(10, 10, 10, 1))

        # Mock get_training_data to return these features and labels
        self.data_manager.get_training_data = MagicMock(return_value=(features, labels))

        # Compute Dask arrays before passing them to the fit method
        computed_features, computed_labels = features.compute(), labels.compute()

        self.manager.fit()

        # Extract the call arguments
        called_args, _ = self.model.fit.call_args

        # Compare the arrays
        npt.assert_array_equal(called_args[0], computed_features)
        npt.assert_array_equal(called_args[1], computed_labels)

    def test_predict(self):
        # Create a dummy feature image
        feature_image = np.random.random((5, 10, 10, 10))  # (c, z, y, x)
        reshaped_features = feature_image.transpose(1, 2, 3, 0).reshape(-1, 5)

        # Set return value for predict
        self.model.predict.return_value = np.random.randint(
            0, 2, reshaped_features.shape[0]
        )

        # Call predict
        predicted_labels = self.manager.predict(feature_image)

        # Extract the call arguments and assert array equality
        called_args, _ = self.model.predict.call_args
        npt.assert_array_almost_equal(called_args[0], reshaped_features)

        self.assertEqual(
            predicted_labels.shape, feature_image.shape[1:]
        )  # Should match (z, y, x)

    def test_predict_not_fitted_error(self):
        self.model.predict.side_effect = NotFittedError("Model is not fitted.")
        feature_image = np.random.random((5, 10, 10, 10))

        with self.assertRaises(NotFittedError):
            self.manager.predict(feature_image)


if __name__ == "__main__":
    unittest.main()
