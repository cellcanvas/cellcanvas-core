import unittest
from unittest.mock import create_autospec

import dask.array as da
import numpy as np

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet


class TestDataManager(unittest.TestCase):
    def setUp(self):
        # Create a mock DataSet
        self.dataset_mock = create_autospec(DataSet)

        # Create fake data for the dataset
        self.concatenated_features = np.random.rand(3, 10, 10, 2)  # (c, h, w, d)
        self.labels = np.random.randint(0, 3, (10, 10, 2))  # (h, w, d), with some zeros

        # Assign the fake data to the mock
        self.dataset_mock.concatenated_features = da.from_array(
            self.concatenated_features
        )
        self.dataset_mock.labels = da.from_array(self.labels)

    def test_initialization_no_datasets(self):
        manager = DataManager()
        # TODO this list used to be a napari SelectableEventedList
        self.assertIsInstance(manager.datasets, list)
        self.assertEqual(len(manager.datasets), 0)

    def test_initialization_with_datasets(self):
        manager = DataManager([self.dataset_mock])
        self.assertEqual(len(manager.datasets), 1)

    def test_initialization_with_single_dataset(self):
        # Wrap mock in a list to simulate single dataset initialization
        manager = DataManager([self.dataset_mock])
        self.assertEqual(len(manager.datasets), 1)

    def test_get_training_data(self):
        manager = DataManager([self.dataset_mock])
        features, labels = manager.get_training_data()

        # Compute the results
        features = features.compute()
        labels = labels.compute()

        # Check shapes and content
        valid_indices = np.where(self.labels.flatten() > 0)[0]
        reshaped_features = self.concatenated_features.reshape(3, -1)
        expected_features = np.stack(
            [reshaped_features[i, valid_indices] for i in range(3)], axis=1
        )
        expected_labels = self.labels.flatten()[valid_indices] - 1

        np.testing.assert_array_almost_equal(features, expected_features)
        np.testing.assert_array_equal(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
