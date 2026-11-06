import unittest

import numpy as np
import zarr

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet


class TestDataManager(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.concatenated_features = rng.random((3, 10, 10, 2))  # (c, z, y, x)
        self.labels = rng.integers(0, 3, (10, 10, 2))  # (z, y, x), with some zeros
        self.dataset_mock = DataSet(np.zeros((1, 10, 10, 2)), {"features": self.concatenated_features},
                                    self.labels, np.zeros((10, 10, 2), dtype="i4"))

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

    def test_training_reads_only_chunks_with_painting(self):
        """Zarr features are read in painted storage chunks only, with the same rows."""
        features = zarr.create_array(store={}, shape=(3, 10, 10, 2), chunks=(3, 1, 5, 2), dtype="f8")
        features[:] = self.concatenated_features
        reads = []

        class Recording:
            shape, dtype, chunks, ndim = features.shape, features.dtype, features.chunks, 4

            def __getitem__(self, key):
                reads.append(key)
                return features[key]

        labels = np.zeros((10, 10, 2), "i4")
        labels[3, 1, 0], labels[3, 2, 1], labels[7, 9, 1] = 1, 2, 1
        dataset = DataSet(np.zeros((1, 10, 10, 2)), {"features": Recording()}, labels,
                          np.zeros_like(labels))
        rows, classes = (a.compute() for a in DataManager([dataset]).get_training_data())
        np.testing.assert_array_equal(rows, self.concatenated_features[:, [3, 3, 7], [1, 2, 9], [0, 1, 1]].T)
        np.testing.assert_array_equal(classes, [0, 1, 0])
        assert len(reads) == 2  # two chunks hold painting, of 20
        assert all(k[1].stop - k[1].start == 1 for k in reads)


if __name__ == "__main__":
    unittest.main()
