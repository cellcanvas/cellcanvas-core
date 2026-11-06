from typing import List, Optional, Tuple

import dask.array as da
import numpy as np
from zarr import Array

from cellcanvas_core.data.data_set import DataSet


class DataManager:
    def __init__(self, datasets: Optional[List[DataSet]] = None):
        if datasets is None:
            datasets = []
        elif isinstance(datasets, DataSet):
            datasets = [datasets]
        # TODO this used to be a napari SelectableEventedList
        self.datasets = list(datasets)

    def get_training_data(self) -> Tuple[Array, Array]:
        """Get the pixel-wise semantic segmentation training data for datasets.

        Only painted voxels are read: labels in bounded Z slabs, then the
        feature rows at those voxels (see `DataSet.feature_rows`), so the
        cost follows the painting, not the volume. Rows are in flat C order
        of each dataset's voxels, datasets in order.

        Returns
        -------
        features : Array
            (n, d) array of features where n is the number of pixels
            and d is the number feature dimensions.
        labels : Array
            (n,) array of labels for each feature.
        """

        features = []
        labels = []
        if not self.datasets:
            raise ValueError("At least one dataset is required for training")
        for dataset in self.datasets:
            indices, values = dataset.painted_voxels()
            if len(indices) == 0:
                continue
            rows = dataset.feature_rows(*np.unravel_index(indices, dataset.labels.shape))
            features.append(da.from_array(rows, chunks=rows.shape))
            labels.append(da.from_array(values - 1, chunks=values.shape))

        if not features:
            raise ValueError("Paint at least one positive label before training")
        return da.concatenate(features), da.concatenate(labels)
