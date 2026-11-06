from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import dask.array as da
import numpy as np
import zarr
from zarr import Array

from cellcanvas_core.data._image import _store_options, open_image
from cellcanvas_core.data._geometry import SpatialTransform, spatial_transform, validate_alignment
from cellcanvas_core.data.patch_grid import PatchGridFeatures, is_patch_grid


def _lazy(value):
    """A dask array over value, with the source's own chunks.

    da.asarray would rechunk a Zarr array to dask's chunk-size limit (128 MiB
    by default), and blocks that cut across storage chunks decode each storage
    chunk many times. Patch-grid features keep their (D, 1, rows, X) chunks
    rather than auto chunks that span many slices.
    """
    if isinstance(value, Array):
        return da.from_zarr(value)
    if isinstance(value, PatchGridFeatures):
        return da.from_array(value, chunks=value.chunks, meta=np.empty((0,) * 4, value.dtype))
    return da.asarray(value)


def _open_features(path, storage_options=None, mode="r"):
    """A feature array: stored per voxel, or a patch-token grid read lazily."""
    array = zarr.open_array(path, mode=mode, **_store_options(path, storage_options))
    return PatchGridFeatures.from_zarr(array) if is_patch_grid(array) else array


def _spatial_chunks(array, axis):
    """Spatial (Z, Y, X) chunk edges of a feature source, or None if unknown."""
    chunks = getattr(array, "chunks", None)
    if chunks is None or isinstance(array, np.ndarray):
        return None
    chunks = list(chunks)
    del chunks[axis]
    edges = []
    for size, chunk in zip([n for i, n in enumerate(array.shape) if i != axis], chunks, strict=True):
        if isinstance(chunk, (int, np.integer)):
            edges.append(np.arange(0, size, int(chunk)))
        else:  # dask: block sizes
            edges.append(np.cumsum((0, *chunk))[:-1])
    return edges


def _open_annotations(path, shape, create, level, expected, grouped=False, storage_options=None):
    options = _store_options(path, storage_options)
    try:
        node = zarr.open(path, mode="r+", **options)
    except FileNotFoundError:
        if not create:
            raise
        if grouped:
            node = zarr.open_group(path, mode="a", zarr_format=3, **options)
        else:
            return zarr.open_array(path, mode="a", shape=shape, dtype="i4", zarr_format=3,
                                   dimension_names=("z", "y", "x"), **options)
    if isinstance(node, zarr.Group):
        metadata = node.attrs.asdict()
        if "ome" in metadata or "multiscales" in metadata:
            node, metadata = open_image(path, level=level, axes=("z", "y", "x"), mode="r+",
                                        storage_options=storage_options)
            validate_alignment(metadata, level, expected)
        elif grouped:
            if "labels" in node:
                node = node["labels"]
            elif create:
                options = {"dimension_names": ("z", "y", "x")} if node.metadata.zarr_format == 3 else {}
                node = node.create_array("labels", shape=shape, dtype="i4", **options)
            else:
                raise ValueError("Annotation group has no labels array")
        else:
            node, _ = open_image(path, level=0, axes=("z", "y", "x"), mode="r+",
                                 storage_options=storage_options)
    if not isinstance(node, zarr.Array) or node.shape != shape:
        raise ValueError("Annotation spatial shape does not match the image")
    names = getattr(node.metadata, "dimension_names", None)
    if names is not None and names != ("z", "y", "x"):
        raise ValueError("Annotation dimension_names must be ordered z, y, x")
    return node

@dataclass
class DataSet:
    """CZYX image, feature-first DZYX arrays and shared ZYX annotations.

    D is the classifier feature width, distinct from image channels C.
    Two-dimensional images retain a singleton Z axis. No time or batch axis
    is stored here; model adapters add a leading batch axis to images.
    """

    image: Array
    features: Dict[str, Array]
    labels: Array
    segmentation: Array
    feature_axis: int = 0
    image_metadata: Dict = field(default_factory=dict)
    image_level: int = 0

    def __post_init__(self):
        if self.image.ndim != 4 or any(size == 0 for size in self.image.shape):
            raise ValueError("Images must have nonempty (C, Z, Y, X) axes")
        spatial_shape = self.image.shape[1:]
        if not self.features:
            raise ValueError("At least one feature array is required")
        for name, values in self.features.items():
            if values.ndim != 4:
                raise ValueError(f"Feature {name} needs one channel axis and the image spatial axes")
            axis = self.feature_axis % values.ndim
            if not -values.ndim <= self.feature_axis < values.ndim:
                raise ValueError("feature_axis is outside the feature array")
            if isinstance(values, PatchGridFeatures) and axis != 0:
                raise ValueError(f"Patch-grid feature {name} is channel-first; use feature_axis=0")
            spatial = values.shape[:axis] + values.shape[axis + 1:]
            if spatial != spatial_shape or values.shape[axis] == 0:
                raise ValueError(f"Feature {name} does not match the image shape")
        if self.labels.shape != spatial_shape or self.segmentation.shape != spatial_shape:
            raise ValueError("Labels and segmentation must match the image spatial shape (Z, Y, X)")

    @property
    def spatial_transform(self) -> SpatialTransform:
        """Image scale, origin and units for aligning ZYX annotations."""
        return spatial_transform(self.image_metadata, self.image_level)

    @property
    def concatenated_features(self) -> da.Array:
        """Return lazy channel-first features in dictionary insertion order.

        Set feature_axis=-1 for prototype arrays with channels last. Reading
        this property builds a task graph without loading the whole volume.
        Zarr arrays keep their storage chunks as dask chunks, so each chunk
        is read once per pass however large it is.
        """
        features = [da.moveaxis(_lazy(value), self.feature_axis, 0)
                    for value in self.features.values()]
        return da.concatenate(features, axis=0)

    @property
    def feature_dtype(self):
        """dtype of concatenated feature rows."""
        return np.result_type(*(np.dtype(values.dtype) for values in self.features.values()))

    def read_features(self, region):
        """(D, z, y, x) NumPy features of one (Z, Y, X) region of slices.

        Each source is read directly in that region: a Zarr array decodes only
        the storage chunks the region touches, a patch grid interpolates only
        the region, so memory follows the region size, not the volume.
        """
        region = tuple(region)
        if len(region) != 3 or not all(isinstance(r, slice) for r in region):
            raise ValueError("region must be three slices (Z, Y, X)")
        blocks = []
        for values in self.features.values():
            axis = self.feature_axis % values.ndim
            key = region[:axis] + (slice(None),) + region[axis:]
            blocks.append(np.moveaxis(np.asarray(values[key]), axis, 0))
        return np.concatenate(blocks, axis=0, dtype=self.feature_dtype)

    def feature_rows(self, z, y, x):
        """(n, D) NumPy feature rows at integer voxel coordinates.

        Patch grids interpolate only those voxels. Other sources are read in
        the bounding box of the voxels inside each of their storage chunks (or
        each Z slice if their chunks are unknown), so sparse painting in a
        large volume reads little of it.
        """
        z, y, x = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (z, y, x))
        columns = []
        for values in self.features.values():
            axis = self.feature_axis % values.ndim
            channels = values.shape[axis]
            if hasattr(values, "sample") and axis == 0:
                columns.append(np.asarray(values.sample(z, y, x)))
                continue
            if isinstance(values, np.ndarray):
                columns.append(np.moveaxis(values, axis, 0)[:, z, y, x].T)
                continue
            rows = np.empty((len(z), channels), np.dtype(values.dtype))
            edges = _spatial_chunks(values, axis) or [np.arange(values.shape[1 if axis == 0 else 0]), [0], [0]]
            block = np.stack([np.searchsorted(e, c, side="right") - 1
                              for e, c in zip(edges, (z, y, x), strict=True)], axis=1)
            _, group = np.unique(block, axis=0, return_inverse=True)
            for g in range(group.max() + 1 if len(group) else 0):
                m = np.flatnonzero(group.reshape(-1) == g)
                low = [int(c[m].min()) for c in (z, y, x)]
                high = [int(c[m].max()) + 1 for c in (z, y, x)]
                region = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
                key = region[:axis] + (slice(None),) + region[axis:]
                box = np.moveaxis(np.asarray(values[key]), axis, 0)
                rows[m] = box[:, z[m] - low[0], y[m] - low[1], x[m] - low[2]].T
            columns.append(rows)
        return np.concatenate(columns, axis=1, dtype=self.feature_dtype)

    def painted_voxels(self, max_voxels=1 << 24):
        """Flat C-order indices and values of the painted (nonzero) labels.

        Labels are read in Z slabs of at most about ``max_voxels`` voxels
        (whole storage chunks in Z when they fit), so the label volume is
        never loaded whole.
        """
        labels = self.labels
        z, y, x = labels.shape
        slab = max(1, max_voxels // (y * x))
        chunks = getattr(labels, "chunks", None)
        if chunks is not None and isinstance(chunks[0], (int, np.integer)) and chunks[0] <= slab:
            slab = slab // int(chunks[0]) * int(chunks[0])
        indices, values = [], []
        for start in range(0, z, slab):
            block = np.asarray(labels[start:start + slab])
            flat = np.flatnonzero(block)
            indices.append(flat + start * y * x)
            values.append(block.reshape(-1)[flat])
        return np.concatenate(indices), np.concatenate(values)

    def __hash__(self):
        """Simple hash function, should updated in the future.

        Note: this might create issues if files are moved.
        """
        return hash(self.image.path)

    @classmethod
    def from_paths(
        cls,
        image_path: str,
        features_path: Union[List[str], str],
        labels_path: str,
        segmentation_path: str,
        make_missing_datasets: bool = False,
        feature_axis: int = 0,
        image_level: int = 0,
        storage_options: Optional[Dict] = None,
    ):
        """Create a DataSet from a set of paths.

        Paths may be fsspec URLs (for example ``s3://bucket/image.zarr``);
        ``storage_options`` are passed to the filesystem of every URL path,
        so remote stores can use anonymous access or a named profile.
        Credentials should come from the environment, not these options.

        todo: add ability to create missing labels/segmentations
        """
        # get the image
        image, image_metadata = open_image(image_path, level=image_level,
                                           storage_options=storage_options)
        geometry = spatial_transform(image_metadata, image_level)

        # get the features
        if isinstance(features_path, str):
            features_path = [features_path]
        # A Zarr array with cellcanvas_patch_grid metadata is a token grid read lazily per voxel.
        features = {path: _open_features(path, storage_options) for path in features_path}

        # get the labels
        labels = _open_annotations(
            labels_path, image.shape[1:], make_missing_datasets, image_level, geometry,
            storage_options=storage_options,
        )

        # get the segmentation
        segmentation = _open_annotations(
            segmentation_path, image.shape[1:], make_missing_datasets, image_level, geometry,
            storage_options=storage_options,
        )

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
            image_metadata=image_metadata,
            image_level=image_level,
        )

    @classmethod
    def from_stores(
        cls,
        image_store,
        features_store,
        labels_store,
        segmentation_store,
        feature_axis: int = 0,
        image_level: int = 0,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # TODO rewrite this to copy everything to be local
        
        # get the image
        image, image_metadata = open_image(image_store, level=image_level)
        geometry = spatial_transform(image_metadata, image_level)

        # get the features
        features = {"features": _open_features(features_store)}

        # get the labels
        labels = _open_annotations(labels_store, image.shape[1:], True, image_level,
                                   geometry, grouped=True)

        # get the segmentation
        segmentation = _open_annotations(segmentation_store, image.shape[1:], True,
                                         image_level, geometry, grouped=True)

        # TODO start a background thread that triggers downloads of the zarrs
            
        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
            image_metadata=image_metadata,
            image_level=image_level,
        )
