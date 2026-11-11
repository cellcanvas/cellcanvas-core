import numpy as np
import pytest
import zarr

from cellcanvas_core.data.data_set import DataSet


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("feature_axis", [0, -1])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_paths_preserve_inputs_and_reopen_annotations(tmp_path, existing, feature_axis, zarr_format):
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    features = image[np.newaxis] * 2
    if feature_axis == -1:
        features = np.moveaxis(features, 0, -1)
    image = np.stack([image, image + 100])
    paths = {name: str(tmp_path / name) for name in
             ("image", "features", "labels", "segmentation")}
    zarr.save_array(paths["image"], image, zarr_format=zarr_format)
    zarr.save_array(paths["features"], features, zarr_format=zarr_format)
    if existing:
        zarr.save_array(paths["labels"], np.full(image.shape[1:], 2, dtype="i4"), zarr_format=zarr_format)
        zarr.save_array(paths["segmentation"], np.full(image.shape[1:], 3, dtype="i4"), zarr_format=zarr_format)

    def reopen():
        return DataSet.from_paths(
            paths["image"], paths["features"], paths["labels"],
            paths["segmentation"], make_missing_datasets=not existing,
            feature_axis=feature_axis,
        )

    dataset = reopen()
    assert dataset.concatenated_features.shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(dataset.labels[:], 2 if existing else 0)
    np.testing.assert_array_equal(dataset.segmentation[:], 3 if existing else 0)
    dataset.labels[0, 1, 2] = 5
    dataset.segmentation[1, 2, 3] = 6
    reopened = reopen()
    assert reopened.image.metadata.zarr_format == zarr_format
    assert reopened.labels.metadata.zarr_format == (zarr_format if existing else 3)
    assert reopened.segmentation.metadata.zarr_format == (zarr_format if existing else 3)
    assert reopened.labels[0, 1, 2] == 5
    assert reopened.segmentation[1, 2, 3] == 6
    np.testing.assert_array_equal(reopened.image[:], image)
    np.testing.assert_array_equal(reopened.features[paths["features"]][:], features)


@pytest.mark.parametrize("feature_axis", [0, -1])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("existing", [False, True])
def test_stores_create_and_reopen_annotations(tmp_path, feature_axis, zarr_format, existing):
    stores = [zarr.storage.LocalStore(str(tmp_path / name)) for name in
              ("image", "features", "labels", "segmentation")]
    image = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    zarr.group(store=stores[0], zarr_format=zarr_format).create_array("0", data=image)
    features = image if feature_axis == 0 else np.moveaxis(image, 0, -1)
    zarr.array(features, store=stores[1], zarr_format=zarr_format)
    if existing:
        for store in stores[2:]:
            zarr.group(store=store, zarr_format=zarr_format).create_array(
                "labels", data=np.full(image.shape[1:], 2, dtype="i4")
            )
    dataset = DataSet.from_stores(*stores, feature_axis=feature_axis)
    np.testing.assert_array_equal(dataset.concatenated_features.compute(), image)
    np.testing.assert_array_equal(dataset.labels[:], 2 if existing else 0)
    np.testing.assert_array_equal(dataset.segmentation[:], 2 if existing else 0)
    dataset.labels[0, 1, 2] = 5
    dataset.segmentation[1, 2, 3] = 6
    reopened = DataSet.from_stores(*stores, feature_axis=feature_axis)
    assert reopened.image.metadata.zarr_format == zarr_format
    assert reopened.labels.metadata.zarr_format == (zarr_format if existing else 3)
    assert reopened.segmentation.metadata.zarr_format == (zarr_format if existing else 3)
    assert reopened.labels[0, 1, 2] == 5
    assert reopened.segmentation[1, 2, 3] == 6
    np.testing.assert_array_equal(reopened.image[:], image)
    np.testing.assert_array_equal(reopened.features["features"][:], features)


def test_missing_annotations_require_explicit_creation(tmp_path):
    image = str(tmp_path / "image")
    features = str(tmp_path / "features")
    zarr.save_array(image, np.ones((2, 1, 3, 4)))
    zarr.save_array(features, np.ones((5, 1, 3, 4)))
    with pytest.raises(FileNotFoundError):
        DataSet.from_paths(image, features, str(tmp_path / "labels"),
                           str(tmp_path / "predictions"))
    assert not (tmp_path / "labels").exists()


@pytest.fixture
def remote_root(tmp_path):
    """A URL root on an in-memory fsspec filesystem that Zarr opens as a remote store."""
    fsspec = pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem

    class RemoteMemory(MemoryFileSystem):
        protocol = ("cellcanvastest",)
        store = {}
        pseudo_dirs = [""]

    fsspec.register_implementation("cellcanvastest", RemoteMemory, clobber=True)
    return f"cellcanvastest://{tmp_path.name}"


def test_url_paths_open_with_storage_options(remote_root):
    root = remote_root
    options = {"cellcanvas_test_option": "kept"}
    image = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    zarr.save_array(f"{root}/image", image, zarr_format=3, storage_options=options)
    zarr.save_array(f"{root}/features", image * 2, zarr_format=3, storage_options=options)
    paths = [f"{root}/{name}" for name in ("image", "features", "labels", "segmentation")]

    dataset = DataSet.from_paths(*paths, make_missing_datasets=True, storage_options=options)
    np.testing.assert_array_equal(dataset.image[:], image)
    dataset.labels[0, 1, 2] = 4
    for array in (dataset.image, dataset.labels, dataset.segmentation, *dataset.features.values()):
        fs = array.store.fs
        fs = getattr(fs, "sync_fs", fs)  # Zarr wraps synchronous filesystems
        assert fs.storage_options["cellcanvas_test_option"] == "kept"
    reopened = DataSet.from_paths(*paths, storage_options=options)
    assert reopened.labels[0, 1, 2] == 4


def test_storage_options_are_ignored_for_local_paths(tmp_path):
    image = np.ones((1, 2, 3, 4), dtype=np.float32)
    zarr.save_array(str(tmp_path / "image"), image, zarr_format=3)
    zarr.save_array(str(tmp_path / "features"), image, zarr_format=3)
    dataset = DataSet.from_paths(str(tmp_path / "image"), str(tmp_path / "features"), str(tmp_path / "labels"),
                                 str(tmp_path / "segmentation"), make_missing_datasets=True,
                                 storage_options={"anon": True})
    assert dataset.labels.shape == (2, 3, 4)
