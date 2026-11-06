"""Patch-token grids read as voxel features: coordinates, sparse reads, region prediction."""

import dask.array as da
import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.data.patch_grid import (
    PATCH_GRID_ATTRIBUTE,
    PatchGridFeatures,
    write_patch_grid,
)
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager

SHAPE = (3, 19, 23)  # odd sizes: token spacing is not an integer


def tokens(d=5, grid=(4, 6), seed=0):
    return np.random.default_rng(seed).normal(size=(d, SHAPE[0], *grid)).astype("f4")


def test_bilinear_align_corners_false_coordinates():
    """Voxel centres map to token coordinates (i + 0.5) * gy / Y - 0.5, clamped at the edges."""
    ramp = np.broadcast_to(np.arange(4, dtype="f4")[None, None, :, None], (1, 1, 4, 2))
    features = PatchGridFeatures(ramp, (1, 8, 2))
    expected = np.clip((np.arange(8) + 0.5) * 4 / 8 - 0.5, 0, 3)
    np.testing.assert_allclose(features[0, 0, :, 0], expected, atol=1e-6)
    constant = PatchGridFeatures(np.full((2, 3, 2, 2), 7, "f4"), (3, 5, 9))
    np.testing.assert_allclose(constant[...], 7, rtol=1e-6)


def test_matches_torch_interpolate():
    torch = pytest.importorskip("torch")
    t = tokens()
    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(t).permute(1, 0, 2, 3),
            size=SHAPE[1:],
            mode="bilinear",
            align_corners=False,
        )
        .permute(1, 0, 2, 3)
        .numpy()
    )
    np.testing.assert_allclose(PatchGridFeatures(t, SHAPE)[...], expected, atol=1e-5)


def test_indexing_and_sample_agree():
    features = PatchGridFeatures(tokens(), SHAPE, np.float16)
    dense = features[...]
    assert dense.shape == (5, *SHAPE) and dense.dtype == np.float16
    np.testing.assert_array_equal(features[2, 1], dense[2, 1])
    np.testing.assert_array_equal(features[:, -1, 4:9, ::3], dense[:, -1, 4:9, ::3])
    np.testing.assert_array_equal(features[..., 0], dense[..., 0])
    assert features[:, 0:0].shape == (5, 0, *SHAPE[1:])
    wide = PatchGridFeatures(tokens(d=40), SHAPE, threads=1)
    threaded = PatchGridFeatures(wide.tokens, SHAPE, threads=3)
    np.testing.assert_array_equal(threaded[:, 1, 3:11], wide[...][:, 1, 3:11])
    z, y, x = np.random.default_rng(1).integers(0, SHAPE, size=(40, 3)).T
    np.testing.assert_array_equal(features.sample(z, y, x), dense[:, z, y, x].T)
    with pytest.raises(IndexError):
        features.sample([0], [SHAPE[1]], [0])
    with pytest.raises(ValueError, match="Z is not resampled"):
        PatchGridFeatures(tokens(), (SHAPE[0] + 1, *SHAPE[1:]))


def test_dask_chunks_are_bounded_slices():
    features = PatchGridFeatures(tokens(), SHAPE, chunk_voxels=50)
    dataset = DataSet(
        np.zeros((1, *SHAPE)),
        {"tokens": features},
        np.zeros(SHAPE, "i4"),
        np.zeros(SHAPE, "i4"),
    )
    lazy = dataset.concatenated_features
    assert lazy.chunksize == (5, 1, 2, SHAPE[2])
    np.testing.assert_array_equal(np.asarray(lazy), features[...])
    with pytest.raises(ValueError, match="channel-first"):
        DataSet(
            np.zeros((1, *SHAPE)),
            {"tokens": features},
            np.zeros(SHAPE, "i4"),
            np.zeros(SHAPE, "i4"),
            feature_axis=-1,
        )


def ome_image(path, shape):
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.create_array(
        "0",
        shape=(1, *shape),
        dtype="f4",
        fill_value=0,
        dimension_names=("c", "z", "y", "x"),
    )
    axes = [{"name": "c", "type": "channel"}] + [
        {"name": a, "type": "space", "unit": "angstrom"} for a in "zyx"
    ]
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1, 10, 10, 10]}
                        ],
                    }
                ],
            }
        ],
    }
    return path


def test_from_paths_trains_and_predicts_like_its_dense_expansion(tmp_path):
    """A stored token grid opens lazily and gives the classifier the same rows as dense features."""
    t = tokens()
    provenance = {"model": "test", "public": True}
    write_patch_grid(str(tmp_path / "tokens.zarr"), t, SHAPE, "float32", provenance)
    stored = zarr.open_array(str(tmp_path / "tokens.zarr"), mode="r")
    assert stored.attrs[PATCH_GRID_ATTRIBUTE]["align_corners"] is False
    dense = PatchGridFeatures(t, SHAPE, np.float32)[...]
    zarr.save_array(str(tmp_path / "dense.zarr"), dense)
    image = ome_image(str(tmp_path / "image.zarr"), SHAPE)
    painting = np.zeros(SHAPE, "i4")
    painting[1, 2:6, 3:9], painting[1, 12:16, 10:20], painting[2, 8, 4:7] = 1, 2, 2

    outputs = {}
    for name in ("tokens", "dense"):
        dataset = DataSet.from_paths(
            image,
            str(tmp_path / f"{name}.zarr"),
            str(tmp_path / f"{name}-paint"),
            str(tmp_path / f"{name}-seg"),
            make_missing_datasets=True,
        )
        dataset.labels[:] = painting
        if name == "tokens":
            (source,) = dataset.features.values()
            assert (
                isinstance(source, PatchGridFeatures)
                and source.provenance == provenance
            )
        manager = SemanticSegmentationManager(
            DataManager([dataset]), DecisionTreeClassifier(random_state=0)
        )
        rows, _ = manager.data.get_training_data()
        outputs[name + "_rows"] = rows.compute()
        manager.fit()
        manager.predict_to_segmentation(dataset, batch_size=37)
        outputs[name] = dataset.segmentation[:]
        # Region prediction writes that slice only, with the same values.
        dataset.segmentation[:] = 0
        written = manager.predict_to_segmentation(
            dataset, batch_size=37, region=(slice(1, 2), slice(None), slice(None))
        )
        assert written == (slice(1, 2), slice(0, SHAPE[1]), slice(0, SHAPE[2]))
        region = dataset.segmentation[:]
        np.testing.assert_array_equal(region[1], outputs[name][1])
        assert not region[[0, 2]].any()
    np.testing.assert_array_equal(outputs["tokens_rows"], outputs["dense_rows"])
    np.testing.assert_array_equal(outputs["tokens"], outputs["dense"])
    assert len(np.unique(outputs["tokens"])) == 2


@pytest.mark.parametrize(
    "region",
    [
        (slice(0, 0), slice(None), slice(None)),
        (slice(None), slice(None), slice(0, 99)),
        (slice(None, None, 2), slice(None), slice(None)),
        (slice(None), slice(None)),
    ],
)
def test_invalid_regions_are_refused(region):
    dataset = DataSet(
        np.zeros((1, *SHAPE)),
        {"tokens": PatchGridFeatures(tokens(), SHAPE)},
        np.zeros(SHAPE, "i4"),
        np.zeros(SHAPE, "i4"),
    )
    manager = SemanticSegmentationManager(
        DataManager([dataset]), DecisionTreeClassifier()
    )
    with pytest.raises(ValueError, match="region"):
        manager.predict_to_segmentation(dataset, region=region)


def test_region_reads_match_dask_for_mixed_sources():
    t = PatchGridFeatures(tokens(), SHAPE)
    extra = np.random.default_rng(2).normal(size=(2, *SHAPE)).astype("f8")
    dataset = DataSet(
        np.zeros((1, *SHAPE)),
        {"tokens": t, "extra": da.from_array(extra, chunks=(2, 1, 7, 9))},
        np.zeros(SHAPE, "i4"),
        np.zeros(SHAPE, "i4"),
    )
    region = (slice(1, 3), slice(4, 15), slice(2, 20))
    expected = np.asarray(dataset.concatenated_features[(slice(None), *region)])
    block = dataset.read_features(region)
    assert block.dtype == np.float64
    np.testing.assert_array_equal(block, expected)
    z, y, x = np.random.default_rng(3).integers(0, SHAPE, size=(25, 3)).T
    np.testing.assert_array_equal(
        dataset.feature_rows(z, y, x),
        np.asarray(dataset.concatenated_features)[:, z, y, x].T,
    )


@pytest.mark.parametrize("chunks", [(1, 19, 23), (3, 8, 8), None])
def test_painted_voxels_read_bounded_slabs(chunks):
    labels = np.zeros(SHAPE, "i4")
    labels[0, 3, 4], labels[2, 18, 22], labels[1, 0, 0] = 2, 1, 5
    if chunks is not None:
        stored = zarr.create_array(store={}, shape=SHAPE, chunks=chunks, dtype="i4")
        stored[:] = labels
        labels = stored
    reads = []

    class Recording:
        shape, dtype = SHAPE, np.dtype("i4")

        def __getitem__(self, key):
            reads.append(key.stop - key.start)
            return labels[key]

    if chunks is not None:
        Recording.chunks = chunks
    dataset = DataSet(np.zeros((1, *SHAPE)), {"tokens": PatchGridFeatures(tokens(), SHAPE)},
                      Recording(), np.zeros(SHAPE, "i4"))
    indices, values = dataset.painted_voxels(max_voxels=SHAPE[1] * SHAPE[2])
    np.testing.assert_array_equal(indices, np.ravel_multi_index(([0, 1, 2], [3, 0, 18], [4, 0, 22]), SHAPE))
    np.testing.assert_array_equal(values, [2, 5, 1])
    assert max(reads) == 1  # one slice per read, even with three-slice chunks
