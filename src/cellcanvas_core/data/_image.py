"""Read image arrays without guessing NGFF axes or resolution paths."""

from copy import deepcopy

import zarr


def open_image(source, level=0):
    """Return a CZYX array and a copy of its source metadata.

    NGFF 0.4 groups use the ordered dataset paths in multiscales metadata.
    Bare arrays and legacy groups containing ``0`` remain supported.
    Coordinate transformations are preserved, not applied to pixel values.
    """
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise ValueError("image_level must be a nonnegative integer")
    root = zarr.open(source, mode="r")
    metadata = deepcopy(root.attrs.asdict())
    if isinstance(root, zarr.Array):
        if level != 0:
            raise ValueError("Bare image arrays only support image_level=0")
        image = root
    elif "multiscales" in metadata:
        multiscales = metadata["multiscales"]
        if not isinstance(multiscales, list) or len(multiscales) != 1:
            raise ValueError("Select a group with exactly one multiscale image")
        multiscale = multiscales[0]
        if multiscale.get("version") != "0.4":
            raise ValueError("Only NGFF version 0.4 image metadata is supported")
        axes = multiscale.get("axes", [])
        if not all(isinstance(axis, dict) for axis in axes) or [
            axis.get("name") for axis in axes
        ] != ["c", "z", "y", "x"]:
            raise ValueError("NGFF image axes must be ordered c, z, y, x")
        datasets = multiscale.get("datasets", [])
        if level >= len(datasets):
            raise ValueError("image_level is not listed in NGFF datasets")
        path = datasets[level].get("path")
        if not isinstance(path, str) or not path or any(
            part in ("", ".", "..") for part in path.split("/")
        ):
            raise ValueError("NGFF dataset path must be relative to the image group")
        try:
            image = root[path]
        except KeyError:
            raise ValueError(f"NGFF image dataset is missing: {path}") from None
    else:
        if level != 0 or "0" not in root:
            raise ValueError("Image group needs NGFF metadata or a legacy array named 0")
        image = root["0"]
    if not isinstance(image, zarr.Array) or image.ndim != 4 or any(
        size == 0 for size in image.shape
    ):
        raise ValueError("Images must have nonempty (C, Z, Y, X) axes")
    return image, metadata
