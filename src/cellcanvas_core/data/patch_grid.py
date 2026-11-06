"""Per-voxel features interpolated on demand from a stored patch-token grid.

A 2D vision transformer applied slice by slice gives one token per image
patch: a (D, Z, gy, gx) grid that is far smaller than features on every
voxel. `PatchGridFeatures` reads that grid as (D, Z, Y, X) voxel features,
resampling each Z slice bilinearly in Y and X with ``align_corners=False``
(the convention of ``torch.nn.functional.interpolate``) and leaving Z as is.
Only the requested region is computed, so a full-resolution dense feature
volume is never stored or allocated.

Interpolation reproduces the usual dense expansion of patch tokens; it adds
no spatial detail finer than the token spacing (Y / gy, X / gx voxels).

A Zarr array of tokens is recognized as a patch grid by its
``cellcanvas_patch_grid`` attribute::

    {
        "shape": [Z, Y, X],  # voxel grid the features are read on
        "interpolation": "bilinear",  # in Y and X, align_corners=False
        "align_corners": false,
        "dtype": "float16",  # dtype of the interpolated features
        "provenance": {...},
    }  # optional: model, preprocessing, source

Tokens are interpolated in float32 and then cast to ``dtype``, so a grid
whose dense expansion was stored as float16 reads back as the same values.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

PATCH_GRID_ATTRIBUTE = "cellcanvas_patch_grid"
# Voxels per default dask chunk: one prediction block of 65536 voxels.
DEFAULT_CHUNK_VOXELS = 65536
# Channels interpolated together; small enough for one block's rows to stay in cache.
_CHANNEL_BLOCK = 16


def _axis_weights(n_in, n_out):
    """Source indices and weights of bilinear resampling, align_corners=False.

    Matches PyTorch's upsample_bilinear2d: coordinates are computed in float32
    and clamped at zero, the last source index is repeated at the far edge.
    Returns (i0, i1, w1, w0) with value = w0 * src[i0] + w1 * src[i1].
    """
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0).astype(np.float32)
    i0 = src.astype(np.int64)
    i1 = np.where(i0 < n_in - 1, i0 + 1, i0)
    w1 = src - i0.astype(np.float32)
    return i0, i1, w1.astype(np.float32), (1.0 - w1).astype(np.float32)


def is_patch_grid(array):
    """Whether a Zarr array carries patch-grid metadata."""
    return PATCH_GRID_ATTRIBUTE in getattr(array, "attrs", {})


class PatchGridFeatures:
    """(D, Z, Y, X) features read lazily from a (D, Z, gy, gx) token grid.

    Parameters
    ----------
    tokens : array-like
        (D, Z, gy, gx) tokens, for example a Zarr array; read in the regions
        a request needs, never whole.
    shape : tuple of int
        (Z, Y, X) voxel grid. Z must equal the token grid's Z.
    dtype : numpy dtype
        dtype of returned features (interpolation itself is float32).
    provenance : dict, optional
        Where the tokens came from; kept with the features.
    chunk_voxels : int
        Voxels per chunk of `chunks`, used when the features are wrapped in a
        dask array. Chunks are (D, 1, rows, X), so one chunk never holds more
        than about ``D * chunk_voxels`` values.
    threads : int
        Threads interpolating one read, over blocks of channels. Values do
        not depend on it.
    """

    ndim = 4

    def __init__(
        self,
        tokens,
        shape,
        dtype=np.float32,
        provenance=None,
        chunk_voxels=DEFAULT_CHUNK_VOXELS,
        threads=4,
    ):
        if getattr(tokens, "ndim", None) != 4 or any(n == 0 for n in tokens.shape):
            raise ValueError("Tokens must be a nonempty (D, Z, gy, gx) array")
        shape = tuple(int(n) for n in shape)
        if len(shape) != 3 or any(n <= 0 for n in shape):
            raise ValueError("shape must be a nonempty (Z, Y, X) voxel grid")
        d, z, gy, gx = tokens.shape
        if shape[0] != z:
            raise ValueError(
                f"Token grid has {z} slices but the voxel grid has {shape[0]}; Z is not resampled"
            )
        if not isinstance(chunk_voxels, int) or chunk_voxels <= 0:
            raise ValueError("chunk_voxels must be a positive integer")
        if not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads must be a positive integer")
        self.threads = threads
        self.tokens = tokens
        self.shape = (d, *shape)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("Patch-grid features must have a floating dtype")
        self.provenance = dict(provenance or {})
        rows = min(shape[1], max(1, chunk_voxels // shape[2]))
        self.chunks = (d, 1, rows, shape[2])
        self._y = _axis_weights(gy, shape[1])
        self._x = _axis_weights(gx, shape[2])

    @classmethod
    def from_zarr(cls, array, chunk_voxels=DEFAULT_CHUNK_VOXELS, threads=4):
        """Open a Zarr token array that has ``cellcanvas_patch_grid`` metadata."""
        metadata = dict(array.attrs.get(PATCH_GRID_ATTRIBUTE) or {})
        if not metadata:
            raise ValueError(f"Array has no {PATCH_GRID_ATTRIBUTE} attribute")
        interpolation = metadata.get("interpolation", "bilinear")
        if (
            interpolation != "bilinear"
            or metadata.get("align_corners", False) is not False
        ):
            raise ValueError(
                "Only bilinear interpolation with align_corners=false is supported"
            )
        if "shape" not in metadata:
            raise ValueError(f"{PATCH_GRID_ATTRIBUTE} needs the voxel grid shape")
        return cls(
            array,
            metadata["shape"],
            metadata.get("dtype", "float32"),
            metadata.get("provenance"),
            chunk_voxels,
            threads,
        )

    @property
    def grid_spacing(self):
        """Voxels per token along Y and X."""
        return (
            self.shape[2] / self.tokens.shape[2],
            self.shape[3] / self.tokens.shape[3],
        )

    @property
    def nbytes(self):
        """Bytes the dense features would take; never allocated here."""
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __array__(self, dtype=None, copy=None):
        values = self[...]
        return values if dtype is None else values.astype(dtype)

    def __dask_tokenize__(self):
        store = getattr(self.tokens, "store", None)
        source = (
            (str(store), getattr(self.tokens, "path", None))
            if store is not None
            else id(self.tokens)
        )
        return ("PatchGridFeatures", source, self.shape, str(self.dtype), self.chunks)

    def __repr__(self):
        return (
            f"PatchGridFeatures(shape={self.shape}, tokens={tuple(self.tokens.shape)}, "
            f"dtype={self.dtype})"
        )

    def _indices(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if sum(k is Ellipsis for k in key) > 1:
            raise IndexError("Only one Ellipsis is allowed")
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (4 - len(key) + 1) + key[i + 1 :]
        if len(key) > 4:
            raise IndexError("Too many indices for (D, Z, Y, X) features")
        key = key + (slice(None),) * (4 - len(key))
        indices, squeeze = [], []
        for axis, (k, n) in enumerate(zip(key, self.shape, strict=True)):
            if isinstance(k, (int, np.integer)):
                k = int(k) + n if k < 0 else int(k)
                if not 0 <= k < n:
                    raise IndexError(
                        f"Index {k} is out of bounds for axis {axis} with size {n}"
                    )
                indices.append(np.array([k]))
                squeeze.append(axis)
            elif isinstance(k, slice):
                indices.append(np.arange(n)[k])
            else:
                raise TypeError(
                    "PatchGridFeatures supports integer and slice indexing; use sample() for voxels"
                )
        return indices, tuple(squeeze)

    def __getitem__(self, key):
        (di, zi, yi, xi), squeeze = self._indices(key)
        shape = (len(di), len(zi), len(yi), len(xi))
        if 0 in shape:
            out = np.zeros(shape, self.dtype)
            return out.squeeze(axis=squeeze) if squeeze else out
        y0, y1, wy1, wy0 = (a[yi] for a in self._y)
        x0, x1, wx1, wx0 = (a[xi] for a in self._x)
        row0, row1 = int(min(y0.min(), y1.min())), int(max(y0.max(), y1.max())) + 1
        col0, col1 = int(min(x0.min(), x1.min())), int(max(x0.max(), x1.max())) + 1
        t = np.asarray(
            self.tokens[
                int(di.min()) : int(di.max()) + 1,
                int(zi.min()) : int(zi.max()) + 1,
                row0:row1,
                col0:col1,
            ],
            dtype=np.float32,
        )
        t = t[di - di.min()][:, zi - zi.min()]
        x0, x1, y0, y1 = x0 - col0, x1 - col0, y0 - row0, y1 - row0
        out = np.empty(shape, self.dtype)

        def blend(start):
            # PyTorch's order: blend along X within each token row, then two rows
            # along Y. Separate elementwise operations (no fused or reordered sums)
            # make each voxel's value independent of the block it is read in and
            # equal to sample(). Channel blocks stay in cache and run in threads.
            part = t[start : start + _CHANNEL_BLOCK]
            a = part[..., x0]
            a *= wx0
            b = part[..., x1]
            b *= wx1
            a += b
            rows = np.take(a, y0, axis=-2)
            rows *= wy0[:, None]
            b = np.take(a, y1, axis=-2)
            b *= wy1[:, None]
            rows += b
            out[start : start + _CHANNEL_BLOCK] = rows

        starts = range(0, len(di), _CHANNEL_BLOCK)
        if self.threads > 1 and len(starts) > 1:
            with ThreadPoolExecutor(min(self.threads, len(starts))) as pool:
                list(pool.map(blend, starts))
        else:
            for start in starts:
                blend(start)
        return out.squeeze(axis=squeeze) if squeeze else out

    def sample(self, z, y, x):
        """(n, D) features at integer voxel coordinates, reading one token slice at a time.

        Equal to ``self[:, z_i, y_i, x_i]`` for each voxel; no block of voxels
        is formed, so memory is the rows plus one (D, gy, gx) token slice.
        """
        z, y, x = (np.asarray(v, dtype=np.int64).reshape(-1) for v in (z, y, x))
        if not len(z) == len(y) == len(x):
            raise ValueError("z, y and x must have equal lengths")
        for values, n in zip((z, y, x), self.shape[1:], strict=True):
            if len(values) and (values.min() < 0 or values.max() >= n):
                raise IndexError("Voxel coordinates are outside the grid")
        y0, y1, wy1, wy0 = (a[y] for a in self._y)
        x0, x1, wx1, wx0 = (a[x] for a in self._x)
        rows = np.empty((len(z), self.shape[0]), self.dtype)
        for slice_index in np.unique(z):
            m = z == slice_index
            t = np.asarray(
                self.tokens[:, int(slice_index)], dtype=np.float32
            )  # (D, gy, gx)
            top = t[:, y0[m], x0[m]] * wx0[m] + t[:, y0[m], x1[m]] * wx1[m]
            bottom = t[:, y1[m], x0[m]] * wx0[m] + t[:, y1[m], x1[m]] * wx1[m]
            rows[m] = (top * wy0[m] + bottom * wy1[m]).T.astype(self.dtype, copy=False)
        return rows


def write_patch_grid(path, tokens, shape, dtype="float16", provenance=None, **options):
    """Store a (D, Z, gy, gx) token grid as Zarr with patch-grid metadata.

    One chunk per Z slice. ``options`` are passed to ``zarr.create_array``
    (for example ``storage_options``). Returns the opened `PatchGridFeatures`.
    """
    import zarr

    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 4:
        raise ValueError("Tokens must be (D, Z, gy, gx)")
    metadata = {
        "shape": [int(n) for n in shape],
        "interpolation": "bilinear",
        "align_corners": False,
        "dtype": str(np.dtype(dtype)),
        "provenance": dict(provenance or {}),
    }
    PatchGridFeatures(tokens, shape, dtype)  # validate before writing
    array = zarr.create_array(
        path,
        shape=tokens.shape,
        chunks=(tokens.shape[0], 1, *tokens.shape[2:]),
        dtype="f4",
        attributes={PATCH_GRID_ATTRIBUTE: metadata},
        **options,
    )
    array[...] = tokens
    return PatchGridFeatures.from_zarr(array)
