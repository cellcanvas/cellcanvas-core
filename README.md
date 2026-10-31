# cellcanvas-core

[![License](https://img.shields.io/pypi/l/cellcanvas-core.svg?color=green)](https://github.com/cellcanvas/cellcanvas-core/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellcanvas-core.svg?color=green)](https://pypi.org/project/cellcanvas-core)
[![Python Version](https://img.shields.io/pypi/pyversions/cellcanvas-core.svg?color=green)](https://python.org)
[![CI](https://github.com/cellcanvas/cellcanvas-core/actions/workflows/ci.yml/badge.svg)](https://github.com/cellcanvas/cellcanvas-core/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cellcanvas/cellcanvas-core/branch/main/graph/badge.svg)](https://codecov.io/gh/cellcanvas/cellcanvas-core)

Core components for CellCanvas

## Array dimensions

Images use **CZYX**: channel, depth, row, column. Two-dimensional data keeps
`Z=1`. A model adapter adds a leading batch axis, giving **BCZYX**. There is
no time axis in the current interface.

Labels and segmentation use **ZYX**, shared across image channels. Zero means
unannotated; positive values identify classes. Classifiers train on labels
minus one and return zero-based predictions; add one before saving them as
segmentation labels.

Classifier features use **DZYX**, where D is the feature width, not the number
of image channels. Feature arrays are joined along D in dictionary insertion
order. The same feature order must be used for training and prediction.
`feature_axis=-1` explicitly accepts ZYXD feature arrays, including through
`DataSet.from_paths()` and `DataSet.from_stores()`.

Per-channel patch embeddings have a separate trailing feature dimension
(`C, Z, nY, nX, D`, or with a leading batch axis). They need an explicit
adapter to align patches with the label grid and combine channel features
before use by the voxel classifier. That adapter is not implemented here.

This changes the image input contract: callers with ZYX images must add a
channel axis explicitly. Existing files are not rewritten automatically.
