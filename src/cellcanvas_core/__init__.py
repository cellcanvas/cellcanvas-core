"""Core components for CellCanvas"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellcanvas-core")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "CellCanvas team"
__email__ = "kevin.yamauchi@gmail.com"
