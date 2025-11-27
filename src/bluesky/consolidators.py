import warnings

from bluesky_tiled_plugins.consolidators import (
    ConsolidatorBase,
    CSVConsolidator,
    HDF5Consolidator,
    JPEGConsolidator,
    MultipartRelatedConsolidator,
    NPYConsolidator,
    TIFFConsolidator,
    consolidator_factory,
)

warnings.warn(
    "The bluesky.consolidators module is deprecated. Please use bluesky_tiled_plugins.consolidators instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "consolidator_factory",
    "ConsolidatorBase",
    "CSVConsolidator",
    "JPEGConsolidator",
    "HDF5Consolidator",
    "MultipartRelatedConsolidator",
    "NPYConsolidator",
    "TIFFConsolidator",
]
