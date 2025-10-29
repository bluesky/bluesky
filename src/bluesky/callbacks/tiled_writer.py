import warnings

from bluesky_tiled_plugins import TiledWriter
from bluesky_tiled_plugins.tiled_writer import RunNormalizer

warnings.warn(
    "The bluesky.callbacks.tiled_writer module is deprecated. "
    "Please import TiledWriter from bluesky_tiled_plugins instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RunNormalizer", "TiledWriter"]
