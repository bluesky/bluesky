import os

BLUESKY_HISTORY_PATH: str | None = os.environ.get("BLUESKY_HISTORY_PATH", None)

BLUESKY_DEBUG_CALLBACKS = os.environ.get("BLUESKY_DEBUG_CALLBACKS", "") in ("1", "true")

BLUESKY_PREDECLARE = os.environ.get("BLUESKY_PREDECLARE", "") in ("1", "true")

BLUESKY_FORCE_READ_ALL_ONE_MSG_PER_DEVICE = os.environ.get("BLUESKY_FORCE_READ_ALL_ONE_MSG_PER_DEVICE", "") in (
    "1",
    "true",
)
