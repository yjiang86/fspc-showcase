from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from autopatcher.rendering.interfaces import FramePacket


@dataclass(slots=True)
class LatestFrameSnapshot:
    frame: FramePacket | None
    version: int
    dropped_frames: int


class LatestFrameStore:
    """Single-producer/latest-consumer store for high-rate camera preview.

    The camera thread is allowed to overwrite stale frames. The display thread only
    cares about the newest frame available, which mirrors how HCImage-style live
    preview pipelines stay responsive under high sensor frame rates.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: FramePacket | None = None
        self._version = 0
        self._dropped = 0

    def publish(self, frame: FramePacket) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame
            self._version += 1

    def snapshot(self) -> LatestFrameSnapshot:
        with self._lock:
            return LatestFrameSnapshot(
                frame=self._frame,
                version=self._version,
                dropped_frames=self._dropped,
            )
