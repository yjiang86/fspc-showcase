from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class DisplayMapping:
    black_percentile: float = 0.5
    white_percentile: float = 99.5
    gamma: float = 1.0
    auto_range: bool = True


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    width: int
    height: int
    rowbytes: int
    pixel_type: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: bytes = b''


@dataclass(slots=True)
class PreviewStats:
    sensor_fps: float = 0.0
    display_fps: float = 0.0
    rendered_frames: int = 0
    dropped_frames: int = 0


class FrameRenderer(ABC):
    @abstractmethod
    def open(self, width: int, height: int, title: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, frame: FramePacket, mapping: DisplayMapping) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_stats(self, stats: PreviewStats) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
