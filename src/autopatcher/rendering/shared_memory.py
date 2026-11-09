from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from threading import Lock

from autopatcher.paths import runtime_dir

MAGIC = b'APT0'
HEADER_STRUCT = struct.Struct('<4sIIIIIIIIQQddd')
HEADER_SIZE = HEADER_STRUCT.size
DEFAULT_MAPPING_SIZE = 64 * 1024 * 1024
DEFAULT_BACKING_FILE = runtime_dir() / 'shared_memory' / 'fusionbt_latest.bin'


@dataclass(slots=True)
class SharedFrameHeader:
    active_slot: int
    slot_size: int
    width: int
    height: int
    rowbytes: int
    pixel_type: int
    frame_index: int
    payload_bytes: int
    version: int
    timestamp_ns: int
    black_percentile: float
    white_percentile: float
    gamma: float


def _prepare_backing_file(path: Path, mapping_size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as fh:
        fh.truncate(mapping_size)
    return path.resolve()


def clear_backing_file_header(backing_file: str | os.PathLike[str] | None = None, mapping_size: int = DEFAULT_MAPPING_SIZE) -> Path:
    path = _prepare_backing_file((Path(backing_file) if backing_file else DEFAULT_BACKING_FILE), mapping_size)
    with path.open('r+b', buffering=0) as fh:
        fh.seek(0)
        fh.write(b'\x00' * HEADER_SIZE)
        fh.flush()
        os.fsync(fh.fileno())
    return path


class SharedFramePublisher:
    def __init__(
        self,
        mapping_name: str = r'Local\\AutopatcherFusionBT',
        mapping_size: int = DEFAULT_MAPPING_SIZE,
        backing_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.mapping_name = mapping_name
        self.mapping_size = mapping_size
        self.backing_file = _prepare_backing_file(Path(backing_file) if backing_file else DEFAULT_BACKING_FILE, mapping_size)
        self.slot_size = (mapping_size - HEADER_SIZE) // 2
        self._lock = Lock()
        self._version = 0
        self._active_slot = 0

    def publish(
        self,
        *,
        width: int,
        height: int,
        rowbytes: int,
        pixel_type: int,
        frame_index: int,
        timestamp_ns: int,
        payload: bytes,
        black_percentile: float = 0.5,
        white_percentile: float = 99.5,
        gamma: float = 1.0,
    ) -> int:
        payload_bytes = len(payload)
        if payload_bytes > self.slot_size:
            raise ValueError(f'frame payload too large for slot: {payload_bytes} > {self.slot_size}')

        with self._lock:
            next_slot = 1 - self._active_slot
            self._version += 1
            payload_offset = HEADER_SIZE + next_slot * self.slot_size
            header = HEADER_STRUCT.pack(
                MAGIC,
                next_slot,
                self.slot_size,
                width,
                height,
                rowbytes,
                pixel_type,
                frame_index,
                payload_bytes,
                self._version,
                timestamp_ns,
                black_percentile,
                white_percentile,
                gamma,
            )
            with self.backing_file.open('r+b', buffering=0) as fh:
                fh.seek(payload_offset)
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
                fh.seek(0)
                fh.write(header)
                fh.flush()
                os.fsync(fh.fileno())
            self._active_slot = next_slot
            return self._version

    def close(self) -> None:
        return None


class SharedFrameReader:
    def __init__(
        self,
        mapping_name: str = r'Local\\AutopatcherFusionBT',
        mapping_size: int = DEFAULT_MAPPING_SIZE,
        backing_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.mapping_name = mapping_name
        self.mapping_size = mapping_size
        self.backing_file = (Path(backing_file) if backing_file else DEFAULT_BACKING_FILE).resolve()

    def read_latest(self) -> tuple[SharedFrameHeader, bytes] | None:
        try:
            with self.backing_file.open('rb') as fh:
                raw_header = fh.read(HEADER_SIZE)
                if len(raw_header) != HEADER_SIZE:
                    return None
                unpacked = HEADER_STRUCT.unpack(raw_header)
                magic = unpacked[0]
                if magic != MAGIC:
                    return None
                header = SharedFrameHeader(
                    active_slot=unpacked[1],
                    slot_size=unpacked[2],
                    width=unpacked[3],
                    height=unpacked[4],
                    rowbytes=unpacked[5],
                    pixel_type=unpacked[6],
                    frame_index=unpacked[7],
                    payload_bytes=unpacked[8],
                    version=unpacked[9],
                    timestamp_ns=unpacked[10],
                    black_percentile=unpacked[11],
                    white_percentile=unpacked[12],
                    gamma=unpacked[13],
                )
                payload_offset = HEADER_SIZE + header.active_slot * header.slot_size
                fh.seek(payload_offset)
                payload = fh.read(header.payload_bytes)
            return header, payload
        except OSError:
            return None

    def close(self) -> None:
        return None
