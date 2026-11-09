from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
from json import JSONDecodeError
import time

from autopatcher.adapters.interfaces import CameraAdapter
from autopatcher.models.domain import CameraAcquisitionState, CameraFrameGeometry, CameraImageReference, CameraObservation, CameraQualityState, CameraRoiGeometry, CameraSubsystemStatus
from autopatcher.rendering.shared_memory import SharedFrameReader


@dataclass(slots=True)
class SharedMemoryCameraConfig:
    mapping_name: str = r"Local\\AutopatcherFusionBT"
    mapping_size: int = 64 * 1024 * 1024
    backing_file: Path | None = None
    metadata_file: Path | None = None
    stale_after_seconds: float = 1.0
    source: str = "dcam-shared-memory"


class SharedMemoryCameraAdapter(CameraAdapter):
    """Read the latest controller-facing camera observation from the native shared-frame path."""

    def __init__(self, config: SharedMemoryCameraConfig) -> None:
        self.config = config
        self.reader = SharedFrameReader(
            mapping_name=config.mapping_name,
            mapping_size=config.mapping_size,
            backing_file=config.backing_file,
        )

    def observe(self) -> CameraObservation:
        latest = self.reader.read_latest()
        metadata = self._load_metadata()
        now_utc = datetime.now(UTC)
        now_ns = time.time_ns()

        if latest is None:
            return CameraObservation(
                observation_id="shared-memory-unavailable",
                frame_id=None,
                source=self.config.source,
                camera_model=metadata.get("camera_model"),
                timestamp_host_utc=now_utc,
                timestamp_monotonic_ns=now_ns,
                frame_geometry=CameraFrameGeometry(0, 0),
                acquisition=self._build_acquisition(metadata),
                image_ref=CameraImageReference(shared_memory_name=self.config.mapping_name),
                quality=CameraQualityState(stale=True),
                status=CameraSubsystemStatus(valid=False, degraded_reason="no_shared_frame"),
            )

        header, _payload = latest
        frame_timestamp = datetime.fromtimestamp(header.timestamp_ns / 1_000_000_000, UTC)
        stale = now_utc - frame_timestamp > timedelta(seconds=self.config.stale_after_seconds)
        degraded_reason = "stale_shared_frame" if stale else None
        bit_depth = self._infer_bit_depth(header.rowbytes, header.width)
        frame_id = metadata.get("frame_id", f"shared-frame-{header.frame_index}")
        timestamp_camera = self._build_camera_timestamp(metadata)
        return CameraObservation(
            observation_id=f"shared-frame-v{header.version}",
            frame_id=frame_id,
            source=self.config.source,
            camera_model=metadata.get("camera_model"),
            timestamp_host_utc=frame_timestamp,
            timestamp_monotonic_ns=header.timestamp_ns,
            timestamp_camera=timestamp_camera,
            frame_geometry=CameraFrameGeometry(
                width=header.width,
                height=header.height,
                pixel_format=metadata.get("pixel_format", f"pixel_type_{header.pixel_type}"),
                bit_depth=bit_depth,
            ),
            roi_geometry=CameraRoiGeometry(
                x=metadata.get("roi_x"),
                y=metadata.get("roi_y"),
                width=metadata.get("roi_width"),
                height=metadata.get("roi_height"),
            ),
            acquisition=self._build_acquisition(metadata),
            quality=CameraQualityState(
                focus_score=metadata.get("focus_score"),
                saturation_fraction=metadata.get("saturation_fraction"),
                sensor_temperature_c=metadata.get("sensor_temperature_c"),
                stale=stale,
            ),
            image_ref=CameraImageReference(
                path=metadata.get("image_path"),
                shared_memory_name=self.config.mapping_name,
                shared_frame_version=header.version,
            ),
            status=CameraSubsystemStatus(valid=not stale, degraded_reason=degraded_reason),
        )

    def close(self) -> None:
        self.reader.close()

    def _load_metadata(self) -> dict:
        if self.config.metadata_file is None or not self.config.metadata_file.exists():
            return {}
        try:
            return json.loads(self.config.metadata_file.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _infer_bit_depth(rowbytes: int, width: int) -> int | None:
        if width <= 0:
            return None
        bytes_per_pixel = abs(rowbytes) // width
        if bytes_per_pixel == 2:
            return 16
        if bytes_per_pixel == 1:
            return 8
        return None

    @staticmethod
    def _build_acquisition(metadata: dict) -> CameraAcquisitionState:
        return CameraAcquisitionState(
            exposure_ms=metadata.get("exposure_ms"),
            binning=metadata.get("binning"),
            scan_mode=metadata.get("scan_mode"),
            streaming_state=metadata.get("streaming_state", "shared-memory"),
        )

    @staticmethod
    def _build_camera_timestamp(metadata: dict):
        sec = metadata.get("camera_timestamp_sec")
        microsec = metadata.get("camera_timestamp_microsec")
        if sec is None or microsec is None:
            return None
        try:
            return int(sec) * 1_000_000 + int(microsec)
        except (TypeError, ValueError):
            return None
