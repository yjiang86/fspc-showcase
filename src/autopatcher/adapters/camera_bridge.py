from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from json import JSONDecodeError
from pathlib import Path
import time

from autopatcher.adapters.interfaces import CameraAdapter
from autopatcher.models.domain import CameraAcquisitionState, CameraDetections, CameraFrameGeometry, CameraImageReference, CameraObservation, CameraQualityState, CameraRoiGeometry, CameraSubsystemStatus


@dataclass(slots=True)
class HCImageBridgeConfig:
    latest_file: Path
    stale_after_seconds: float = 1.0
    source: str = "hcimage-bridge"


class HCImageBridgeCameraAdapter(CameraAdapter):
    """Read the latest camera metadata and analysis from an HCImage bridge file."""

    def __init__(self, config: HCImageBridgeConfig) -> None:
        self.config = config

    def observe(self) -> CameraObservation:
        now_utc = datetime.now(UTC)
        now_ns = time.time_ns()
        payload = self._load_payload()
        if payload is None:
            return CameraObservation(
                observation_id="hcimage-bridge-unavailable",
                frame_id=None,
                source=self.config.source,
                timestamp_host_utc=now_utc,
                timestamp_monotonic_ns=now_ns,
                frame_geometry=CameraFrameGeometry(0, 0),
                acquisition=CameraAcquisitionState(streaming_state="bridge"),
                quality=CameraQualityState(stale=True),
                status=CameraSubsystemStatus(valid=False, degraded_reason="invalid_bridge_payload"),
            )

        timestamp = self._parse_timestamp(payload.get("timestamp_utc"))
        metadata = dict(payload.get("metadata", {}))
        image_path = payload.get("image_path")
        stale = self._is_stale(timestamp, now_utc)
        degraded_reason = None
        valid = True
        if timestamp is None:
            stale = True
            valid = False
            degraded_reason = "missing_bridge_timestamp"
        elif stale:
            valid = False
            degraded_reason = "stale_bridge_payload"

        timestamp_monotonic_ns = self._parse_int(
            payload.get("timestamp_monotonic_ns", payload.get("timestamp_ns")),
            default=now_ns,
        )

        return CameraObservation(
            observation_id=payload.get("frame_id", "unknown-frame"),
            frame_id=payload.get("frame_id"),
            source=self.config.source,
            camera_model=metadata.get("camera_model"),
            timestamp_host_utc=timestamp or now_utc,
            timestamp_monotonic_ns=timestamp_monotonic_ns,
            timestamp_camera=self._parse_camera_timestamp(payload, metadata),
            frame_geometry=CameraFrameGeometry(
                width=self._parse_int(payload.get("width"), default=0),
                height=self._parse_int(payload.get("height"), default=0),
                pixel_format=payload.get("pixel_format") or metadata.get("pixel_format"),
                bit_depth=self._parse_int(payload.get("bit_depth", metadata.get("bit_depth"))),
            ),
            roi_geometry=CameraRoiGeometry(
                x=self._parse_int(payload.get("roi_x", metadata.get("roi_x"))),
                y=self._parse_int(payload.get("roi_y", metadata.get("roi_y"))),
                width=self._parse_int(payload.get("roi_width", metadata.get("roi_width"))),
                height=self._parse_int(payload.get("roi_height", metadata.get("roi_height"))),
            ),
            acquisition=CameraAcquisitionState(
                exposure_ms=metadata.get("exposure_ms"),
                binning=metadata.get("binning"),
                scan_mode=metadata.get("scan_mode"),
                streaming_state=payload.get("streaming_state", metadata.get("streaming_state", "bridge")),
            ),
            quality=CameraQualityState(
                focus_score=payload.get("focus_score"),
                saturation_fraction=payload.get("saturation_fraction"),
                sensor_temperature_c=metadata.get("sensor_temperature_c"),
                stale=stale,
            ),
            detections=CameraDetections(
                cell_probability=payload.get("cell_probability"),
                pipette_tip_probability=payload.get("pipette_tip_probability"),
                membrane_dimple_score=payload.get("membrane_dimple_score"),
            ),
            image_ref=CameraImageReference(path=image_path),
            status=CameraSubsystemStatus(valid=valid, degraded_reason=degraded_reason),
        )

    def _load_payload(self) -> dict | None:
        try:
            return json.loads(self.config.latest_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, JSONDecodeError):
            return None

    def _is_stale(self, timestamp: datetime | None, now_utc: datetime) -> bool:
        if timestamp is None:
            return True
        return now_utc - timestamp > timedelta(seconds=self.config.stale_after_seconds)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _parse_int(value, default: int | None = None) -> int | None:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _parse_camera_timestamp(self, payload: dict, metadata: dict) -> datetime | int | None:
        direct_value = payload.get("timestamp_camera", metadata.get("timestamp_camera"))
        if direct_value is not None:
            if isinstance(direct_value, str):
                parsed = self._parse_timestamp(direct_value)
                return parsed if parsed is not None else direct_value
            return direct_value

        sec = payload.get("camera_timestamp_sec", metadata.get("camera_timestamp_sec"))
        microsec = payload.get("camera_timestamp_microsec", metadata.get("camera_timestamp_microsec"))
        if sec is None or microsec is None:
            return None
        try:
            return int(sec) * 1_000_000 + int(microsec)
        except (TypeError, ValueError):
            return None
