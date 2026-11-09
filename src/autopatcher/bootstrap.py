from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from autopatcher.adapters.interfaces import ManipulatorAdapter, ManipulatorLimits
from autopatcher.adapters.camera_bridge import HCImageBridgeCameraAdapter, HCImageBridgeConfig
from autopatcher.adapters.mock import MockCamera, MockCleaning, MockEjection, MockEphys, MockFocus, MockManipulator, MockObjectiveSelector, MockPressure
from autopatcher.adapters.phidget_objective_selector import PhidgetObjectiveSelectorAdapter, PhidgetObjectiveSelectorConfig
from autopatcher.adapters.shared_memory_camera import SharedMemoryCameraAdapter, SharedMemoryCameraConfig
from autopatcher.adapters.sutter_manipulator import SutterManipulatorAdapter, SutterManipulatorConfig, load_protocol_from_json
from autopatcher.core.baseline import RigAdapters
from autopatcher.hardware_safety import LiveHardwareAuthorization, require_live_hardware_authorization
from autopatcher.models.domain import ObjectiveInterlockState, ObjectiveVerificationSource, Vec3
from autopatcher.paths import resolve_runtime_path


class ExecutionMode(str, Enum):
    MOCK = "mock"
    REPLAY = "replay"
    LIVE = "live"


class AppConfigSafetyError(ValueError):
    """Raised before adapter construction when a rig configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class AppConfigSpec:
    source_path: Path
    payload: dict[str, Any]
    execution_mode: ExecutionMode
    output_dir: Path
    target_id: str


@dataclass(slots=True)
class AppConfig:
    output_dir: Path
    target_id: str
    rig_adapters: RigAdapters
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    camera_ready_timeout_seconds: float = 3.0
    camera_ready_poll_seconds: float = 0.1


def parse_app_config(path: Path) -> AppConfigSpec:
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AppConfigSafetyError("app config root must be a JSON object")
    try:
        execution_mode = ExecutionMode(str(payload.get("execution_mode", ExecutionMode.MOCK.value)))
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ExecutionMode)
        raise AppConfigSafetyError(f"execution_mode must be one of: {allowed}") from exc
    return AppConfigSpec(
        source_path=source_path,
        payload=payload,
        execution_mode=execution_mode,
        output_dir=resolve_runtime_path(
            payload.get("output_dir", "runs"),
            base_dir=source_path.parent,
        ),
        target_id=str(payload.get("target_id", "demo-cell-001")),
    )


def materialize_app_config(
    spec: AppConfigSpec,
    *,
    authorize_live_hardware: bool = False,
    allow_bench_mixed_rig: bool = False,
) -> AppConfig:
    payload = spec.payload
    _validate_execution_boundary(
        payload,
        spec.execution_mode,
        authorize_live_hardware=authorize_live_hardware,
        allow_bench_mixed_rig=allow_bench_mixed_rig,
    )

    live_authorization: LiveHardwareAuthorization | None = None
    if spec.execution_mode is ExecutionMode.LIVE:
        live_authorization = require_live_hardware_authorization(
            authorize_live_hardware,
            operation=f"materialize app config {spec.source_path.name}",
        )

    controller_payload = payload.get("controller", {})
    manipulator = _load_manipulator(
        payload.get("manipulator", {}),
        base_dir=spec.source_path.parent,
        live_authorization=live_authorization,
    )
    rig_adapters = RigAdapters(
        manipulator=manipulator,
        focus=MockFocus(),
        pressure=MockPressure(),
        ejection=_load_ejection(payload.get("ejection", {})),
        ephys=MockEphys(),
        camera=_load_camera(payload.get("camera", {}), base_dir=spec.source_path.parent),
        cleaning=MockCleaning(),
        objective_selector=_load_objective_selector(payload.get("objective_selector", {})),
    )
    return AppConfig(
        output_dir=spec.output_dir,
        target_id=spec.target_id,
        rig_adapters=rig_adapters,
        execution_mode=spec.execution_mode,
        camera_ready_timeout_seconds=float(controller_payload.get("camera_ready_timeout_seconds", 3.0)),
        camera_ready_poll_seconds=float(controller_payload.get("camera_ready_poll_seconds", 0.1)),
    )


def load_app_config(
    path: Path,
    *,
    authorize_live_hardware: bool = False,
    allow_bench_mixed_rig: bool = False,
) -> AppConfig:
    return materialize_app_config(
        parse_app_config(path),
        authorize_live_hardware=authorize_live_hardware,
        allow_bench_mixed_rig=allow_bench_mixed_rig,
    )


def build_demo_config(
    output_dir: str | Path | None = None,
    target_id: str = "demo-cell-001",
) -> AppConfig:
    return AppConfig(
        output_dir=resolve_runtime_path(output_dir or "runs"),
        target_id=target_id,
        execution_mode=ExecutionMode.MOCK,
        rig_adapters=RigAdapters(
            manipulator=MockManipulator(),
            focus=MockFocus(),
            pressure=MockPressure(),
            ejection=MockEjection(),
            ephys=MockEphys(),
            camera=MockCamera(),
            cleaning=MockCleaning(),
            objective_selector=MockObjectiveSelector(),
        ),
        camera_ready_timeout_seconds=3.0,
        camera_ready_poll_seconds=0.1,
    )


def close_rig_adapters(rig_adapters: RigAdapters) -> None:
    for adapter in (
        rig_adapters.manipulator,
        rig_adapters.focus,
        rig_adapters.pressure,
        rig_adapters.ejection,
        rig_adapters.ephys,
        rig_adapters.camera,
        rig_adapters.cleaning,
        rig_adapters.objective_selector,
    ):
        if adapter is not None:
            close_method = getattr(adapter, "close", None)
            if callable(close_method):
                close_method()


def _load_manipulator(
    payload: dict[str, Any],
    base_dir: Path,
    live_authorization: LiveHardwareAuthorization | None,
) -> ManipulatorAdapter:
    manipulator_type = payload.get("type", "mock")
    if manipulator_type == "mock":
        return MockManipulator()
    if manipulator_type != "sutter_manipulator":
        raise ValueError(f"Unsupported manipulator type: {manipulator_type}")

    protocol_path = _resolve_path(base_dir, payload["protocol_json"])
    protocol = load_protocol_from_json(protocol_path)
    limits_payload = payload.get("limits_um")
    if not isinstance(limits_payload, dict):
        raise AppConfigSafetyError("live Sutter configuration requires explicit limits_um")
    limits = ManipulatorLimits(
        minimum=_load_required_vec3(limits_payload.get("minimum"), field="limits_um.minimum"),
        maximum=_load_required_vec3(limits_payload.get("maximum"), field="limits_um.maximum"),
    )
    if any(
        minimum >= maximum
        for minimum, maximum in (
            (limits.minimum.x_um, limits.maximum.x_um),
            (limits.minimum.y_um, limits.maximum.y_um),
            (limits.minimum.z_um, limits.maximum.z_um),
        )
    ):
        raise AppConfigSafetyError("each live Sutter minimum limit must be below its maximum")
    return SutterManipulatorAdapter(
        SutterManipulatorConfig(
            port=payload.get("port"),
            protocol=protocol,
            limits_um=limits,
            baudrate=int(payload.get("baudrate", 128000)),
            timeout_seconds=float(payload.get("timeout_seconds", 1.0)),
            transport_backend=str(payload.get("transport_backend", "serial")),
            d2xx_device_index=int(payload.get("d2xx_device_index", 0)),
            d2xx_dll_path=_resolve_path(base_dir, payload["d2xx_dll_path"]) if payload.get("d2xx_dll_path") else None,
        ),
        live_authorization=live_authorization,
    )


def _validate_execution_boundary(
    payload: dict[str, Any],
    execution_mode: ExecutionMode,
    *,
    authorize_live_hardware: bool,
    allow_bench_mixed_rig: bool,
) -> None:
    manipulator_type = str(payload.get("manipulator", {}).get("type", "mock"))
    camera_payload = payload.get("camera", {})
    camera_type = str(camera_payload.get("type", "mock"))
    selector_payload = payload.get("objective_selector", {})
    selector_hardware = bool(selector_payload.get("allow_hardware", False))
    real_manipulator = manipulator_type != "mock"
    live_camera = camera_type == "shared_memory" and not camera_payload.get("backing_file")

    if execution_mode is ExecutionMode.MOCK:
        if real_manipulator or camera_type != "mock" or selector_hardware:
            raise AppConfigSafetyError(
                "execution_mode=mock permits only mock actuators and observations"
            )
        return

    if execution_mode is ExecutionMode.REPLAY:
        if real_manipulator or selector_hardware or live_camera:
            raise AppConfigSafetyError(
                "execution_mode=replay forbids live actuators and live shared-memory mappings"
            )
        return

    if not authorize_live_hardware:
        raise AppConfigSafetyError(
            "execution_mode=live requires caller-side live hardware authorization"
        )
    if real_manipulator and not allow_bench_mixed_rig:
        raise AppConfigSafetyError(
            "the baseline controller cannot combine a live manipulator with mock "
            "focus, pressure, electrophysiology, or cleaning adapters"
        )


def _load_camera(payload: dict[str, Any], base_dir: Path):
    camera_type = payload.get("type", "mock")
    if camera_type == "mock":
        return MockCamera()
    if camera_type == "hcimage_bridge":
        latest_file = _resolve_path(base_dir, payload["latest_file"])
        return HCImageBridgeCameraAdapter(
            HCImageBridgeConfig(
                latest_file=latest_file,
                stale_after_seconds=float(payload.get("stale_after_seconds", 1.0)),
            )
        )
    if camera_type == "shared_memory":
        backing_file = payload.get("backing_file")
        metadata_file = payload.get("metadata_file")
        return SharedMemoryCameraAdapter(
            SharedMemoryCameraConfig(
                mapping_name=payload.get("mapping_name", r"Local\\AutopatcherFusionBT"),
                mapping_size=int(payload.get("mapping_size", 64 * 1024 * 1024)),
                backing_file=_resolve_path(base_dir, backing_file) if backing_file else None,
                metadata_file=_resolve_path(base_dir, metadata_file) if metadata_file else None,
                stale_after_seconds=float(payload.get("stale_after_seconds", 1.0)),
                source=payload.get("source", "dcam-shared-memory"),
            )
        )
    raise ValueError(f"Unsupported camera type: {camera_type}")


def _load_ejection(payload: dict[str, Any]):
    ejection_type = payload.get("type", "mock")
    if ejection_type == "mock":
        return MockEjection()
    raise ValueError(f"Unsupported ejection type: {ejection_type}")


def _load_objective_selector(payload: dict[str, Any]):
    selector_type = payload.get("type", "mock")
    if selector_type == "mock":
        return MockObjectiveSelector()
    if selector_type != "phidget_objective_selector":
        raise ValueError(f"Unsupported objective selector type: {selector_type}")

    verification_source_raw = payload.get("verification_source", ObjectiveVerificationSource.SERVO_CONTROLLER.value)
    interlock_payload = payload.get("interlocks", {})
    interlock_state = ObjectiveInterlockState(
        manipulator_parked=bool(interlock_payload.get("manipulator_parked", True)),
        pipette_clear=bool(interlock_payload.get("pipette_clear", True)),
        safe_z_ready=bool(interlock_payload.get("safe_z_ready", True)),
        patch_phase_allows_switch=bool(interlock_payload.get("patch_phase_allows_switch", True)),
        motion_quiescent=bool(interlock_payload.get("motion_quiescent", True)),
        pressure_quiescent=bool(interlock_payload.get("pressure_quiescent", True)),
        camera_ready_for_resume=bool(interlock_payload.get("camera_ready_for_resume", True)),
        operator_override_present=bool(interlock_payload.get("operator_override_present", False)),
    )
    return PhidgetObjectiveSelectorAdapter(
        PhidgetObjectiveSelectorConfig(
            available_objectives=list(payload.get("available_objectives", ["4x", "10x", "40x"])),
            initial_objective=payload.get("initial_objective", "10x"),
            settle_wait_ms=int(payload.get("settle_wait_ms", 250)),
            allow_hardware=bool(payload.get("allow_hardware", False)),
            device_serial_number=payload.get("device_serial_number"),
            hub_port=payload.get("hub_port"),
            channel=payload.get("channel"),
            verification_source=ObjectiveVerificationSource(verification_source_raw),
            interlock_state=interlock_state,
            operator_override_available=bool(payload.get("operator_override_available", True)),
        )
    )


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    return resolve_runtime_path(raw_path, base_dir=base_dir)


def _load_vec3(payload: dict[str, Any] | None, fallback: Vec3) -> Vec3:
    if payload is None:
        return fallback
    return Vec3(
        x_um=float(payload.get("x_um", fallback.x_um)),
        y_um=float(payload.get("y_um", fallback.y_um)),
        z_um=float(payload.get("z_um", fallback.z_um)),
    )


def _load_required_vec3(payload: Any, *, field: str) -> Vec3:
    if not isinstance(payload, dict):
        raise AppConfigSafetyError(f"{field} must be an object with x_um, y_um, and z_um")
    missing = [axis for axis in ("x_um", "y_um", "z_um") if axis not in payload]
    if missing:
        raise AppConfigSafetyError(f"{field} is missing required fields: {', '.join(missing)}")
    return Vec3(
        x_um=float(payload["x_um"]),
        y_um=float(payload["y_um"]),
        z_um=float(payload["z_um"]),
    )
