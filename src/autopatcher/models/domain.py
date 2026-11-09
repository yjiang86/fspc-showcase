from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class PatchState(str, Enum):
    IDLE = "idle"
    POSITIONING = "positioning"
    APPROACHING = "approaching"
    CONTACT_DETECTED = "contact_detected"
    SEAL_FORMING = "seal_forming"
    GIGASEAL = "gigaseal"
    BREAK_IN = "break_in"
    WHOLE_CELL = "whole_cell"
    CLEANING = "cleaning"
    FAILED = "failed"
    ABORTED = "aborted"


class Outcome(str, Enum):
    UNKNOWN = "unknown"
    CAMERA_NOT_READY = "camera_not_ready"
    NO_CELL = "no_cell"
    BAD_TARGET = "bad_target"
    CLOGGED = "clogged"
    CONTACT_NO_SEAL = "contact_no_seal"
    SEAL_FAILED = "seal_failed"
    BREAK_IN_FAILED = "break_in_failed"
    PRESSURE_RELEASE_FAILED = "pressure_release_failed"
    WHOLE_CELL_SUCCESS = "whole_cell_success"
    USER_ABORT = "user_abort"


class ActionType(str, Enum):
    HOLD = "hold"
    ADVANCE = "advance"
    RETRACT = "retract"
    SET_PRESSURE = "set_pressure"
    PULSE_PRESSURE = "pulse_pressure"
    EJECT = "eject"
    NON_PATCH_FLUSH = "non_patch_flush"
    SWITCH_OBJECTIVE = "switch_objective"
    ZAP = "zap"
    MOVE_TO_CLEAN = "move_to_clean"
    CLEAN = "clean"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass(slots=True)
class Vec3:
    x_um: float
    y_um: float
    z_um: float


@dataclass(slots=True)
class PressureReading:
    setpoint_mbar: float
    measured_mbar: float | None = None
    channel: int = 1


@dataclass(slots=True)
class ElectrophysiologyReading:
    seal_resistance_mohm: float | None = None
    membrane_voltage_mv: float | None = None
    holding_current_pa: float | None = None
    access_resistance_mohm: float | None = None
    capacitance_pf: float | None = None
    waveform_features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class VisionReading:
    focus_score: float | None = None
    cell_probability: float | None = None
    pipette_tip_probability: float | None = None
    membrane_dimple_score: float | None = None
    image_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CameraFrameGeometry:
    width: int
    height: int
    pixel_format: str | None = None
    bit_depth: int | None = None


@dataclass(slots=True)
class CameraRoiGeometry:
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class CameraAcquisitionState:
    exposure_ms: float | None = None
    binning: str | None = None
    scan_mode: str | None = None
    streaming_state: str | None = None


@dataclass(slots=True)
class CameraQualityState:
    focus_score: float | None = None
    saturation_fraction: float | None = None
    sensor_temperature_c: float | None = None
    stale: bool = False


@dataclass(slots=True)
class CameraDetections:
    cell_probability: float | None = None
    pipette_tip_probability: float | None = None
    membrane_dimple_score: float | None = None
    cell_position_px: tuple[float, float] | None = None
    pipette_tip_position_px: tuple[float, float] | None = None


@dataclass(slots=True)
class CameraImageReference:
    path: str | None = None
    shared_memory_name: str | None = None
    shared_frame_version: int | None = None


@dataclass(slots=True)
class CameraSubsystemStatus:
    valid: bool = True
    degraded_reason: str | None = None


class EjectionEventType(str, Enum):
    EJECTION = "ejection"
    NON_PATCH_FLUSH = "non_patch_flush"


class EjectionTriggerMode(str, Enum):
    SOFTWARE = "software_triggered"
    SEMI_SOFTWARE = "semi_software_triggered"
    MANUAL = "manual"


@dataclass(slots=True)
class EjectionEvent:
    event_type: EjectionEventType
    hardware: str = "Picospritzer III"
    trigger_mode: EjectionTriggerMode = EjectionTriggerMode.SOFTWARE
    trigger_context: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    pulse_duration_ms: int | None = None
    pressure_path_separated: bool = True
    operator_note: str | None = None
    status_detail: str | None = None


class ObjectiveCommandMode(str, Enum):
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL_OVERRIDE = "manual_override"


class ObjectiveVerificationSource(str, Enum):
    UNKNOWN = "unknown"
    TURRET_SENSOR = "turret_sensor"
    SERVO_CONTROLLER = "servo_controller"
    INDEXED_MAPPING = "indexed_mapping"
    OPERATOR_CONFIRMATION = "operator_confirmation"


class ObjectiveSwitchOutcome(str, Enum):
    ACCEPTED = "accepted"
    MOVING = "moving"
    SETTLED = "settled"
    TIMEOUT = "timeout"
    FAULT = "fault"
    REJECTED = "rejected"


@dataclass(slots=True)
class ObjectiveInterlockState:
    manipulator_parked: bool = False
    pipette_clear: bool = False
    safe_z_ready: bool = False
    patch_phase_allows_switch: bool = False
    motion_quiescent: bool = False
    pressure_quiescent: bool = False
    camera_ready_for_resume: bool = False
    operator_override_present: bool = False

    def automatic_switch_allowed(self) -> bool:
        return (
            self.manipulator_parked
            and self.pipette_clear
            and self.safe_z_ready
            and self.patch_phase_allows_switch
            and self.motion_quiescent
            and self.pressure_quiescent
            and self.camera_ready_for_resume
        )

    def failed_fields(self) -> list[str]:
        failed: list[str] = []
        if not self.manipulator_parked:
            failed.append("manipulator_parked")
        if not self.pipette_clear:
            failed.append("pipette_clear")
        if not self.safe_z_ready:
            failed.append("safe_z_ready")
        if not self.patch_phase_allows_switch:
            failed.append("patch_phase_allows_switch")
        if not self.motion_quiescent:
            failed.append("motion_quiescent")
        if not self.pressure_quiescent:
            failed.append("pressure_quiescent")
        if not self.camera_ready_for_resume:
            failed.append("camera_ready_for_resume")
        return failed


@dataclass(slots=True)
class ObjectiveState:
    current_objective: str | None = None
    available_objectives: list[str] = field(default_factory=list)
    verified: bool = False
    verification_source: ObjectiveVerificationSource = ObjectiveVerificationSource.UNKNOWN
    switching_allowed: bool = True
    interlock_state: ObjectiveInterlockState = field(default_factory=ObjectiveInterlockState)
    last_switch_at: datetime | None = None
    last_requested_objective: str | None = None
    last_rejected_reason: str | None = None
    settle_wait_ms: int | None = None
    settle_timer_cleared: bool = True
    status_detail: str | None = None


@dataclass(slots=True)
class ObjectiveSwitchEvent:
    requested_objective: str
    previous_objective: str | None = None
    effective_objective: str | None = None
    command_mode: ObjectiveCommandMode = ObjectiveCommandMode.AUTOMATIC
    verification_source: ObjectiveVerificationSource = ObjectiveVerificationSource.UNKNOWN
    outcome: ObjectiveSwitchOutcome = ObjectiveSwitchOutcome.ACCEPTED
    started_at: datetime | None = None
    settled_at: datetime | None = None
    settle_wait_ms: int | None = None
    interlock_state: ObjectiveInterlockState = field(default_factory=ObjectiveInterlockState)
    rejected_reason: str | None = None
    operator_override_available: bool | None = None
    fault_code: str | None = None
    operator_note: str | None = None


@dataclass(slots=True)
class CameraObservation:
    observation_id: str
    timestamp_host_utc: datetime
    timestamp_monotonic_ns: int
    source: str
    frame_id: str | None = None
    camera_model: str | None = None
    timestamp_camera: datetime | int | None = None
    frame_geometry: CameraFrameGeometry = field(default_factory=lambda: CameraFrameGeometry(0, 0))
    roi_geometry: CameraRoiGeometry = field(default_factory=CameraRoiGeometry)
    acquisition: CameraAcquisitionState = field(default_factory=CameraAcquisitionState)
    quality: CameraQualityState = field(default_factory=CameraQualityState)
    detections: CameraDetections = field(default_factory=CameraDetections)
    image_ref: CameraImageReference = field(default_factory=CameraImageReference)
    rig_state_ref: str | None = None
    status: CameraSubsystemStatus = field(default_factory=CameraSubsystemStatus)

    def to_vision_reading(self) -> VisionReading:
        metadata: dict[str, Any] = {}
        if self.acquisition.exposure_ms is not None:
            metadata["exposure_ms"] = self.acquisition.exposure_ms
        if self.acquisition.binning is not None:
            metadata["binning"] = self.acquisition.binning
        if self.acquisition.scan_mode is not None:
            metadata["scan_mode"] = self.acquisition.scan_mode
        if self.camera_model is not None:
            metadata["camera_model"] = self.camera_model
        if self.image_ref.path is not None:
            metadata["image_path"] = self.image_ref.path
        if self.quality.stale:
            metadata["stale"] = True
        if self.quality.sensor_temperature_c is not None:
            metadata["sensor_temperature_c"] = self.quality.sensor_temperature_c
        if self.status.degraded_reason is not None:
            metadata["degraded_reason"] = self.status.degraded_reason
        return VisionReading(
            focus_score=self.quality.focus_score,
            cell_probability=self.detections.cell_probability,
            pipette_tip_probability=self.detections.pipette_tip_probability,
            membrane_dimple_score=self.detections.membrane_dimple_score,
            image_id=self.frame_id,
            metadata=metadata,
        )


@dataclass(slots=True)
class RigSnapshot:
    timestamp: datetime
    patch_state: PatchState
    manipulator_position: Vec3
    focus_position_um: float
    pressure: PressureReading
    ephys: ElectrophysiologyReading
    vision: VisionReading
    camera_observation: CameraObservation | None = None
    objective_state: ObjectiveState | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ControlAction:
    action_type: ActionType
    value: float | Vec3 | str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    issued_at: datetime | None = None
    issued_monotonic_ns: int | None = None
    completed_at: datetime | None = None
    completed_monotonic_ns: int | None = None


@dataclass(slots=True)
class SessionEvent:
    timestamp: datetime
    level: Literal["debug", "info", "warning", "error"]
    message: str
    state: PatchState
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptRecord:
    attempt_id: str
    started_at: datetime
    target_id: str | None = None
    outcome: Outcome = Outcome.UNKNOWN
    events: list[SessionEvent] = field(default_factory=list)
    actions: list[ControlAction] = field(default_factory=list)
    snapshots: list[RigSnapshot] = field(default_factory=list)
