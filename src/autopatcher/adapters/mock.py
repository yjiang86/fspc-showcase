from __future__ import annotations

from datetime import UTC, datetime
import time

from autopatcher.adapters.interfaces import CameraAdapter, CleaningAdapter, EjectionAdapter, EphysAdapter, FocusAdapter, ManipulatorAdapter, ManipulatorLimits, ObjectiveSelectorAdapter, PressureAdapter
from autopatcher.models.domain import CameraAcquisitionState, CameraDetections, CameraFrameGeometry, CameraImageReference, CameraObservation, CameraQualityState, CameraSubsystemStatus, EjectionEvent, EjectionEventType, EjectionTriggerMode, ElectrophysiologyReading, ObjectiveCommandMode, ObjectiveInterlockState, ObjectiveState, ObjectiveSwitchEvent, ObjectiveSwitchOutcome, ObjectiveVerificationSource, PressureReading, Vec3


class MockManipulator(ManipulatorAdapter):
    def __init__(self) -> None:
        self.position = Vec3(0.0, 0.0, 0.0)
        self.limits = ManipulatorLimits(
            minimum=Vec3(-10_000.0, -10_000.0, -10_000.0),
            maximum=Vec3(10_000.0, 10_000.0, 10_000.0),
        )

    def get_position(self) -> Vec3:
        return self.position

    def move_absolute(self, target: Vec3) -> None:
        self.position = target

    def move_relative(self, delta: Vec3) -> None:
        self.position = Vec3(
            self.position.x_um + delta.x_um,
            self.position.y_um + delta.y_um,
            self.position.z_um + delta.z_um,
        )

    def get_limits(self) -> ManipulatorLimits:
        return self.limits

    def stop(self) -> None:
        return None


class MockFocus(FocusAdapter):
    def __init__(self) -> None:
        self.position_um = 0.0

    def get_position_um(self) -> float:
        return self.position_um

    def move_relative_um(self, delta_um: float) -> None:
        self.position_um += delta_um


class MockPressure(PressureAdapter):
    def __init__(self) -> None:
        self.setpoint = 30.0

    def get_pressure(self, channel: int = 1) -> PressureReading:
        return PressureReading(setpoint_mbar=self.setpoint, measured_mbar=self.setpoint, channel=channel)

    def set_pressure(self, mbar: float, channel: int = 1) -> None:
        self.setpoint = mbar

    def pulse_pressure(self, mbar: float, duration_ms: int, channel: int = 1) -> None:
        self.setpoint = mbar

    def release_to_neutral(self, channel: int = 1) -> None:
        self.setpoint = 0.0


class MockEjection(EjectionAdapter):
    def __init__(self) -> None:
        self.last_event: EjectionEvent | None = None
        self.stopped = False

    def eject(
        self,
        duration_ms: int,
        *,
        trigger_context: str = "",
        trigger_mode: EjectionTriggerMode = EjectionTriggerMode.SOFTWARE,
    ) -> EjectionEvent:
        return self._make_event(
            EjectionEventType.EJECTION,
            duration_ms=duration_ms,
            trigger_context=trigger_context,
            trigger_mode=trigger_mode,
            status_detail="mock_ejection_complete",
        )

    def non_patch_flush(
        self,
        duration_ms: int,
        *,
        trigger_context: str = "",
        trigger_mode: EjectionTriggerMode = EjectionTriggerMode.SOFTWARE,
    ) -> EjectionEvent:
        return self._make_event(
            EjectionEventType.NON_PATCH_FLUSH,
            duration_ms=duration_ms,
            trigger_context=trigger_context,
            trigger_mode=trigger_mode,
            status_detail="mock_non_patch_flush_complete",
        )

    def stop(self) -> None:
        self.stopped = True

    def _make_event(
        self,
        event_type: EjectionEventType,
        *,
        duration_ms: int,
        trigger_context: str,
        trigger_mode: EjectionTriggerMode,
        status_detail: str,
    ) -> EjectionEvent:
        started_at = datetime.now(UTC)
        event = EjectionEvent(
            event_type=event_type,
            trigger_mode=trigger_mode,
            trigger_context=trigger_context,
            started_at=started_at,
            ended_at=started_at,
            pulse_duration_ms=duration_ms,
            status_detail=status_detail,
        )
        self.last_event = event
        self.stopped = False
        return event


class MockEphys(EphysAdapter):
    def __init__(self) -> None:
        self.resistance = 5.0

    def read(self) -> ElectrophysiologyReading:
        self.resistance += 250.0
        return ElectrophysiologyReading(seal_resistance_mohm=self.resistance, holding_current_pa=-35.0)

    def zap(self) -> None:
        return None


class MockCamera(CameraAdapter):
    def observe(self) -> CameraObservation:
        now = datetime.now(UTC)
        return CameraObservation(
            observation_id="mock-observation",
            frame_id="mock-frame",
            source="mock",
            camera_model="mock-camera",
            timestamp_host_utc=now,
            timestamp_monotonic_ns=time.time_ns(),
            frame_geometry=CameraFrameGeometry(width=2048, height=2048, pixel_format="mono16", bit_depth=16),
            acquisition=CameraAcquisitionState(exposure_ms=10.0, binning="1x1", streaming_state="mock"),
            quality=CameraQualityState(focus_score=0.8, stale=False),
            detections=CameraDetections(cell_probability=0.95, pipette_tip_probability=0.99, membrane_dimple_score=0.4),
            image_ref=CameraImageReference(),
            status=CameraSubsystemStatus(valid=True),
        )


class MockCleaning(CleaningAdapter):
    def move_to_cleaning_position(self) -> None:
        return None

    def execute_cleaning_cycle(self) -> None:
        return None


class MockObjectiveSelector(ObjectiveSelectorAdapter):
    def __init__(self) -> None:
        self._available = ["4x", "10x", "40x"]
        interlocks = ObjectiveInterlockState(
            manipulator_parked=True,
            pipette_clear=True,
            safe_z_ready=True,
            patch_phase_allows_switch=True,
            motion_quiescent=True,
            pressure_quiescent=True,
            camera_ready_for_resume=True,
            operator_override_present=False,
        )
        self._state = ObjectiveState(
            current_objective="10x",
            available_objectives=list(self._available),
            verified=True,
            verification_source=ObjectiveVerificationSource.INDEXED_MAPPING,
            switching_allowed=interlocks.automatic_switch_allowed(),
            interlock_state=interlocks,
        )

    def get_state(self) -> ObjectiveState:
        return ObjectiveState(
            current_objective=self._state.current_objective,
            available_objectives=list(self._state.available_objectives),
            verified=self._state.verified,
            verification_source=self._state.verification_source,
            switching_allowed=self._state.switching_allowed,
            interlock_state=self._state.interlock_state,
            last_switch_at=self._state.last_switch_at,
            last_requested_objective=self._state.last_requested_objective,
            last_rejected_reason=self._state.last_rejected_reason,
            settle_wait_ms=self._state.settle_wait_ms,
            settle_timer_cleared=self._state.settle_timer_cleared,
            status_detail=self._state.status_detail,
        )

    def list_available_objectives(self) -> list[str]:
        return list(self._available)

    def switch_to_objective(self, target: str, command_mode: ObjectiveCommandMode = ObjectiveCommandMode.AUTOMATIC) -> ObjectiveSwitchEvent:
        started_at = datetime.now(UTC)
        if target not in self._available:
            return ObjectiveSwitchEvent(
                requested_objective=target,
                previous_objective=self._state.current_objective,
                effective_objective=self._state.current_objective,
                command_mode=command_mode,
                verification_source=self._state.verification_source,
                outcome=ObjectiveSwitchOutcome.REJECTED,
                started_at=started_at,
                settled_at=started_at,
                interlock_state=self._state.interlock_state,
                rejected_reason="unknown_objective",
                operator_override_available=False,
                fault_code="unknown_objective",
            )

        previous = self._state.current_objective
        self._state.current_objective = target
        self._state.last_requested_objective = target
        self._state.last_switch_at = started_at
        self._state.settle_wait_ms = 250
        self._state.verified = True
        self._state.verification_source = ObjectiveVerificationSource.INDEXED_MAPPING
        self._state.last_rejected_reason = None
        self._state.settle_timer_cleared = True
        self._state.status_detail = "mock_switch_complete"
        return ObjectiveSwitchEvent(
            requested_objective=target,
            previous_objective=previous,
            effective_objective=target,
            command_mode=command_mode,
            verification_source=self._state.verification_source,
            outcome=ObjectiveSwitchOutcome.SETTLED,
            started_at=started_at,
            settled_at=started_at,
            settle_wait_ms=self._state.settle_wait_ms,
            interlock_state=self._state.interlock_state,
            operator_override_available=True,
        )

    def stop(self) -> None:
        self._state.status_detail = "stopped"
