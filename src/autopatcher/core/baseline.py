from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Callable, Protocol
from uuid import uuid4

from autopatcher.adapters.interfaces import CameraAdapter, CleaningAdapter, EjectionAdapter, EphysAdapter, FocusAdapter, ManipulatorAdapter, ObjectiveSelectorAdapter, PressureAdapter
from autopatcher.models.domain import ActionType, AttemptRecord, CameraObservation, ControlAction, Outcome, PatchState, RigSnapshot, SessionEvent, Vec3


@dataclass(slots=True)
class BaselineThresholds:
    contact_resistance_mohm: float = 20.0
    gigaseal_resistance_mohm: float = 1000.0
    minimum_focus_score: float = 0.2
    maximum_seal_seconds: float = 30.0


@dataclass(slots=True)
class RigAdapters:
    manipulator: ManipulatorAdapter
    focus: FocusAdapter
    pressure: PressureAdapter
    ejection: EjectionAdapter | None
    ephys: EphysAdapter
    camera: CameraAdapter
    cleaning: CleaningAdapter
    objective_selector: ObjectiveSelectorAdapter | None = None


class BaselineControllerObserver(Protocol):
    def on_attempt_started(self, attempt: AttemptRecord) -> None: ...
    def on_event(self, attempt: AttemptRecord, event: SessionEvent) -> None: ...
    def on_action(self, attempt: AttemptRecord, action: ControlAction) -> None: ...
    def on_snapshot(self, attempt: AttemptRecord, snapshot: RigSnapshot) -> None: ...
    def on_attempt_finished(self, attempt: AttemptRecord) -> None: ...


class CameraObservationNotReady(RuntimeError):
    def __init__(self, observation: CameraObservation) -> None:
        self.observation = observation
        reason = observation.status.degraded_reason or "camera_not_ready"
        super().__init__(f"camera observation is not usable: {reason}")


class PressureReleaseError(RuntimeError):
    pass


class BaselineAutopatchController:
    def __init__(
        self,
        adapters: RigAdapters,
        thresholds: BaselineThresholds | None = None,
        observer: BaselineControllerObserver | None = None,
        camera_ready_timeout_seconds: float = 3.0,
        camera_ready_poll_seconds: float = 0.1,
    ) -> None:
        self.adapters = adapters
        self.thresholds = thresholds or BaselineThresholds()
        self.state = PatchState.IDLE
        self.observer = observer
        self.camera_ready_timeout_seconds = camera_ready_timeout_seconds
        self.camera_ready_poll_seconds = camera_ready_poll_seconds

    def start_attempt(self, target_id: str | None = None) -> AttemptRecord:
        now = datetime.now(UTC)
        attempt = AttemptRecord(
            attempt_id=now.strftime("attempt-%Y%m%d-%H%M%S-") + uuid4().hex[:8],
            started_at=now,
            target_id=target_id,
        )
        self._log(attempt, "info", "Attempt initialized", PatchState.IDLE)
        camera_observation = self._await_camera_ready(attempt)
        if not camera_observation.status.valid or camera_observation.quality.stale:
            reason = camera_observation.status.degraded_reason or "camera_not_ready"
            self._fail(
                attempt,
                Outcome.CAMERA_NOT_READY,
                f"Camera observation not ready at attempt start: {reason}",
                camera_observation=camera_observation,
            )
        else:
            self._snapshot(attempt, camera_observation)
        self._notify_attempt_started(attempt)
        return attempt

    def run_baseline_sequence(self, attempt: AttemptRecord) -> AttemptRecord:
        try:
            if attempt.outcome is Outcome.CAMERA_NOT_READY or self.state is PatchState.FAILED:
                return attempt
            self._transition(attempt, PatchState.POSITIONING, "Positioning over selected target")
            self._ensure_focus(attempt)
            self._transition(attempt, PatchState.APPROACHING, "Beginning pipette approach")
            self._approach_until_contact(attempt)

            ephys = self.adapters.ephys.read()
            if (ephys.seal_resistance_mohm or 0.0) < self.thresholds.contact_resistance_mohm:
                self._fail(attempt, Outcome.CONTACT_NO_SEAL, "Contact threshold not achieved")
                return attempt

            self._transition(attempt, PatchState.SEAL_FORMING, "Forming seal with negative pressure")
            self._apply_seal_protocol(attempt)

            ephys = self.adapters.ephys.read()
            if (ephys.seal_resistance_mohm or 0.0) < self.thresholds.gigaseal_resistance_mohm:
                self._fail(attempt, Outcome.SEAL_FAILED, "Gigaseal threshold not achieved")
                return attempt

            self._transition(attempt, PatchState.GIGASEAL, "Gigaseal achieved")
            self._transition(attempt, PatchState.BREAK_IN, "Applying break-in")
            self._require_camera_ready()
            self._execute_and_record_action(
                attempt,
                ControlAction(ActionType.ZAP, reason="Baseline break-in action"),
                self.adapters.ephys.zap,
            )
            self._snapshot(attempt)

            self._transition(attempt, PatchState.WHOLE_CELL, "Whole-cell state reached")
            attempt.outcome = Outcome.WHOLE_CELL_SUCCESS
            self._record_action(attempt, ControlAction(ActionType.COMPLETE, reason="Baseline sequence completed"))
            return attempt
        except CameraObservationNotReady as exc:
            reason = exc.observation.status.degraded_reason or "camera_not_ready"
            self._fail(
                attempt,
                Outcome.CAMERA_NOT_READY,
                f"Camera observation became unusable before actuator motion: {reason}",
                camera_observation=exc.observation,
            )
            return attempt
        finally:
            try:
                self._release_pressure_to_neutral(attempt)
            finally:
                self._notify_attempt_finished(attempt)

    def abort_and_clean(self, attempt: AttemptRecord, reason: str) -> AttemptRecord:
        try:
            self._transition(attempt, PatchState.CLEANING, reason)
            self._execute_and_record_action(
                attempt,
                ControlAction(ActionType.MOVE_TO_CLEAN, reason=reason),
                self.adapters.cleaning.move_to_cleaning_position,
            )
            self._execute_and_record_action(
                attempt,
                ControlAction(ActionType.CLEAN, reason="Cleaning cycle executed"),
                self.adapters.cleaning.execute_cleaning_cycle,
            )
            self._transition(attempt, PatchState.ABORTED, "Attempt aborted after cleaning")
            attempt.outcome = Outcome.USER_ABORT
            self._snapshot(attempt)
            return attempt
        finally:
            try:
                self._release_pressure_to_neutral(attempt)
            finally:
                self._notify_attempt_finished(attempt)

    def _ensure_focus(self, attempt: AttemptRecord) -> None:
        observation = self._require_camera_ready()
        vision = observation.to_vision_reading()
        if (vision.focus_score or 0.0) >= self.thresholds.minimum_focus_score:
            self._snapshot(attempt, observation)
            return
        self._execute_and_record_action(
            attempt,
            ControlAction(ActionType.HOLD, value=5.0, reason="Focus score below threshold; nudged Z"),
            lambda: self.adapters.focus.move_relative_um(5.0),
        )
        self._snapshot(attempt)

    def _approach_until_contact(self, attempt: AttemptRecord) -> None:
        for _ in range(5):
            ephys = self.adapters.ephys.read()
            if (ephys.seal_resistance_mohm or 0.0) >= self.thresholds.contact_resistance_mohm:
                self._transition(attempt, PatchState.CONTACT_DETECTED, "Electrical contact detected")
                return
            self._require_camera_ready()
            delta = Vec3(0.0, 0.0, -1.0)
            self._execute_and_record_action(
                attempt,
                ControlAction(ActionType.ADVANCE, value=delta, reason="Baseline advance step"),
                lambda: self.adapters.manipulator.move_relative(delta),
            )
            self._snapshot(attempt)

    def _apply_seal_protocol(self, attempt: AttemptRecord) -> None:
        self._require_camera_ready()
        self._execute_and_record_action(
            attempt,
            ControlAction(ActionType.SET_PRESSURE, value=-20.0, reason="Baseline seal hold"),
            lambda: self.adapters.pressure.set_pressure(-20.0),
        )
        self._snapshot(attempt)
        self._require_camera_ready()
        self._execute_and_record_action(
            attempt,
            ControlAction(ActionType.PULSE_PRESSURE, value=-50.0, reason="Baseline suction pulse", metadata={"duration_ms": 250}),
            lambda: self.adapters.pressure.pulse_pressure(-50.0, duration_ms=250),
        )
        self._snapshot(attempt)

    def _release_pressure_to_neutral(self, attempt: AttemptRecord) -> None:
        action = ControlAction(
            ActionType.SET_PRESSURE,
            value=0.0,
            reason="Terminal pressure release to neutral",
        )
        try:
            self._execute_action(action, self.adapters.pressure.release_to_neutral)
        except Exception as error:
            reason = f"Pressure release to neutral failed: {error}"
            self.state = PatchState.FAILED
            attempt.outcome = Outcome.PRESSURE_RELEASE_FAILED
            self._record_action(attempt, ControlAction(ActionType.FAIL, reason=reason))
            self._log(
                attempt,
                "error",
                reason,
                PatchState.FAILED,
                {"outcome": Outcome.PRESSURE_RELEASE_FAILED.value},
            )
            raise PressureReleaseError(reason) from error
        self._record_action(attempt, action)
    def _fail(
        self,
        attempt: AttemptRecord,
        outcome: Outcome,
        reason: str,
        *,
        camera_observation: CameraObservation | None = None,
    ) -> None:
        self.state = PatchState.FAILED
        attempt.outcome = outcome
        self._record_action(attempt, ControlAction(ActionType.FAIL, reason=reason))
        self._log(attempt, "warning", reason, PatchState.FAILED, {"outcome": outcome.value})
        self._snapshot(attempt, camera_observation)

    def _transition(self, attempt: AttemptRecord, state: PatchState, message: str) -> None:
        self.state = state
        self._log(attempt, "info", message, state)
        self._snapshot(attempt)

    def _await_camera_ready(self, attempt: AttemptRecord) -> CameraObservation:
        deadline = time.monotonic() + max(0.0, self.camera_ready_timeout_seconds)
        last_observation: CameraObservation | None = None
        while True:
            last_observation = self.adapters.camera.observe()
            if last_observation.status.valid and not last_observation.quality.stale:
                return last_observation
            if time.monotonic() >= deadline:
                return last_observation
            time.sleep(max(0.0, self.camera_ready_poll_seconds))

    def _require_camera_ready(self) -> CameraObservation:
        observation = self.adapters.camera.observe()
        if not observation.status.valid or observation.quality.stale:
            raise CameraObservationNotReady(observation)
        return observation

    def _snapshot(self, attempt: AttemptRecord, camera_observation: CameraObservation | None = None) -> None:
        camera_observation = camera_observation or self.adapters.camera.observe()
        snapshot = RigSnapshot(
            timestamp=datetime.now(UTC),
            patch_state=self.state,
            manipulator_position=self.adapters.manipulator.get_position(),
            focus_position_um=self.adapters.focus.get_position_um(),
            pressure=self.adapters.pressure.get_pressure(),
            ephys=self.adapters.ephys.read(),
            vision=camera_observation.to_vision_reading(),
            camera_observation=camera_observation,
            objective_state=self.adapters.objective_selector.get_state() if self.adapters.objective_selector is not None else None,
        )
        attempt.snapshots.append(snapshot)
        self._notify_snapshot(attempt, snapshot)

    def _log(self, attempt: AttemptRecord, level: str, message: str, state: PatchState, payload: dict | None = None) -> None:
        event = SessionEvent(
            timestamp=datetime.now(UTC),
            level=level,
            message=message,
            state=state,
            payload=payload or {},
        )
        attempt.events.append(event)
        self._notify_event(attempt, event)

    def _record_action(self, attempt: AttemptRecord, action: ControlAction) -> None:
        if action.issued_at is None:
            action.issued_at = datetime.now(UTC)
        if action.issued_monotonic_ns is None:
            action.issued_monotonic_ns = time.monotonic_ns()
        if action.completed_monotonic_ns is None:
            action.completed_monotonic_ns = time.monotonic_ns()
        if action.completed_at is None:
            action.completed_at = datetime.now(UTC)
        attempt.actions.append(action)
        self._notify_action(attempt, action)

    def _execute_and_record_action(
        self,
        attempt: AttemptRecord,
        action: ControlAction,
        operation: Callable[[], object],
    ) -> None:
        self._execute_action(action, operation)
        self._record_action(attempt, action)

    @staticmethod
    def _execute_action(action: ControlAction, operation: Callable[[], object]) -> None:
        action.issued_at = datetime.now(UTC)
        action.issued_monotonic_ns = time.monotonic_ns()
        operation()
        action.completed_monotonic_ns = time.monotonic_ns()
        action.completed_at = datetime.now(UTC)

    def _notify_attempt_started(self, attempt: AttemptRecord) -> None:
        if self.observer is not None:
            self.observer.on_attempt_started(attempt)

    def _notify_event(self, attempt: AttemptRecord, event: SessionEvent) -> None:
        if self.observer is not None:
            self.observer.on_event(attempt, event)

    def _notify_action(self, attempt: AttemptRecord, action: ControlAction) -> None:
        if self.observer is not None:
            self.observer.on_action(attempt, action)

    def _notify_snapshot(self, attempt: AttemptRecord, snapshot: RigSnapshot) -> None:
        if self.observer is not None:
            self.observer.on_snapshot(attempt, snapshot)

    def _notify_attempt_finished(self, attempt: AttemptRecord) -> None:
        if self.observer is not None:
            self.observer.on_attempt_finished(attempt)
