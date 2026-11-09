from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autopatcher.adapters.mock import MockPressure
from autopatcher.bootstrap import build_demo_config
from autopatcher.core.baseline import BaselineAutopatchController, PressureReleaseError
from autopatcher.models.domain import ActionType, CameraAcquisitionState, CameraFrameGeometry, CameraImageReference, CameraObservation, CameraQualityState, CameraSubsystemStatus, Outcome, PatchState
from datetime import UTC, datetime
import time


class _RecordingObserver:
    def __init__(self) -> None:
        self.started = 0
        self.finished = 0
        self.events = []
        self.actions = []
        self.snapshots = []

    def on_attempt_started(self, attempt) -> None:
        self.started += 1

    def on_event(self, attempt, event) -> None:
        self.events.append(event)

    def on_action(self, attempt, action) -> None:
        self.actions.append(action)

    def on_snapshot(self, attempt, snapshot) -> None:
        self.snapshots.append(snapshot)

    def on_attempt_finished(self, attempt) -> None:
        self.finished += 1


class _DelayedReadyCamera:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self) -> CameraObservation:
        self.calls += 1
        ready = self.calls >= 3
        return CameraObservation(
            observation_id=f"delayed-{self.calls}",
            frame_id="frame-1" if ready else None,
            source="test",
            camera_model="test-camera",
            timestamp_host_utc=datetime.now(UTC),
            timestamp_monotonic_ns=time.time_ns(),
            frame_geometry=CameraFrameGeometry(width=1200 if ready else 0, height=1200 if ready else 0),
            acquisition=CameraAcquisitionState(streaming_state="streaming" if ready else "starting"),
            quality=CameraQualityState(stale=not ready),
            image_ref=CameraImageReference(shared_memory_name="test"),
            status=CameraSubsystemStatus(valid=ready, degraded_reason=None if ready else "warming_up"),
        )


class _NeverReadyCamera(_DelayedReadyCamera):
    def observe(self) -> CameraObservation:
        observation = super().observe()
        observation.status.valid = False
        observation.status.degraded_reason = "persistent_camera_fault"
        observation.quality.stale = True
        return observation


class _ReadyThenFaultCamera(_DelayedReadyCamera):
    def observe(self) -> CameraObservation:
        observation = super().observe()
        if self.calls == 1:
            observation.status.valid = True
            observation.status.degraded_reason = None
            observation.quality.stale = False
        else:
            observation.status.valid = False
            observation.status.degraded_reason = "camera_lost_after_start"
            observation.quality.stale = True
        return observation


class _FailingReleasePressure(MockPressure):
    def release_to_neutral(self, channel: int = 1) -> None:
        raise RuntimeError("release actuator unavailable")


class BaselineObserverTests(unittest.TestCase):
    def test_observer_receives_attempt_events_actions_and_snapshots(self) -> None:
        config = build_demo_config()
        observer = _RecordingObserver()
        controller = BaselineAutopatchController(config.rig_adapters, observer=observer)

        attempt = controller.start_attempt(target_id="observer-demo")
        controller.run_baseline_sequence(attempt)

        self.assertEqual(observer.started, 1)
        self.assertEqual(observer.finished, 1)
        self.assertGreater(len(observer.events), 0)
        self.assertGreater(len(observer.actions), 0)
        self.assertGreater(len(observer.snapshots), 0)
        self.assertEqual(observer.events[0].message, "Attempt initialized")
        self.assertEqual(config.rig_adapters.pressure.get_pressure().setpoint_mbar, 0.0)
        self.assertEqual(observer.actions[-1].action_type, ActionType.SET_PRESSURE)
        self.assertEqual(observer.actions[-1].value, 0.0)

    def test_start_attempt_waits_briefly_for_camera_ready(self) -> None:
        config = build_demo_config()
        config.rig_adapters.camera = _DelayedReadyCamera()
        controller = BaselineAutopatchController(
            config.rig_adapters,
            camera_ready_timeout_seconds=0.05,
            camera_ready_poll_seconds=0.0,
        )

        attempt = controller.start_attempt(target_id="camera-ready-demo")

        self.assertGreaterEqual(config.rig_adapters.camera.calls, 3)
        self.assertTrue(attempt.snapshots[0].camera_observation.status.valid)
        self.assertFalse(attempt.snapshots[0].camera_observation.quality.stale)

    def test_persistent_camera_failure_fails_closed_before_actuator_work(self) -> None:
        config = build_demo_config()
        config.rig_adapters.camera = _NeverReadyCamera()
        initial_position = config.rig_adapters.manipulator.get_position()
        controller = BaselineAutopatchController(
            config.rig_adapters,
            camera_ready_timeout_seconds=0.0,
            camera_ready_poll_seconds=0.0,
        )

        attempt = controller.start_attempt(target_id="camera-fault-demo")
        controller.run_baseline_sequence(attempt)

        final_position = config.rig_adapters.manipulator.get_position()
        self.assertEqual(attempt.outcome, Outcome.CAMERA_NOT_READY)
        self.assertEqual(final_position, initial_position)
        self.assertEqual(
            [action.action_type for action in attempt.actions],
            [ActionType.FAIL, ActionType.SET_PRESSURE],
        )
        self.assertEqual(config.rig_adapters.pressure.get_pressure().setpoint_mbar, 0.0)
        self.assertTrue(attempt.snapshots[-1].camera_observation.quality.stale)

    def test_camera_loss_after_start_fails_before_first_actuator_action(self) -> None:
        config = build_demo_config()
        config.rig_adapters.camera = _ReadyThenFaultCamera()
        initial_position = config.rig_adapters.manipulator.get_position()
        controller = BaselineAutopatchController(config.rig_adapters)

        attempt = controller.start_attempt(target_id="camera-loss-demo")
        controller.run_baseline_sequence(attempt)

        self.assertEqual(attempt.outcome, Outcome.CAMERA_NOT_READY)
        self.assertEqual(config.rig_adapters.manipulator.get_position(), initial_position)
        self.assertEqual(
            [action.action_type for action in attempt.actions],
            [ActionType.FAIL, ActionType.SET_PRESSURE],
        )
        self.assertEqual(config.rig_adapters.pressure.get_pressure().setpoint_mbar, 0.0)

    def test_abort_releases_pressure_to_neutral(self) -> None:
        config = build_demo_config()
        controller = BaselineAutopatchController(config.rig_adapters)
        attempt = controller.start_attempt(target_id="abort-demo")
        config.rig_adapters.pressure.set_pressure(-20.0)

        controller.abort_and_clean(attempt, "operator abort")

        self.assertEqual(attempt.outcome, Outcome.USER_ABORT)
        self.assertEqual(config.rig_adapters.pressure.get_pressure().setpoint_mbar, 0.0)
        self.assertEqual(attempt.actions[-1].action_type, ActionType.SET_PRESSURE)
        self.assertEqual(attempt.actions[-1].value, 0.0)

    def test_release_failure_overrides_false_success_and_notifies_observer(self) -> None:
        config = build_demo_config()
        config.rig_adapters.pressure = _FailingReleasePressure()
        observer = _RecordingObserver()
        controller = BaselineAutopatchController(config.rig_adapters, observer=observer)
        attempt = controller.start_attempt(target_id="release-failure-demo")

        with self.assertRaisesRegex(PressureReleaseError, "release actuator unavailable"):
            controller.run_baseline_sequence(attempt)

        self.assertEqual(controller.state, PatchState.FAILED)
        self.assertEqual(attempt.outcome, Outcome.PRESSURE_RELEASE_FAILED)
        self.assertEqual(attempt.actions[-1].action_type, ActionType.FAIL)
        self.assertEqual(observer.finished, 1)


if __name__ == "__main__":
    unittest.main()
