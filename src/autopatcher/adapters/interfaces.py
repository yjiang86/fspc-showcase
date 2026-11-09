from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from autopatcher.models.domain import CameraFrameGeometry, CameraObservation, CameraSubsystemStatus, EjectionEvent, EjectionTriggerMode, ElectrophysiologyReading, ObjectiveCommandMode, ObjectiveState, ObjectiveSwitchEvent, PressureReading, Vec3, VisionReading


@dataclass(slots=True)
class CameraFrame:
    timestamp: datetime
    frame_id: str
    width: int
    height: int
    source: str


@dataclass(slots=True)
class ManipulatorLimits:
    minimum: Vec3
    maximum: Vec3


class ManipulatorAdapter(ABC):
    @abstractmethod
    def get_position(self) -> Vec3:
        raise NotImplementedError

    @abstractmethod
    def move_absolute(self, target: Vec3) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_relative(self, delta: Vec3) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_limits(self) -> ManipulatorLimits:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class FocusAdapter(ABC):
    @abstractmethod
    def get_position_um(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def move_relative_um(self, delta_um: float) -> None:
        raise NotImplementedError


class PressureAdapter(ABC):
    @abstractmethod
    def get_pressure(self, channel: int = 1) -> PressureReading:
        raise NotImplementedError

    @abstractmethod
    def set_pressure(self, mbar: float, channel: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def pulse_pressure(self, mbar: float, duration_ms: int, channel: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_to_neutral(self, channel: int = 1) -> None:
        raise NotImplementedError


class EjectionAdapter(ABC):
    @abstractmethod
    def eject(
        self,
        duration_ms: int,
        *,
        trigger_context: str = "",
        trigger_mode: EjectionTriggerMode = EjectionTriggerMode.SOFTWARE,
    ) -> EjectionEvent:
        raise NotImplementedError

    @abstractmethod
    def non_patch_flush(
        self,
        duration_ms: int,
        *,
        trigger_context: str = "",
        trigger_mode: EjectionTriggerMode = EjectionTriggerMode.SOFTWARE,
    ) -> EjectionEvent:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class EphysAdapter(ABC):
    @abstractmethod
    def read(self) -> ElectrophysiologyReading:
        raise NotImplementedError

    @abstractmethod
    def zap(self) -> None:
        raise NotImplementedError


class CameraAdapter(ABC):
    @abstractmethod
    def observe(self) -> CameraObservation:
        raise NotImplementedError

    def latest_frame(self) -> CameraFrame:
        observation = self.observe()
        geometry: CameraFrameGeometry = observation.frame_geometry
        return CameraFrame(
            timestamp=observation.timestamp_host_utc,
            frame_id=observation.frame_id or observation.observation_id,
            width=geometry.width,
            height=geometry.height,
            source=observation.source,
        )

    def analyze(self) -> VisionReading:
        return self.observe().to_vision_reading()

    def get_status(self) -> CameraSubsystemStatus:
        return self.observe().status


class CleaningAdapter(ABC):
    @abstractmethod
    def move_to_cleaning_position(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_cleaning_cycle(self) -> None:
        raise NotImplementedError


class ObjectiveSelectorAdapter(ABC):
    @abstractmethod
    def get_state(self) -> ObjectiveState:
        raise NotImplementedError

    @abstractmethod
    def list_available_objectives(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def switch_to_objective(self, target: str, command_mode: ObjectiveCommandMode = ObjectiveCommandMode.AUTOMATIC) -> ObjectiveSwitchEvent:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def get_current_objective(self) -> str | None:
        return self.get_state().current_objective
