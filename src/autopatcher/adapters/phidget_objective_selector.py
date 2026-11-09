from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from autopatcher.adapters.interfaces import ObjectiveSelectorAdapter
from autopatcher.models.domain import (
    ObjectiveCommandMode,
    ObjectiveInterlockState,
    ObjectiveState,
    ObjectiveSwitchEvent,
    ObjectiveSwitchOutcome,
    ObjectiveVerificationSource,
)


@dataclass(slots=True)
class PhidgetObjectiveSelectorConfig:
    available_objectives: list[str]
    initial_objective: str
    settle_wait_ms: int = 250
    allow_hardware: bool = False
    device_serial_number: int | None = None
    hub_port: int | None = None
    channel: int | None = None
    verification_source: ObjectiveVerificationSource = ObjectiveVerificationSource.SERVO_CONTROLLER
    interlock_state: ObjectiveInterlockState = field(default_factory=ObjectiveInterlockState)
    operator_override_available: bool = True


class PhidgetObjectiveSelectorAdapter(ObjectiveSelectorAdapter):
    """First-pass stateful skeleton for the Phidget-backed objective path.

    This adapter intentionally stops short of real servo I/O unless a later
    hardware-specific transport is added. The current purpose is to freeze the
    controller-facing contract and config shape without pretending hardware
    validation already exists.
    """

    def __init__(self, config: PhidgetObjectiveSelectorConfig) -> None:
        if config.initial_objective not in config.available_objectives:
            raise ValueError("initial_objective must exist in available_objectives")
        self.config = config
        self._state = ObjectiveState(
            current_objective=config.initial_objective,
            available_objectives=list(config.available_objectives),
            verified=False,
            verification_source=ObjectiveVerificationSource.UNKNOWN,
            switching_allowed=config.interlock_state.automatic_switch_allowed(),
            interlock_state=config.interlock_state,
            settle_wait_ms=config.settle_wait_ms,
            status_detail="phidget_objective_selector_skeleton",
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
        return list(self._state.available_objectives)

    def switch_to_objective(self, target: str, command_mode: ObjectiveCommandMode = ObjectiveCommandMode.AUTOMATIC) -> ObjectiveSwitchEvent:
        started_at = datetime.now(UTC)
        interlocks = self._state.interlock_state
        self._state.last_requested_objective = target
        if target not in self._state.available_objectives:
            self._state.last_rejected_reason = "unknown_objective"
            return ObjectiveSwitchEvent(
                requested_objective=target,
                previous_objective=self._state.current_objective,
                effective_objective=self._state.current_objective,
                command_mode=command_mode,
                verification_source=self._state.verification_source,
                outcome=ObjectiveSwitchOutcome.REJECTED,
                started_at=started_at,
                settled_at=started_at,
                settle_wait_ms=self._state.settle_wait_ms,
                interlock_state=interlocks,
                rejected_reason="unknown_objective",
                operator_override_available=False,
                fault_code="unknown_objective",
                operator_note="Target objective not present in configured objective list.",
            )

        self._state.switching_allowed = interlocks.automatic_switch_allowed()
        if command_mode is ObjectiveCommandMode.MANUAL_OVERRIDE:
            override_confirmed = (
                self.config.operator_override_available
                and interlocks.operator_override_present
            )
            if not override_confirmed:
                self._state.last_rejected_reason = "operator_override_not_confirmed"
                return ObjectiveSwitchEvent(
                    requested_objective=target,
                    previous_objective=self._state.current_objective,
                    effective_objective=self._state.current_objective,
                    command_mode=command_mode,
                    verification_source=self._state.verification_source,
                    outcome=ObjectiveSwitchOutcome.REJECTED,
                    started_at=started_at,
                    settled_at=started_at,
                    settle_wait_ms=self._state.settle_wait_ms,
                    interlock_state=interlocks,
                    rejected_reason="operator_override_not_confirmed",
                    operator_override_available=self.config.operator_override_available,
                    fault_code="operator_override_not_confirmed",
                    operator_note="Manual override requires both an enabled path and explicit operator confirmation.",
                )
        elif not self._state.switching_allowed:
            failed_fields = interlocks.failed_fields()
            self._state.last_rejected_reason = "interlock_blocked"
            return ObjectiveSwitchEvent(
                requested_objective=target,
                previous_objective=self._state.current_objective,
                effective_objective=self._state.current_objective,
                command_mode=command_mode,
                verification_source=self._state.verification_source,
                outcome=ObjectiveSwitchOutcome.REJECTED,
                started_at=started_at,
                settled_at=started_at,
                settle_wait_ms=self._state.settle_wait_ms,
                interlock_state=interlocks,
                rejected_reason="interlock_blocked",
                operator_override_available=self.config.operator_override_available,
                fault_code="interlock_blocked",
                operator_note="Blocked by objective-switch interlocks: " + ", ".join(failed_fields),
            )

        previous = self._state.current_objective
        self._state.last_rejected_reason = "transport_unavailable"
        self._state.status_detail = "phidget_transport_unavailable"
        return ObjectiveSwitchEvent(
            requested_objective=target,
            previous_objective=previous,
            effective_objective=previous,
            command_mode=command_mode,
            verification_source=self._state.verification_source,
            outcome=ObjectiveSwitchOutcome.REJECTED,
            started_at=started_at,
            settled_at=started_at,
            settle_wait_ms=self._state.settle_wait_ms,
            interlock_state=interlocks,
            rejected_reason="transport_unavailable",
            operator_override_available=self.config.operator_override_available,
            fault_code="transport_unavailable",
            operator_note="No Phidget servo transport is wired; no objective state was changed.",
        )

    def stop(self) -> None:
        self._state.status_detail = "phidget_stop_requested"
