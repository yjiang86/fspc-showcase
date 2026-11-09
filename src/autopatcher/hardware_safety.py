from __future__ import annotations

import argparse
from dataclasses import dataclass


class LiveHardwareAuthorizationError(RuntimeError):
    """Raised before a live adapter is opened without explicit authorization."""


@dataclass(frozen=True, slots=True)
class LiveHardwareAuthorization:
    operation: str
    operator_confirmed: bool

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("live hardware authorization requires an operation")
        if not self.operator_confirmed:
            raise LiveHardwareAuthorizationError(
                "live hardware authorization requires an explicit operator confirmation"
            )


def add_live_hardware_authorization_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--authorize-live-hardware",
        action="store_true",
        help="Explicitly authorize this live-hardware operation; this flag records intent, not a review receipt.",
    )


def require_live_hardware_authorization(
    authorized: bool,
    *,
    operation: str,
) -> LiveHardwareAuthorization:
    if not authorized:
        raise LiveHardwareAuthorizationError(
            f"live hardware access refused for {operation!r}; "
            "pass --authorize-live-hardware only after reviewing the physical plan"
        )
    return LiveHardwareAuthorization(
        operation=operation,
        operator_confirmed=True,
    )


def assert_live_hardware_authorization(
    authorization: LiveHardwareAuthorization | None,
    *,
    operation: str,
) -> LiveHardwareAuthorization:
    """Reject adapter construction without an explicit operator-intent token."""
    if not isinstance(authorization, LiveHardwareAuthorization) or not authorization.operator_confirmed:
        raise LiveHardwareAuthorizationError(
            f"live hardware access refused for {operation!r}; "
            "construct the adapter only with an explicit operator-intent token"
        )
    return authorization
