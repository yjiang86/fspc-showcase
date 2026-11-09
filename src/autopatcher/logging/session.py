from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import json

from autopatcher.models.domain import AttemptRecord, ControlAction, RigSnapshot, SessionEvent


class SessionLogger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_attempt(self, attempt: AttemptRecord) -> Path:
        payload = {
            "attempt_id": attempt.attempt_id,
            "started_at": attempt.started_at.isoformat(),
            "target_id": attempt.target_id,
            "outcome": attempt.outcome.value,
            "events": [self._event_to_dict(event) for event in attempt.events],
            "actions": [self._action_to_dict(action) for action in attempt.actions],
            "snapshots": [self._snapshot_to_dict(snapshot) for snapshot in attempt.snapshots],
        }
        output = self.root / f"{attempt.attempt_id}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    @staticmethod
    def _event_to_dict(event: SessionEvent) -> dict:
        return SessionLogger._jsonify(event)

    @staticmethod
    def _action_to_dict(action: ControlAction) -> dict:
        return SessionLogger._jsonify(action)

    @staticmethod
    def _snapshot_to_dict(snapshot: RigSnapshot) -> dict:
        return SessionLogger._jsonify(snapshot)

    @staticmethod
    def _jsonify(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {key: SessionLogger._jsonify(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {key: SessionLogger._jsonify(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SessionLogger._jsonify(item) for item in value]
        return value
