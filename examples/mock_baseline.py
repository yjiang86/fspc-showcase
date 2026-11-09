"""Exercise the existing controller with synthetic adapters only."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopatcher.adapters.mock import (
    MockCamera, MockCleaning, MockEjection, MockEphys, MockFocus,
    MockManipulator, MockObjectiveSelector, MockPressure,
)
from autopatcher.core.baseline import BaselineAutopatchController, RigAdapters


def run():
    pressure = MockPressure()
    adapters = RigAdapters(
        manipulator=MockManipulator(), focus=MockFocus(), pressure=pressure,
        ejection=MockEjection(), ephys=MockEphys(), camera=MockCamera(),
        cleaning=MockCleaning(), objective_selector=MockObjectiveSelector(),
    )
    controller = BaselineAutopatchController(adapters)
    attempt = controller.start_attempt(target_id="synthetic-example")
    result = controller.run_baseline_sequence(attempt)
    return {
        "mode": "MOCK_ONLY", "input": "SYNTHETIC_NO_SUBJECT_DATA",
        "controller_outcome_label": result.outcome.value,
        "controller_state": controller.state.value,
        "pressure_mbar_after_terminal": pressure.setpoint,
        "action_count": len(result.actions),
        "snapshot_count": len(result.snapshots),
        "physical_patch_success": "NOT_TESTED",
        "clock_calibration": "NOT_TESTED",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
