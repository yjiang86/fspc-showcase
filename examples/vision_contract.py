"""Print existing evaluation contracts without inventing benchmark scores."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopatcher.vision_eval import FSPCVisionTask, default_metric_profile


def run():
    result = []
    for task in FSPCVisionTask:
        profile = default_metric_profile(task)
        result.append({
            "task": task.value, "required_metrics": profile.primary_metrics,
            "operational_metrics": profile.operational_metrics,
            "contract_checks": profile.acceptance_gates,
        })
    return {"kind": "CONTRACT_INSPECTION_NOT_MODEL_EVALUATION", "tasks": result}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
