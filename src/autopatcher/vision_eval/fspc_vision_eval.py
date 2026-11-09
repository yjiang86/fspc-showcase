from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .fspc_vision_schema import FSPCVisionTask


@dataclass(slots=True)
class FSPCVisionMetricProfile:
    task_name: FSPCVisionTask
    primary_metrics: list[str]
    operational_metrics: list[str]
    acceptance_gates: list[str]


@dataclass(slots=True)
class FSPCVisionEvalReport:
    task_name: FSPCVisionTask
    dataset_name: str
    split_name: str
    model_name: str
    n_items: int
    metrics: dict[str, float]
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def default_metric_profile(task_name: FSPCVisionTask) -> FSPCVisionMetricProfile:
    if task_name == FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION:
        return FSPCVisionMetricProfile(
            task_name=task_name,
            primary_metrics=["mAP@0.50", "mAP@0.50:0.95", "f1_balanced", "recall_at_fixed_precision", "false_positive_rate_per_frame"],
            operational_metrics=["latency_ms_mean", "throughput_fps", "calibration_error"],
            acceptance_gates=["holdout_split_required", "no_random_only_split", "separate_nomination_vs_patch_success"],
        )
    if task_name == FSPCVisionTask.PIPETTE_TIP_LOCALIZATION:
        return FSPCVisionMetricProfile(
            task_name=task_name,
            primary_metrics=["mean_pixel_error", "median_pixel_error", "mean_micron_error", "success_rate_px_threshold", "success_rate_um_threshold"],
            operational_metrics=["latency_ms_mean", "motion_blur_failure_rate", "angle_robustness_score"],
            acceptance_gates=["pixel_and_micron_reporting_when_calibrated", "visible_tip_subset_split"],
        )
    return FSPCVisionMetricProfile(
        task_name=task_name,
        primary_metrics=["mean_abs_error_reduction", "median_error_reduction", "success_rate_final_threshold", "closed_loop_success_rate"],
        operational_metrics=["overshoot_rate", "collision_proxy_rate", "abort_trigger_rate", "latency_ms_mean"],
        acceptance_gates=["before_after_error_reporting", "offline_vs_closed_loop_split", "safety_metrics_required"],
    )


def validate_eval_report(report: FSPCVisionEvalReport) -> None:
    profile = default_metric_profile(report.task_name)
    required = [*profile.primary_metrics, *profile.operational_metrics]
    missing = [name for name in required if name not in report.metrics]
    if missing:
        raise ValueError(f"Vision eval report missing required metrics: {missing}")
    if report.n_items <= 0:
        raise ValueError("Vision eval report requires n_items > 0.")


def build_eval_report(
    *,
    task_name: FSPCVisionTask,
    dataset_name: str,
    split_name: str,
    model_name: str,
    n_items: int,
    metrics: dict[str, float],
    notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FSPCVisionEvalReport:
    report = FSPCVisionEvalReport(
        task_name=task_name,
        dataset_name=dataset_name,
        split_name=split_name,
        model_name=model_name,
        n_items=n_items,
        metrics=dict(metrics),
        notes=list(notes or []),
        metadata=dict(metadata or {}),
    )
    validate_eval_report(report)
    return report


def write_eval_report(report: FSPCVisionEvalReport, path: str | Path) -> Path:
    validate_eval_report(report)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["task_name"] = report.task_name.value
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
