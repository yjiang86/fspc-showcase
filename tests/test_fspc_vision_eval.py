from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autopatcher.vision_eval import (
    BoundingBoxLabel,
    CorrectionLabel,
    DatasetSplit,
    FSPCVisionTask,
    TipLocalizationLabel,
    VisionDatasetSample,
    build_eval_report,
    sample_to_record,
    validate_sample,
    write_eval_report,
)


class TestFSPCVisionEval(unittest.TestCase):
    def test_detection_sample_requires_boxes(self) -> None:
        sample = VisionDatasetSample(
            sample_id="sample-1",
            task_name=FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION,
            rig_id="rig-a",
            session_id="session-1",
            attempt_id="attempt-1",
            frame_id="frame-1",
            stack_id=None,
            timestamp_host_utc="2026-04-09T12:00:00Z",
            modality="dic",
            image_path="runtime/frame-1.tif",
            pixel_size_um=0.25,
            objective_magnification=40.0,
            species_or_prep="mouse_slice",
            slice_or_culture_context="cortex",
            operator_label_source="local_annotation",
            split=DatasetSplit.TRAIN,
            provenance="local_source",
        )
        with self.assertRaises(ValueError):
            validate_sample(sample)

    def test_localization_and_correction_sample_to_record(self) -> None:
        sample = VisionDatasetSample(
            sample_id="sample-2",
            task_name=FSPCVisionTask.PIPETTE_CORRECTION,
            rig_id="rig-a",
            session_id="session-2",
            attempt_id="attempt-2",
            frame_id="frame-2",
            stack_id=None,
            timestamp_host_utc="2026-04-09T12:01:00Z",
            modality="dic",
            image_path="runtime/frame-2.tif",
            pixel_size_um=0.3,
            objective_magnification=60.0,
            species_or_prep="mouse_slice",
            slice_or_culture_context="hippocampus",
            operator_label_source="local_annotation",
            split=DatasetSplit.TEST,
            provenance="mixed_source",
            tip_position=TipLocalizationLabel(x_px=120.0, y_px=90.0, confidence=0.92, z_um=5.0),
            correction_delta=CorrectionLabel(dx_um=1.0, dy_um=-0.5, dz_um=0.25, confidence=0.88, bounded_move_allowed=True),
        )
        record = sample_to_record(sample)
        self.assertEqual(record["task_name"], "pipette_correction")
        self.assertEqual(record["split"], "test")
        self.assertEqual(record["tip_position"]["x_px"], 120.0)
        self.assertTrue(record["correction_delta"]["bounded_move_allowed"])

    def test_detection_report_requires_full_metric_profile(self) -> None:
        with self.assertRaises(ValueError):
            build_eval_report(
                task_name=FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION,
                dataset_name="dic-public-local-mix",
                split_name="holdout_day",
                model_name="detector-v1",
                n_items=32,
                metrics={"mAP@0.50": 0.8},
            )

    def test_write_eval_report(self) -> None:
        report = build_eval_report(
            task_name=FSPCVisionTask.PIPETTE_TIP_LOCALIZATION,
            dataset_name="pipette-tip-v1",
            split_name="test",
            model_name="tip-net-v1",
            n_items=24,
            metrics={
                "mean_pixel_error": 2.5,
                "median_pixel_error": 2.0,
                "mean_micron_error": 0.8,
                "success_rate_px_threshold": 0.91,
                "success_rate_um_threshold": 0.87,
                "latency_ms_mean": 12.0,
                "motion_blur_failure_rate": 0.08,
                "angle_robustness_score": 0.84,
            },
            notes=["holdout_day included"],
            metadata={"calibration_version": "cal-v1"},
        )
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_eval_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmpdir / "vision_eval.json"
            write_eval_report(report, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_name"], "pipette_tip_localization")
            self.assertEqual(payload["dataset_name"], "pipette-tip-v1")
            self.assertEqual(payload["metrics"]["mean_pixel_error"], 2.5)
            self.assertEqual(payload["notes"], ["holdout_day included"])
        finally:
            if path.exists():
                path.unlink()
            tmpdir.rmdir()

    def test_detection_sample_to_record(self) -> None:
        sample = VisionDatasetSample(
            sample_id="sample-3",
            task_name=FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION,
            rig_id="rig-b",
            session_id="session-3",
            attempt_id="attempt-3",
            frame_id="frame-3",
            stack_id=None,
            timestamp_host_utc="2026-04-09T12:02:00Z",
            modality="dic",
            image_path="runtime/frame-3.tif",
            pixel_size_um=0.28,
            objective_magnification=40.0,
            species_or_prep="mouse_slice",
            slice_or_culture_context="thalamus",
            operator_label_source="literature_import",
            split=DatasetSplit.HOLDOUT_DAY,
            provenance="public_source",
            candidate_boxes=[
                BoundingBoxLabel(
                    x_px=10.0,
                    y_px=20.0,
                    width_px=30.0,
                    height_px=25.0,
                    confidence=0.95,
                    class_label="healthy_target_cell",
                )
            ],
        )
        record = sample_to_record(sample)
        self.assertEqual(record["candidate_boxes"][0]["class_label"], "healthy_target_cell")
        self.assertEqual(record["split"], "holdout_day")


if __name__ == "__main__":
    unittest.main()
