from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import uuid
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autopatcher.vision_eval import (
    BoundingBoxLabel,
    DatasetSplit,
    FSPCVisionTask,
    TipLocalizationLabel,
    VisionDatasetSample,
    build_dataset_manifest,
    export_split_bundle,
    export_dataset_jsonl,
    export_training_jsonl,
    load_samples_jsonl,
    record_to_sample,
    sample_to_training_record,
    validate_split_integrity,
    write_export_bundle_manifest,
    write_dataset_manifest,
    write_export_report,
)


class TestFSPCVisionExport(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            VisionDatasetSample(
                sample_id="det-1",
                task_name=FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION,
                rig_id="rig-a",
                session_id="s1",
                attempt_id="a1",
                frame_id="f1",
                stack_id=None,
                timestamp_host_utc="2026-04-09T12:00:00Z",
                modality="dic",
                image_path="runtime/f1.tif",
                pixel_size_um=0.25,
                objective_magnification=40.0,
                species_or_prep="mouse_slice",
                slice_or_culture_context="cortex",
                operator_label_source="local_annotation",
                split=DatasetSplit.TRAIN,
                provenance="local_source",
                candidate_boxes=[
                    BoundingBoxLabel(
                        x_px=10.0,
                        y_px=20.0,
                        width_px=30.0,
                        height_px=40.0,
                        confidence=0.9,
                        class_label="healthy_target_cell",
                    )
                ],
            ),
            VisionDatasetSample(
                sample_id="tip-1",
                task_name=FSPCVisionTask.PIPETTE_TIP_LOCALIZATION,
                rig_id="rig-a",
                session_id="s1",
                attempt_id="a2",
                frame_id="f2",
                stack_id=None,
                timestamp_host_utc="2026-04-09T12:01:00Z",
                modality="dic",
                image_path="runtime/f2.tif",
                pixel_size_um=0.25,
                objective_magnification=60.0,
                species_or_prep="mouse_slice",
                slice_or_culture_context="cortex",
                operator_label_source="local_annotation",
                split=DatasetSplit.TEST,
                provenance="mixed_source",
                tip_position=TipLocalizationLabel(x_px=55.0, y_px=65.0, confidence=0.94, z_um=4.0),
            ),
        ]

    def test_sample_to_training_record_detection_shape(self) -> None:
        record = sample_to_training_record(self.samples[0])
        self.assertEqual(record["task_name"], "dic_healthy_cell_detection")
        self.assertIn("candidate_boxes", record["labels"])

    def test_export_dataset_jsonl_filters_by_split(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_export_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmpdir / "dataset.jsonl"
            report = export_dataset_jsonl(self.samples, path, split=DatasetSplit.TRAIN)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(report.exported_count, 1)
            self.assertEqual(report.split_counts, {"train": 1})
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["sample_id"], "det-1")
        finally:
            if path.exists():
                path.unlink()
            tmpdir.rmdir()

    def test_export_training_jsonl_filters_by_task(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_train_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmpdir / "train.jsonl"
            report = export_training_jsonl(
                self.samples,
                path,
                task_name=FSPCVisionTask.PIPETTE_TIP_LOCALIZATION,
            )
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(report.exported_count, 1)
            self.assertEqual(report.task_counts, {"pipette_tip_localization": 1})
            payload = json.loads(lines[0])
            self.assertIn("tip_position", payload["labels"])
        finally:
            if path.exists():
                path.unlink()
            tmpdir.rmdir()

    def test_build_and_write_manifest(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_manifest_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            dataset_path = tmpdir / "train.jsonl"
            report_path = tmpdir / "train.report.json"
            manifest_path = tmpdir / "train.manifest.json"
            report = export_training_jsonl(self.samples, dataset_path)
            write_export_report(report, report_path)
            manifest = build_dataset_manifest(
                dataset_name="fspc-vision-v1",
                export_format="vision_training_jsonl",
                data_path=dataset_path,
                report_path=report_path,
                report=report,
            )
            write_dataset_manifest(manifest, manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_name"], "fspc-vision-v1")
            self.assertEqual(payload["row_count"], 2)
            self.assertEqual(payload["task_counts"]["dic_healthy_cell_detection"], 1)
            self.assertEqual(payload["task_counts"]["pipette_tip_localization"], 1)
        finally:
            for path in [dataset_path, report_path, manifest_path]:
                if path.exists():
                    path.unlink()
            tmpdir.rmdir()

    def test_record_roundtrip_and_load_samples_jsonl(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_roundtrip_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmpdir / "samples.jsonl"
            export_dataset_jsonl(self.samples, path)
            loaded = load_samples_jsonl(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].task_name, FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION)
            self.assertEqual(loaded[1].task_name, FSPCVisionTask.PIPETTE_TIP_LOCALIZATION)
            record = {
                "sample_id": "det-9",
                "task_name": "dic_healthy_cell_detection",
                "rig_id": "rig-z",
                "session_id": "sz",
                "attempt_id": "az",
                "frame_id": "fz",
                "stack_id": None,
                "timestamp_host_utc": "2026-04-09T12:05:00Z",
                "modality": "dic",
                "image_path": "runtime/fz.tif",
                "pixel_size_um": 0.22,
                "objective_magnification": 40.0,
                "species_or_prep": "mouse_slice",
                "slice_or_culture_context": "cortex",
                "operator_label_source": "local_annotation",
                "split": "val",
                "provenance": "local_source",
                "camera_geometry": {},
                "roi_geometry": {},
                "task_metadata": {},
                "candidate_boxes": [
                    {
                        "x_px": 1.0,
                        "y_px": 2.0,
                        "width_px": 3.0,
                        "height_px": 4.0,
                        "confidence": 0.8,
                        "class_label": "healthy_target_cell",
                    }
                ],
            }
            sample = record_to_sample(record)
            self.assertEqual(sample.split, DatasetSplit.VAL)
            self.assertEqual(sample.candidate_boxes[0].class_label, "healthy_target_cell")
        finally:
            if path.exists():
                path.unlink()
            tmpdir.rmdir()

    def test_export_split_bundle_writes_per_split_manifest(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="vision_bundle_"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            manifest = export_split_bundle(
                self.samples,
                tmpdir,
                dataset_name="fspc_bundle_v1",
                export_kind="training",
            )
            manifest_path = tmpdir / "bundle.manifest.json"
            write_export_bundle_manifest(manifest, manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_name"], "fspc_bundle_v1")
            self.assertEqual(payload["total_row_count"], 2)
            self.assertEqual(len(payload["split_artifacts"]), 2)
            self.assertEqual(payload["attempt_counts"]["train"], 1)
            self.assertEqual(payload["attempt_counts"]["test"], 1)
        finally:
            for path in tmpdir.glob("*"):
                path.unlink()
            tmpdir.rmdir()

    def test_validate_split_integrity_rejects_cross_split_attempt(self) -> None:
        leaking = [
            self.samples[0],
            VisionDatasetSample(
                sample_id="det-2",
                task_name=FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION,
                rig_id="rig-a",
                session_id="s1",
                attempt_id="a1",
                frame_id="f3",
                stack_id=None,
                timestamp_host_utc="2026-04-09T12:02:00Z",
                modality="dic",
                image_path="runtime/f3.tif",
                pixel_size_um=0.25,
                objective_magnification=40.0,
                species_or_prep="mouse_slice",
                slice_or_culture_context="cortex",
                operator_label_source="local_annotation",
                split=DatasetSplit.TEST,
                provenance="local_source",
                candidate_boxes=[
                    BoundingBoxLabel(
                        x_px=1.0,
                        y_px=2.0,
                        width_px=3.0,
                        height_px=4.0,
                        confidence=0.7,
                        class_label="healthy_target_cell",
                    )
                ],
            ),
        ]
        with self.assertRaises(ValueError):
            validate_split_integrity(leaking)
