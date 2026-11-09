from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

from .fspc_vision_schema import (
    BoundingBoxLabel,
    CorrectionLabel,
    DatasetSplit,
    FSPCVisionTask,
    TipLocalizationLabel,
    VisionDatasetSample,
    sample_to_record,
)


@dataclass(slots=True)
class VisionExportReport:
    input_count: int
    exported_count: int
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    output_path: str


@dataclass(slots=True)
class VisionDatasetManifest:
    dataset_name: str
    export_format: str
    data_path: str
    report_path: str
    row_count: int
    split_counts: dict[str, int] = field(default_factory=dict)
    task_counts: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class VisionSplitArtifact:
    split: str
    row_count: int
    attempt_count: int
    data_path: str
    report_path: str


@dataclass(slots=True)
class VisionExportBundleManifest:
    dataset_name: str
    export_format: str
    output_dir: str
    total_row_count: int
    split_artifacts: list[VisionSplitArtifact] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    task_counts: dict[str, int] = field(default_factory=dict)
    attempt_counts: dict[str, int] = field(default_factory=dict)
    filters: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def export_dataset_jsonl(
    samples: list[VisionDatasetSample],
    path: str | Path,
    *,
    split: DatasetSplit | None = None,
    task_name: FSPCVisionTask | None = None,
) -> VisionExportReport:
    filtered = _filter_samples(samples, split=split, task_name=task_name)
    records = [sample_to_record(sample) for sample in filtered]
    _write_jsonl(records, path)
    return VisionExportReport(
        input_count=len(samples),
        exported_count=len(records),
        split_counts=_count_splits(filtered),
        task_counts=_count_tasks(filtered),
        output_path=Path(path).as_posix(),
    )


def load_samples_jsonl(path: str | Path) -> list[VisionDatasetSample]:
    samples: list[VisionDatasetSample] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(record_to_sample(record))
    return samples


def export_training_jsonl(
    samples: list[VisionDatasetSample],
    path: str | Path,
    *,
    split: DatasetSplit | None = None,
    task_name: FSPCVisionTask | None = None,
) -> VisionExportReport:
    filtered = _filter_samples(samples, split=split, task_name=task_name)
    records = [sample_to_training_record(sample) for sample in filtered]
    _write_jsonl(records, path)
    return VisionExportReport(
        input_count=len(samples),
        exported_count=len(records),
        split_counts=_count_splits(filtered),
        task_counts=_count_tasks(filtered),
        output_path=Path(path).as_posix(),
    )


def sample_to_training_record(sample: VisionDatasetSample) -> dict[str, object]:
    base = {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name.value,
        "image_path": sample.image_path,
        "split": sample.split.value,
        "modality": sample.modality,
        "attempt_id": sample.attempt_id,
        "frame_id": sample.frame_id,
        "stack_id": sample.stack_id,
        "pixel_size_um": sample.pixel_size_um,
        "objective_magnification": sample.objective_magnification,
        "species_or_prep": sample.species_or_prep,
        "slice_or_culture_context": sample.slice_or_culture_context,
        "camera_geometry": dict(sample.camera_geometry),
        "roi_geometry": dict(sample.roi_geometry),
        "task_metadata": dict(sample.task_metadata),
    }
    if sample.task_name == FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION:
        base["labels"] = {"candidate_boxes": [asdict(box) for box in sample.candidate_boxes]}
        return base
    if sample.task_name == FSPCVisionTask.PIPETTE_TIP_LOCALIZATION:
        base["labels"] = {"tip_position": asdict(sample.tip_position) if sample.tip_position else None}
        return base
    base["labels"] = {
        "tip_position": asdict(sample.tip_position) if sample.tip_position else None,
        "correction_delta": asdict(sample.correction_delta) if sample.correction_delta else None,
    }
    return base


def build_dataset_manifest(
    *,
    dataset_name: str,
    export_format: str,
    data_path: str | Path,
    report_path: str | Path,
    report: VisionExportReport,
) -> VisionDatasetManifest:
    return VisionDatasetManifest(
        dataset_name=dataset_name,
        export_format=export_format,
        data_path=Path(data_path).as_posix(),
        report_path=Path(report_path).as_posix(),
        row_count=report.exported_count,
        split_counts=dict(report.split_counts),
        task_counts=dict(report.task_counts),
    )


def validate_split_integrity(
    samples: list[VisionDatasetSample],
    *,
    task_name: FSPCVisionTask | None = None,
) -> None:
    filtered = _filter_samples(samples, split=None, task_name=task_name)
    attempts_to_splits: dict[str, set[str]] = {}
    for sample in filtered:
        attempt_id = sample.attempt_id
        attempts_to_splits.setdefault(attempt_id, set()).add(sample.split.value)
    leaking = {
        attempt_id: sorted(split_values)
        for attempt_id, split_values in attempts_to_splits.items()
        if len(split_values) > 1
    }
    if leaking:
        details = ", ".join(
            f"{attempt_id}={'+'.join(split_values)}"
            for attempt_id, split_values in sorted(leaking.items())
        )
        raise ValueError(
            "Vision dataset split leakage detected across attempts: "
            f"{details}"
        )


def export_split_bundle(
    samples: list[VisionDatasetSample],
    output_dir: str | Path,
    *,
    dataset_name: str,
    export_kind: str,
    task_name: FSPCVisionTask | None = None,
    splits: list[DatasetSplit] | None = None,
) -> VisionExportBundleManifest:
    if export_kind not in {"dataset", "training"}:
        raise ValueError(f"Unsupported export kind: {export_kind}")

    validate_split_integrity(samples, task_name=task_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_splits = list(splits) if splits is not None else _ordered_splits(samples, task_name=task_name)
    split_artifacts: list[VisionSplitArtifact] = []
    export_format = "vision_dataset_jsonl" if export_kind == "dataset" else "vision_training_jsonl"

    for split in selected_splits:
        filtered = _filter_samples(samples, split=split, task_name=task_name)
        if not filtered:
            continue
        base_name = ".".join(
            part
            for part in [
                dataset_name,
                task_name.value if task_name is not None else None,
                split.value,
                export_kind,
            ]
            if part
        )
        data_path = output_dir / f"{base_name}.jsonl"
        report_path = output_dir / f"{base_name}.report.json"
        if export_kind == "dataset":
            report = export_dataset_jsonl(filtered, data_path)
        else:
            report = export_training_jsonl(filtered, data_path)
        write_export_report(report, report_path)
        split_artifacts.append(
            VisionSplitArtifact(
                split=split.value,
                row_count=report.exported_count,
                attempt_count=len({sample.attempt_id for sample in filtered}),
                data_path=data_path.as_posix(),
                report_path=report_path.as_posix(),
            )
        )

    filtered_all = _filter_samples(samples, split=None, task_name=task_name)
    return VisionExportBundleManifest(
        dataset_name=dataset_name,
        export_format=export_format,
        output_dir=output_dir.as_posix(),
        total_row_count=sum(artifact.row_count for artifact in split_artifacts),
        split_artifacts=split_artifacts,
        split_counts=_count_splits(filtered_all),
        task_counts=_count_tasks(filtered_all),
        attempt_counts=_count_attempts_by_split(filtered_all),
        filters={
            "task_name": task_name.value if task_name is not None else None,
            "export_kind": export_kind,
        },
    )


def record_to_sample(record: dict[str, object]) -> VisionDatasetSample:
    candidate_boxes = [
        BoundingBoxLabel(**box)
        for box in record.get("candidate_boxes", [])
    ]
    tip_position = (
        TipLocalizationLabel(**record["tip_position"])
        if record.get("tip_position") is not None
        else None
    )
    correction_delta = (
        CorrectionLabel(**record["correction_delta"])
        if record.get("correction_delta") is not None
        else None
    )
    return VisionDatasetSample(
        sample_id=str(record["sample_id"]),
        task_name=FSPCVisionTask(str(record["task_name"])),
        rig_id=str(record["rig_id"]),
        session_id=str(record["session_id"]),
        attempt_id=str(record["attempt_id"]),
        frame_id=record.get("frame_id"),
        stack_id=record.get("stack_id"),
        timestamp_host_utc=str(record["timestamp_host_utc"]),
        modality=str(record["modality"]),
        image_path=record.get("image_path"),
        pixel_size_um=record.get("pixel_size_um"),
        objective_magnification=record.get("objective_magnification"),
        species_or_prep=str(record["species_or_prep"]),
        slice_or_culture_context=str(record["slice_or_culture_context"]),
        operator_label_source=str(record["operator_label_source"]),
        split=DatasetSplit(str(record["split"])),
        provenance=str(record["provenance"]),
        camera_geometry=dict(record.get("camera_geometry", {})),
        roi_geometry=dict(record.get("roi_geometry", {})),
        task_metadata=dict(record.get("task_metadata", {})),
        candidate_boxes=candidate_boxes,
        tip_position=tip_position,
        correction_delta=correction_delta,
    )


def write_export_report(report: VisionExportReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_dataset_manifest(manifest: VisionDatasetManifest, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_export_bundle_manifest(
    manifest: VisionExportBundleManifest,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _filter_samples(
    samples: list[VisionDatasetSample],
    *,
    split: DatasetSplit | None,
    task_name: FSPCVisionTask | None,
) -> list[VisionDatasetSample]:
    filtered = list(samples)
    if split is not None:
        filtered = [sample for sample in filtered if sample.split == split]
    if task_name is not None:
        filtered = [sample for sample in filtered if sample.task_name == task_name]
    return filtered


def _count_splits(samples: list[VisionDatasetSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample.split.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_tasks(samples: list[VisionDatasetSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample.task_name.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_attempts_by_split(samples: list[VisionDatasetSample]) -> dict[str, int]:
    split_to_attempts: dict[str, set[str]] = {}
    for sample in samples:
        split_to_attempts.setdefault(sample.split.value, set()).add(sample.attempt_id)
    return {
        split_name: len(attempt_ids)
        for split_name, attempt_ids in split_to_attempts.items()
    }


def _ordered_splits(
    samples: list[VisionDatasetSample],
    *,
    task_name: FSPCVisionTask | None,
) -> list[DatasetSplit]:
    filtered = _filter_samples(samples, split=None, task_name=task_name)
    seen = {sample.split for sample in filtered}
    ordered = [split for split in DatasetSplit if split in seen]
    return ordered


def _write_jsonl(records: list[dict[str, object]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
