from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FSPCVisionTask(str, Enum):
    DIC_HEALTHY_CELL_DETECTION = "dic_healthy_cell_detection"
    PIPETTE_TIP_LOCALIZATION = "pipette_tip_localization"
    PIPETTE_CORRECTION = "pipette_correction"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT_DAY = "holdout_day"


@dataclass(slots=True)
class BoundingBoxLabel:
    x_px: float
    y_px: float
    width_px: float
    height_px: float
    confidence: float
    class_label: str


@dataclass(slots=True)
class TipLocalizationLabel:
    x_px: float
    y_px: float
    confidence: float
    z_index: int | None = None
    z_um: float | None = None


@dataclass(slots=True)
class CorrectionLabel:
    dx_um: float
    dy_um: float
    dz_um: float
    confidence: float
    bounded_move_allowed: bool
    overshoot_risk: float | None = None
    collision_risk: float | None = None


@dataclass(slots=True)
class VisionDatasetSample:
    sample_id: str
    task_name: FSPCVisionTask
    rig_id: str
    session_id: str
    attempt_id: str
    frame_id: str | None
    stack_id: str | None
    timestamp_host_utc: str
    modality: str
    image_path: str | None
    pixel_size_um: float | None
    objective_magnification: float | None
    species_or_prep: str
    slice_or_culture_context: str
    operator_label_source: str
    split: DatasetSplit
    provenance: str
    camera_geometry: dict[str, Any] = field(default_factory=dict)
    roi_geometry: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_boxes: list[BoundingBoxLabel] = field(default_factory=list)
    tip_position: TipLocalizationLabel | None = None
    correction_delta: CorrectionLabel | None = None


def validate_sample(sample: VisionDatasetSample) -> None:
    if not sample.frame_id and not sample.stack_id:
        raise ValueError("Vision sample requires frame_id or stack_id.")
    if not sample.image_path:
        raise ValueError("Vision sample requires image_path.")
    if sample.modality not in {"dic", "brightfield", "phase_contrast"}:
        raise ValueError(f"Unsupported modality: {sample.modality}")
    if sample.task_name == FSPCVisionTask.DIC_HEALTHY_CELL_DETECTION:
        if not sample.candidate_boxes:
            raise ValueError("DIC healthy-cell detection sample requires candidate_boxes.")
    elif sample.task_name == FSPCVisionTask.PIPETTE_TIP_LOCALIZATION:
        if sample.tip_position is None:
            raise ValueError("Pipette-tip localization sample requires tip_position.")
    elif sample.task_name == FSPCVisionTask.PIPETTE_CORRECTION:
        if sample.tip_position is None or sample.correction_delta is None:
            raise ValueError("Pipette correction sample requires tip_position and correction_delta.")


def sample_to_record(sample: VisionDatasetSample) -> dict[str, Any]:
    validate_sample(sample)
    payload = asdict(sample)
    payload["task_name"] = sample.task_name.value
    payload["split"] = sample.split.value
    return payload
