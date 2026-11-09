from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from autopatcher.models.domain import AttemptRecord, Outcome


CONTRACT_VERSION = "fsp_attempt_consumer_v1"


class ConsumerRecordKind(str, Enum):
    REPLAY_RECORD = "replay_record"
    REVIEW_CANDIDATE = "review_candidate"


@dataclass(slots=True)
class ConsumerQuality:
    confidence: float = 0.0
    support: float = 0.0
    consistency: float = 0.0
    novelty: float = 0.0
    utility: float = 0.0
    risk: float = 0.0
    final_score: float = 0.0


@dataclass(slots=True)
class AttemptConsumerRecord:
    record_id: str
    kind: ConsumerRecordKind
    payload: dict[str, Any]
    source_attempt_ids: list[str]
    topic_tags: list[str] = field(default_factory=list)
    quality: ConsumerQuality = field(default_factory=ConsumerQuality)
    use_hint: str = "review"
    exportable: bool = False
    dedupe_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptReplaySchema:
    attempt_id: str
    target_id: str | None
    outcome: str
    objective: str
    text: str
    summary: str
    context: list[str] = field(default_factory=list)
    phase_sequence: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptReviewSchema:
    attempt_id: str
    target_id: str | None
    outcome: str
    review_status: str
    instruction: str
    input: str
    output: str
    rationale_summary: str
    phase_sequence: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptConsumerExportReport:
    input_count: int
    replay_schema_count: int
    review_schema_count: int
    replay_record_count: int
    review_candidate_count: int
    review_exportable_count: int
    output_dir: str
    output_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptConsumerManifest:
    dataset_name: str
    contract_version: str
    output_dir: str
    row_count: int
    replay_schema_path: str
    review_schema_path: str
    replay_record_path: str
    review_candidate_path: str
    review_exportable_count: int
    review_status_counts: dict[str, int] = field(default_factory=dict)
    outcome_counts: dict[str, int] = field(default_factory=dict)


def build_attempt_replay_schema(attempt: AttemptRecord) -> AttemptReplaySchema:
    phase_sequence = _phase_sequence(attempt)
    image_ids = _image_ids(attempt)
    outcome = attempt.outcome.value
    context = [
        f"attempt_id={attempt.attempt_id}",
        f"target_id={attempt.target_id or 'none'}",
        f"outcome={outcome}",
        f"events={len(attempt.events)}",
        f"actions={len(attempt.actions)}",
        f"snapshots={len(attempt.snapshots)}",
    ]
    summary = (
        f"Attempt {attempt.attempt_id} finished with outcome {outcome} after "
        f"{len(attempt.actions)} actions and {len(attempt.snapshots)} snapshots."
    )
    return AttemptReplaySchema(
        attempt_id=attempt.attempt_id,
        target_id=attempt.target_id,
        outcome=outcome,
        objective="Rehearse an outcome-linked physical attempt trace for bounded downstream review.",
        text="\n".join(
            [
                summary,
                f"Phases: {', '.join(phase_sequence) if phase_sequence else 'none'}",
                f"Actions: {_action_summary(attempt)}",
                f"Events: {_event_summary(attempt)}",
                f"Snapshots: {_snapshot_summary(attempt)}",
            ]
        ),
        summary=summary,
        context=context,
        phase_sequence=phase_sequence,
        image_ids=image_ids,
        metadata={
            "contract_version": CONTRACT_VERSION,
            "source_schema": "fsp_attempt_replay_v1",
            "attempt_id": attempt.attempt_id,
            "target_id": attempt.target_id,
            "outcome": outcome,
            "event_count": len(attempt.events),
            "action_count": len(attempt.actions),
            "snapshot_count": len(attempt.snapshots),
            "phase_sequence": phase_sequence,
            "image_ids": image_ids,
        },
    )


def build_attempt_review_schema(
    attempt: AttemptRecord,
    *,
    review_status: str = "pending",
) -> AttemptReviewSchema:
    if review_status not in {"pending", "accepted", "rejected"}:
        raise ValueError("review_status must be pending, accepted, or rejected")
    phase_sequence = _phase_sequence(attempt)
    outcome = attempt.outcome.value
    evidence = {
        "event_count": len(attempt.events),
        "action_count": len(attempt.actions),
        "snapshot_count": len(attempt.snapshots),
        "image_ids": _image_ids(attempt),
        "final_state": phase_sequence[-1] if phase_sequence else None,
    }
    return AttemptReviewSchema(
        attempt_id=attempt.attempt_id,
        target_id=attempt.target_id,
        outcome=outcome,
        review_status=review_status,
        instruction="Convert the reviewed physical trace into a bounded, outcome-linked lesson.",
        input="\n".join(
            [
                f"attempt_id={attempt.attempt_id}",
                f"target_id={attempt.target_id or 'none'}",
                f"phase_sequence={', '.join(phase_sequence) if phase_sequence else 'none'}",
                f"action_summary={_action_summary(attempt)}",
                f"snapshot_summary={_snapshot_summary(attempt)}",
            ]
        ),
        output="\n".join(
            [
                f"outcome={outcome}",
                f"lesson={_lesson_from_outcome(attempt.outcome)}",
                f"recommended_use={_consumer_route(outcome)}",
            ]
        ),
        rationale_summary=(
            "Review candidates remain non-exportable until accepted; the physical trace and "
            "outcome-linked lesson are preserved without selecting a downstream AI runtime."
        ),
        phase_sequence=phase_sequence,
        evidence=evidence,
        metadata={
            "contract_version": CONTRACT_VERSION,
            "source_schema": "fsp_attempt_review_v1",
            "attempt_id": attempt.attempt_id,
            "target_id": attempt.target_id,
            "outcome": outcome,
            "phase_sequence": phase_sequence,
            "review_status": review_status,
            "consumer_route": _consumer_route(outcome),
            "evidence": evidence,
        },
    )


def attempt_replay_schema_to_consumer_record(schema: AttemptReplaySchema) -> AttemptConsumerRecord:
    dedupe_key = _dedupe_key("replay", schema.attempt_id, schema.phase_sequence)
    return AttemptConsumerRecord(
        record_id=f"fsp-replay-{dedupe_key[:24]}",
        kind=ConsumerRecordKind.REPLAY_RECORD,
        payload={
            "text": schema.text,
            "summary": schema.summary,
            "context": list(schema.context),
            "objective": schema.objective,
        },
        source_attempt_ids=[schema.attempt_id],
        topic_tags=_topic_tags(schema.outcome, schema.phase_sequence),
        quality=_quality_for_attempt(schema.outcome, len(schema.context)),
        use_hint="replay_only",
        exportable=True,
        dedupe_key=dedupe_key,
        metadata=dict(schema.metadata),
    )


def attempt_review_schema_to_consumer_record(schema: AttemptReviewSchema) -> AttemptConsumerRecord:
    dedupe_key = _dedupe_key("review", schema.attempt_id, schema.phase_sequence)
    return AttemptConsumerRecord(
        record_id=f"fsp-review-{dedupe_key[:24]}",
        kind=ConsumerRecordKind.REVIEW_CANDIDATE,
        payload={
            "instruction": schema.instruction,
            "input": schema.input,
            "output": schema.output,
            "rationale_summary": schema.rationale_summary,
        },
        source_attempt_ids=[schema.attempt_id],
        topic_tags=_topic_tags(schema.outcome, schema.phase_sequence),
        quality=_quality_for_attempt(schema.outcome, len(schema.phase_sequence)),
        use_hint="downstream_training_candidate",
        exportable=schema.review_status == "accepted",
        dedupe_key=dedupe_key,
        metadata=dict(schema.metadata),
    )


def build_attempt_consumer_records(
    attempt: AttemptRecord,
    *,
    review_status: str = "pending",
) -> tuple[AttemptConsumerRecord, AttemptConsumerRecord]:
    replay_schema = build_attempt_replay_schema(attempt)
    review_schema = build_attempt_review_schema(attempt, review_status=review_status)
    return (
        attempt_replay_schema_to_consumer_record(replay_schema),
        attempt_review_schema_to_consumer_record(review_schema),
    )


def replay_schema_to_record(schema: AttemptReplaySchema) -> dict[str, Any]:
    return asdict(schema)


def review_schema_to_record(schema: AttemptReviewSchema) -> dict[str, Any]:
    return asdict(schema)


def consumer_record_to_record(record: AttemptConsumerRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["kind"] = record.kind.value
    return payload


def export_attempt_consumer_bundle(
    attempts: list[AttemptRecord],
    output_dir: str | Path,
    *,
    dataset_name: str,
    review_status: str = "pending",
    review_status_by_attempt: dict[str, str] | None = None,
) -> tuple[AttemptConsumerExportReport, AttemptConsumerManifest]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_schemas: list[AttemptReplaySchema] = []
    review_schemas: list[AttemptReviewSchema] = []
    replay_records: list[AttemptConsumerRecord] = []
    review_candidates: list[AttemptConsumerRecord] = []
    review_status_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}

    for attempt in attempts:
        effective_status = (
            review_status_by_attempt.get(attempt.attempt_id, review_status)
            if review_status_by_attempt is not None
            else review_status
        )
        replay_schema = build_attempt_replay_schema(attempt)
        review_schema = build_attempt_review_schema(attempt, review_status=effective_status)
        replay_schemas.append(replay_schema)
        review_schemas.append(review_schema)
        replay_records.append(attempt_replay_schema_to_consumer_record(replay_schema))
        review_candidates.append(attempt_review_schema_to_consumer_record(review_schema))
        review_status_counts[effective_status] = review_status_counts.get(effective_status, 0) + 1
        outcome_counts[attempt.outcome.value] = outcome_counts.get(attempt.outcome.value, 0) + 1

    replay_schema_path = output_dir / f"{dataset_name}.attempt_replay_v1.jsonl"
    review_schema_path = output_dir / f"{dataset_name}.attempt_review_v1.jsonl"
    replay_record_path = output_dir / f"{dataset_name}.consumer_replay_v1.jsonl"
    review_candidate_path = output_dir / f"{dataset_name}.consumer_review_candidates_v1.jsonl"

    _write_jsonl([replay_schema_to_record(item) for item in replay_schemas], replay_schema_path)
    _write_jsonl([review_schema_to_record(item) for item in review_schemas], review_schema_path)
    _write_jsonl([consumer_record_to_record(item) for item in replay_records], replay_record_path)
    _write_jsonl([consumer_record_to_record(item) for item in review_candidates], review_candidate_path)

    report = AttemptConsumerExportReport(
        input_count=len(attempts),
        replay_schema_count=len(replay_schemas),
        review_schema_count=len(review_schemas),
        replay_record_count=len(replay_records),
        review_candidate_count=len(review_candidates),
        review_exportable_count=sum(1 for item in review_candidates if item.exportable),
        output_dir=output_dir.as_posix(),
        output_paths={
            "replay_schema_path": replay_schema_path.as_posix(),
            "review_schema_path": review_schema_path.as_posix(),
            "replay_record_path": replay_record_path.as_posix(),
            "review_candidate_path": review_candidate_path.as_posix(),
        },
    )
    manifest = AttemptConsumerManifest(
        dataset_name=dataset_name,
        contract_version=CONTRACT_VERSION,
        output_dir=output_dir.as_posix(),
        row_count=len(attempts),
        replay_schema_path=replay_schema_path.as_posix(),
        review_schema_path=review_schema_path.as_posix(),
        replay_record_path=replay_record_path.as_posix(),
        review_candidate_path=review_candidate_path.as_posix(),
        review_exportable_count=report.review_exportable_count,
        review_status_counts=review_status_counts,
        outcome_counts=outcome_counts,
    )
    return report, manifest


def write_attempt_consumer_report(report: AttemptConsumerExportReport, path: str | Path) -> Path:
    return _write_json(asdict(report), path)


def write_attempt_consumer_manifest(manifest: AttemptConsumerManifest, path: str | Path) -> Path:
    return _write_json(asdict(manifest), path)


def _phase_sequence(attempt: AttemptRecord) -> list[str]:
    seen: list[str] = []
    for value in [
        *(event.state.value for event in attempt.events),
        *(snapshot.patch_state.value for snapshot in attempt.snapshots),
    ]:
        if value not in seen:
            seen.append(value)
    return seen


def _image_ids(attempt: AttemptRecord) -> list[str]:
    image_ids: list[str] = []
    for snapshot in attempt.snapshots:
        image_id = snapshot.vision.image_id
        if image_id and image_id not in image_ids:
            image_ids.append(image_id)
    return image_ids


def _action_summary(attempt: AttemptRecord) -> str:
    if not attempt.actions:
        return "none"
    counts: dict[str, int] = {}
    for action in attempt.actions:
        key = action.action_type.value
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _event_summary(attempt: AttemptRecord) -> str:
    return " | ".join(event.message for event in attempt.events[:3]) if attempt.events else "none"


def _snapshot_summary(attempt: AttemptRecord) -> str:
    if not attempt.snapshots:
        return "none"
    last = attempt.snapshots[-1]
    parts = [
        f"final_state={last.patch_state.value}",
        f"pressure_setpoint={last.pressure.setpoint_mbar}",
    ]
    if last.ephys.seal_resistance_mohm is not None:
        parts.append(f"seal_resistance_mohm={last.ephys.seal_resistance_mohm}")
    if last.vision.cell_probability is not None:
        parts.append(f"cell_probability={last.vision.cell_probability}")
    if last.vision.pipette_tip_probability is not None:
        parts.append(f"pipette_tip_probability={last.vision.pipette_tip_probability}")
    return ", ".join(parts)


def _lesson_from_outcome(outcome: Outcome) -> str:
    if outcome is Outcome.WHOLE_CELL_SUCCESS:
        return "Preserve the successful action sequence and final confirmation pattern."
    if outcome is Outcome.CONTACT_NO_SEAL:
        return "Review approach and pressure timing before the seal phase."
    if outcome is Outcome.SEAL_FAILED:
        return "Review seal-formation telemetry and pressure protocol."
    if outcome is Outcome.USER_ABORT:
        return "Retain manual intervention context and abort reasons for review."
    if outcome is Outcome.UNKNOWN:
        return "Keep as review-first until outcome and labels are clarified."
    return "Preserve the failure pattern as a bounded negative lesson for later review."


def _consumer_route(outcome: str) -> str:
    if outcome == Outcome.WHOLE_CELL_SUCCESS.value:
        return "successful_trace"
    if outcome in {Outcome.UNKNOWN.value, Outcome.USER_ABORT.value}:
        return "review_required_trace"
    return "failure_lesson"


def _topic_tags(outcome: str, phase_sequence: list[str]) -> list[str]:
    return sorted({"fspc", "attempt_log", "consumer_contract", outcome, *phase_sequence})


def _quality_for_attempt(outcome: str, density_hint: int) -> ConsumerQuality:
    confidence = 0.85 if outcome == Outcome.WHOLE_CELL_SUCCESS.value else 0.6
    if outcome == Outcome.UNKNOWN.value:
        confidence = 0.35
    support = min(1.0, max(0.2, density_hint / 6.0))
    consistency = 0.9 if outcome != Outcome.UNKNOWN.value else 0.45
    novelty = 0.2
    utility = 0.85 if outcome == Outcome.WHOLE_CELL_SUCCESS.value else 0.7
    risk = 0.15 if outcome == Outcome.WHOLE_CELL_SUCCESS.value else 0.35
    if outcome == Outcome.UNKNOWN.value:
        risk = 0.5
    final_score = max(
        0.0,
        min(
            1.0,
            0.30 * confidence
            + 0.20 * support
            + 0.20 * consistency
            + 0.15 * utility
            + 0.10 * novelty
            - 0.15 * risk,
        ),
    )
    return ConsumerQuality(
        confidence=confidence,
        support=support,
        consistency=consistency,
        novelty=novelty,
        utility=utility,
        risk=risk,
        final_score=final_score,
    )


def _dedupe_key(prefix: str, attempt_id: str, phase_sequence: list[str]) -> str:
    raw = f"{prefix}|{attempt_id}|{'|'.join(phase_sequence)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
