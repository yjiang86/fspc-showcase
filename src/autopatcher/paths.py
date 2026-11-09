from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Mapping


FSP_RUNTIME_ROOT_ENV = "FSP_RUNTIME_ROOT"
FSP_RUNTIME_ROOT_TOKEN = "${FSP_RUNTIME_ROOT}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validated_external_runtime_root(root: Path, *, source: str) -> Path:
    expanded = root.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{source} must be an absolute path")
    resolved_root = expanded.resolve()
    resolved_repository = repo_root().resolve()
    if resolved_root == resolved_repository or resolved_root.is_relative_to(
        resolved_repository
    ):
        raise ValueError(f"{source} must resolve outside the repository")
    return resolved_root


def fsp_runtime_root(env: Mapping[str, str] | None = None) -> Path:
    variables = os.environ if env is None else env
    override = variables.get(FSP_RUNTIME_ROOT_ENV, "").strip()
    if override:
        return _validated_external_runtime_root(
            Path(override),
            source=FSP_RUNTIME_ROOT_ENV,
        )

    local_app_data = variables.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return _validated_external_runtime_root(
            Path(local_app_data) / "fsp",
            source="LOCALAPPDATA",
        )

    xdg_data_home = variables.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return _validated_external_runtime_root(
            Path(xdg_data_home) / "fsp",
            source="XDG_DATA_HOME",
        )

    return _validated_external_runtime_root(
        Path.home() / ".local" / "share" / "fsp",
        source="user-home runtime root",
    )


def runs_dir(env: Mapping[str, str] | None = None) -> Path:
    return fsp_runtime_root(env) / "runs"


def runtime_dir(env: Mapping[str, str] | None = None) -> Path:
    return fsp_runtime_root(env) / "runtime"


def external_refs_dir(env: Mapping[str, str] | None = None) -> Path:
    return runtime_dir(env) / "external_refs"


def resolve_runtime_path(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    raw = os.fspath(raw_path)
    normalized = raw.replace("\\", "/")
    root = fsp_runtime_root(env)

    if normalized == FSP_RUNTIME_ROOT_TOKEN:
        return root
    token_prefix = FSP_RUNTIME_ROOT_TOKEN + "/"
    if normalized.startswith(token_prefix):
        return root / PurePosixPath(normalized[len(token_prefix) :])

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    first_component = normalized.split("/", 1)[0]
    if first_component in {"runs", "runtime"}:
        return root / PurePosixPath(normalized)

    if base_dir is not None:
        return (base_dir / path).resolve()
    return path


def resolve_existing_artifact_path(
    raw_path: str | os.PathLike[str],
    *,
    data_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()

    normalized = os.fspath(raw_path).replace("\\", "/")
    runtime_root = fsp_runtime_root(env)
    repository_data_root = repo_root() if data_root is None else data_root

    for marker in ("runtime/", "runs/"):
        index = normalized.find(marker)
        if index >= 0:
            rebound = runtime_root / PurePosixPath(normalized[index:])
            if rebound.exists():
                return rebound.resolve()

    data_index = normalized.find("data/")
    if data_index >= 0:
        rebound = repository_data_root / PurePosixPath(normalized[data_index:])
        if rebound.exists():
            return rebound.resolve()

    return candidate
