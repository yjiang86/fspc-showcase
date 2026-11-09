from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autopatcher.paths import (
    external_refs_dir,
    fsp_runtime_root,
    resolve_existing_artifact_path,
    resolve_runtime_path,
    runs_dir,
    runtime_dir,
)


class RuntimePathTests(unittest.TestCase):
    def test_explicit_runtime_root_owns_runs_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {"FSP_RUNTIME_ROOT": temp_dir}
            root = Path(temp_dir).resolve()
            self.assertEqual(fsp_runtime_root(env), root)
            self.assertEqual(runs_dir(env), root / "runs")
            self.assertEqual(runtime_dir(env), root / "runtime")
            self.assertEqual(external_refs_dir(env), root / "runtime" / "external_refs")

    def test_explicit_runtime_root_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an absolute path"):
            fsp_runtime_root({"FSP_RUNTIME_ROOT": "relative/runtime"})

    def test_explicit_runtime_root_must_be_outside_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "must resolve outside"):
            fsp_runtime_root({"FSP_RUNTIME_ROOT": str(ROOT / "runtime")})

    def test_local_app_data_is_the_windows_style_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {"LOCALAPPDATA": temp_dir}
            self.assertEqual(
                fsp_runtime_root(env),
                Path(temp_dir).resolve() / "fsp",
            )

    def test_xdg_data_home_is_the_portable_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {"XDG_DATA_HOME": temp_dir}
            self.assertEqual(
                fsp_runtime_root(env),
                Path(temp_dir).resolve() / "fsp",
            )

    def test_home_data_directory_is_the_last_resort_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir).resolve()
            with patch("autopatcher.paths.Path.home", return_value=home):
                self.assertEqual(
                    fsp_runtime_root({}),
                    home / ".local" / "share" / "fsp",
                )

    def test_platform_default_roots_must_be_absolute(self) -> None:
        for variable in ("LOCALAPPDATA", "XDG_DATA_HOME"):
            with self.subTest(variable=variable):
                with self.assertRaisesRegex(ValueError, "must be an absolute path"):
                    fsp_runtime_root({variable: "relative/runtime"})

    def test_platform_default_roots_must_be_outside_repository(self) -> None:
        for variable in ("LOCALAPPDATA", "XDG_DATA_HOME"):
            with self.subTest(variable=variable):
                with self.assertRaisesRegex(ValueError, "must resolve outside"):
                    fsp_runtime_root({variable: str(ROOT)})

    def test_runtime_markers_resolve_under_external_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            env = {"FSP_RUNTIME_ROOT": str(root)}
            self.assertEqual(
                resolve_runtime_path("runs/result.json", env=env),
                root / "runs" / "result.json",
            )
            self.assertEqual(
                resolve_runtime_path("runtime/frames/latest.tif", env=env),
                root / "runtime" / "frames" / "latest.tif",
            )

    def test_runtime_token_resolves_under_external_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            env = {"FSP_RUNTIME_ROOT": str(root)}
            self.assertEqual(
                resolve_runtime_path(
                    "${FSP_RUNTIME_ROOT}/runtime/shared_memory/latest.bin",
                    env=env,
                ),
                root / "runtime" / "shared_memory" / "latest.bin",
            )

    def test_other_relative_paths_stay_config_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            self.assertEqual(
                resolve_runtime_path("protocol.json", base_dir=base_dir),
                (base_dir / "protocol.json").resolve(),
            )

    def test_tracked_data_path_can_resolve_without_becoming_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            artifact = data_root / "data" / "fixture.json"
            artifact.parent.mkdir()
            artifact.write_text("{}", encoding="utf-8")
            self.assertEqual(
                resolve_existing_artifact_path("old/prefix/data/fixture.json", data_root=data_root),
                artifact.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
