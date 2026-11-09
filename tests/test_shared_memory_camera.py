from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autopatcher.adapters.shared_memory_camera import SharedMemoryCameraAdapter, SharedMemoryCameraConfig
from autopatcher.rendering.shared_memory import SharedFramePublisher, clear_backing_file_header


class SharedMemoryCameraTests(unittest.TestCase):
    def test_shared_memory_camera_maps_latest_frame_into_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            backing_file = temp_path / "fusionbt_latest.bin"
            metadata_file = temp_path / "fusionbt_latest.json"
            metadata_file.write_text(
                json.dumps(
                    {
                        "camera_model": "ORCA-Fusion BT",
                        "exposure_ms": 3.0,
                        "binning": "1x1",
                        "scan_mode": "normal",
                        "roi_x": 128,
                        "roi_y": 256,
                        "roi_width": 2304,
                        "roi_height": 1152,
                        "sensor_temperature_c": -8.3,
                        "camera_timestamp_sec": 12,
                        "camera_timestamp_microsec": 345,
                        "streaming_state": "native-preview",
                    }
                ),
                encoding="utf-8",
            )
            clear_backing_file_header(backing_file=backing_file)
            publisher = SharedFramePublisher(backing_file=backing_file)
            publisher.publish(
                width=2304,
                height=1152,
                rowbytes=4608,
                pixel_type=2,
                frame_index=42,
                timestamp_ns=time.time_ns(),
                payload=b"\x00" * (4608 * 4),
            )

            adapter = SharedMemoryCameraAdapter(
                SharedMemoryCameraConfig(
                    backing_file=backing_file,
                    metadata_file=metadata_file,
                    stale_after_seconds=10.0,
                )
            )
            observation = adapter.observe()

        self.assertEqual(observation.frame_id, "shared-frame-42")
        self.assertEqual(observation.frame_geometry.width, 2304)
        self.assertEqual(observation.frame_geometry.bit_depth, 16)
        self.assertEqual(observation.camera_model, "ORCA-Fusion BT")
        self.assertEqual(observation.acquisition.exposure_ms, 3.0)
        self.assertEqual(observation.roi_geometry.x, 128)
        self.assertEqual(observation.roi_geometry.y, 256)
        self.assertEqual(observation.roi_geometry.width, 2304)
        self.assertEqual(observation.roi_geometry.height, 1152)
        self.assertEqual(observation.quality.sensor_temperature_c, -8.3)
        self.assertFalse(observation.quality.stale)
        self.assertTrue(observation.status.valid)
        self.assertEqual(observation.image_ref.shared_frame_version, 1)
        self.assertIsNotNone(observation.timestamp_camera)

    def test_shared_memory_camera_reports_missing_frame_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backing_file = Path(temp_dir) / "empty.bin"
            clear_backing_file_header(backing_file=backing_file)
            adapter = SharedMemoryCameraAdapter(
                SharedMemoryCameraConfig(
                    backing_file=backing_file,
                    stale_after_seconds=0.1,
                )
            )
            observation = adapter.observe()

        self.assertFalse(observation.status.valid)
        self.assertEqual(observation.status.degraded_reason, "no_shared_frame")

    def test_shared_memory_camera_tolerates_missing_backing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backing_file = Path(temp_dir) / "missing.bin"
            adapter = SharedMemoryCameraAdapter(
                SharedMemoryCameraConfig(
                    backing_file=backing_file,
                    stale_after_seconds=0.1,
                )
            )
            observation = adapter.observe()

        self.assertFalse(observation.status.valid)
        self.assertEqual(observation.status.degraded_reason, "no_shared_frame")

    def test_shared_memory_camera_tolerates_partial_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            backing_file = temp_path / "fusionbt_latest.bin"
            metadata_file = temp_path / "fusionbt_latest.json"
            metadata_file.write_text('{"camera_model": ', encoding="utf-8")
            clear_backing_file_header(backing_file=backing_file)
            publisher = SharedFramePublisher(backing_file=backing_file)
            publisher.publish(
                width=512,
                height=512,
                rowbytes=1024,
                pixel_type=2,
                frame_index=7,
                timestamp_ns=time.time_ns(),
                payload=b"\x00" * (1024 * 4),
            )

            adapter = SharedMemoryCameraAdapter(
                SharedMemoryCameraConfig(
                    backing_file=backing_file,
                    metadata_file=metadata_file,
                    stale_after_seconds=10.0,
                )
            )
            observation = adapter.observe()

        self.assertTrue(observation.status.valid)
        self.assertEqual(observation.frame_id, "shared-frame-7")
        self.assertEqual(observation.frame_geometry.width, 512)
        self.assertIsNone(observation.camera_model)
