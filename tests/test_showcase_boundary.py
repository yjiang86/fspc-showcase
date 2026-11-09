import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ShowcaseBoundaryTests(unittest.TestCase):
    def test_example_is_mock_only_and_releases_pressure(self):
        spec = importlib.util.spec_from_file_location("showcase_mock_example", ROOT / "examples/mock_baseline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run()
        self.assertEqual(result["mode"], "MOCK_ONLY")
        self.assertEqual(result["pressure_mbar_after_terminal"], 0.0)
        self.assertEqual(result["physical_patch_success"], "NOT_TESTED")

    def test_no_training_or_vendor_binary_payload(self):
        self.assertFalse((ROOT / "data").exists())
        self.assertFalse((ROOT / "native").exists())
        self.assertFalse((ROOT / "src/autopatcher/world_model").exists())
        for pattern in ("*.dll", "*.so", "*.dylib", "*.pt", "*.pth", "*.tif", "*.tiff"):
            self.assertEqual(list(ROOT.rglob(pattern)), [])


if __name__ == "__main__":
    unittest.main()
