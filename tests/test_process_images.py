import unittest
from unittest import mock
import tempfile
import shutil
from pathlib import Path
from io import StringIO
from PIL import Image
import sys

from backend.app import process_images

class DummyBoxes:
    def __init__(self):
        self.xyxy = [[0, 0, 10, 10]]
        self.conf = [0.95]
        self.cls = [0]


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        self.predict = mock.Mock(return_value=[DummyResult()])


class TestLoadConfig(unittest.TestCase):
    def test_load_config_returns_defaults_if_file_missing(self):
        with mock.patch.object(process_images, "CONFIG_PATH", new=Path("/nonexistent/path.yaml")):
            cfg = process_images.load_config()
            self.assertIn("inference", cfg)
            self.assertIn("drawing", cfg)

    def test_load_config_merges_existing(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("inference:\n  device: cpu\n")
            f.flush()
            with mock.patch.object(process_images, "CONFIG_PATH", new=Path(f.name)):
                cfg = process_images.load_config()
                self.assertEqual(cfg["inference"]["device"], "cpu")
                self.assertIn("imgsz", cfg["inference"])


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmpdir, ignore_errors=True))
        self.inp = Path(self.tmpdir) / "input"
        self.out = Path(self.tmpdir) / "output"
        self.inp.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), color="blue").save(self.inp / "img1.png")
        self.dummy_model = DummyYOLO()

    def test_main_creates_output_and_calls_model(self):
        # Set up all mocks
        mock_stdout = StringIO()
        patchers = [
            mock.patch("sys.argv", ["process_images.py", "-i", str(self.inp), "-o", str(self.out), "--mode", "cli"]),
            mock.patch("sys.stdout", mock_stdout),
            mock.patch("backend.app.process_images.YOLO", autospec=True, return_value=self.dummy_model),
            mock.patch("backend.app.process_images.MODEL_PATH", Path(self.tmpdir) / "yolo12n.pt"),
            mock.patch("backend.app.process_images.Path.exists", return_value=True)
        ]
        
        # Start all patchers
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        # Run the test
        process_images.main()
        
        # Verify results
        output = mock_stdout.getvalue()
        self.assertIn("Loading model:", output)
        self.assertTrue(self.dummy_model.predict.called)


class TestProcessPILIntegration(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (10, 10), color="green")
        self.cfg = process_images.load_config()

    @mock.patch("backend.app.process_images.YOLO", new=DummyYOLO)
    def test_draw_boxes_called(self):
        boxes = [{"xyxy": [0, 0, 5, 5], "conf": 0.9, "class": 0}]
        out_img = process_images.draw_boxes(self.img.copy(), boxes, self.cfg)
        self.assertIsInstance(out_img, Image.Image)


if __name__ == "__main__":
    unittest.main()
