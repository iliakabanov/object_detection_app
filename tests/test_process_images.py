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
        pass

    def predict(self, source, device, imgsz, conf, verbose):
        return [DummyResult()]


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
        self.inp.mkdir()
        Image.new("RGB", (10, 10), color="blue").save(self.inp / "img1.png")

    dummy_model = DummyYOLO()
    @mock.patch("backend.app.process_images.YOLO", return_value=dummy_model)
    @mock.patch("sys.stdout", new_callable=StringIO)
    @mock.patch("sys.argv", new_callable=lambda: ["process_images.py", "-i", "", "-o", ""])
    def test_main_creates_output_and_calls_model(self, fake_argv, fake_out):
        with mock.patch("backend.app.process_images.Path.mkdir"):
            with mock.patch("backend.app.process_images.MODEL_PATH", self.tmpdir / "yolo12n.pt"):
                with mock.patch("backend.app.process_images.Path.exists", return_value=True):
                    dummy_model = DummyYOLO()
                    with mock.patch("backend.app.process_images.YOLO", return_value=dummy_model):
                        process_images.main()
                        output = fake_out.getvalue()
                        self.assertIn("Loading model:", output)
                        self.assertTrue(dummy_model.predict.called)


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
