import unittest
import tempfile
from pathlib import Path
from io import StringIO
from unittest import mock
from PIL import Image
import shutil

from backend.app.detection.detection import is_image, draw_boxes, process_folder


class DummyBoxes:
    def __init__(self):
        self.xyxy = [[0, 0, 10, 10]]
        self.conf = [0.95]
        self.cls = [0]


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()


class DummyModel:
    def predict(self, source, device, imgsz, conf, verbose):
        return [DummyResult()]

class TestIsImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        self.img_path = Path(self.tmpdir) / "test.png"
        Image.new("RGB", (10, 10), color="red").save(self.img_path)

    def test_valid_image(self):
        self.assertTrue(is_image(self.img_path))

    def test_invalid_extension(self):
        fake = Path(self.tmpdir) / "test.txt"
        fake.write_text("not an image")
        self.assertFalse(is_image(fake))

    def test_nonexistent_file(self):
        self.assertFalse(is_image(Path(self.tmpdir) / "no.png"))


class TestDrawBoxes(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (100, 100), color="white")
        self.box = [{"xyxy": [10, 10, 50, 50], "conf": 0.9, "class": 0}]
        self.cfg = {
            "drawing": {
                "box_color": (255, 0, 0),
                "box_thickness": 2,
                "label_bg_color": (0, 0, 0),
                "label_text_color": (255, 255, 255),
                "font_size": 10,
                "label_padding": (2, 2),
            }
        }

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_valid_draw(self, _):
        out = draw_boxes(self.img.copy(), self.box, self.cfg)
        self.assertIsInstance(out, Image.Image)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_empty_boxes(self, _):
        out = draw_boxes(self.img.copy(), [], self.cfg)
        self.assertIsInstance(out, Image.Image)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_invalid_font_size(self, _):
        cfg = {"drawing": {"font_size": "invalid"}}
        out = draw_boxes(self.img.copy(), self.box, cfg)
        self.assertIsInstance(out, Image.Image)


class TestProcessFolder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        self.inp = self.tmpdir / "input"
        self.out = self.tmpdir / "output"
        self.inp.mkdir()
        Image.new("RGB", (10, 10), color="blue").save(self.inp / "img1.png")
        self.cfg = {"drawing": {}}
        self.model = DummyModel()

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_process_folder_valid(self, fake_out):
        process_folder(self.model, self.inp, self.out, conf=0.5, device="cpu", imgsz=640, cfg=self.cfg)
        output = fake_out.getvalue()
        self.assertIn("Processing", output)
        self.assertIn("Saved", output)
        self.assertTrue(any(self.out.glob("*.png")))

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_process_folder_no_images(self, fake_out):
        empty_dir = self.tmpdir / "empty"
        empty_dir.mkdir()
        process_folder(self.model, empty_dir, self.out, conf=0.5, device="cpu", imgsz=640, cfg=self.cfg)
        self.assertIn("No images", fake_out.getvalue())


class TestProcessFolderMock(unittest.TestCase):
    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_mock_stdout(self, fake_out):
        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "inp"
            out = Path(tmp) / "out"
            inp.mkdir()
            Image.new("RGB", (10, 10), color="green").save(inp / "x.png")

            process_folder(model, inp, out, 0.5, "cpu", 640, {"drawing": {}})
            output = fake_out.getvalue()
            self.assertIn("Processing", output)
            self.assertIn("Saved", output)


if __name__ == "__main__":
    unittest.main()
