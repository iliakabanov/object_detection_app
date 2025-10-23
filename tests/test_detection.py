from pathlib import Path
from PIL import Image
import pytest

from backend.app.detection import detection as det


# === Utility helpers ===

def make_test_image(path: Path, size=(200, 120), color=(10, 20, 30)):
    """Create and save a simple RGB test image."""
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


class FakeBoxes:
    """Mock class for YOLO result boxes with xyxy, conf, cls attributes."""

    def __init__(self, xyxy_list, confs, clss):
        self.xyxy = xyxy_list
        self.conf = confs
        self.cls = clss

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    """Mock YOLO result object."""
    def __init__(self, boxes: FakeBoxes):
        self.boxes = boxes


class FakeModel:
    """Fake YOLO model with predictable output."""
    def __init__(self, results):
        # results: list of FakeResult
        self._results = results

    def predict(self, source, device, imgsz, conf, verbose=False):
        return self._results


# === Tests ===

def test_is_image(tmp_path):
    """Ensure is_image correctly identifies image-like filenames."""
    img_path = tmp_path / "img.JPG"
    img_path.write_text("not an image")
    assert det.is_image(img_path)

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("hello")
    assert not det.is_image(txt_path)


def test_draw_boxes_and_label_area(tmp_path):
    """Verify that bounding boxes and labels are drawn on the image."""
    img_path = make_test_image(tmp_path / "a.jpg", size=(200, 120))
    img = Image.open(img_path).convert("RGB")

    boxes = [{"xyxy": [50, 20, 150, 100], "conf": 0.987, "class": 0}]
    cfg = {
        "drawing": {
            "box_color": [255, 0, 0],
            "box_thickness": 2,
            "label_bg_color": [123, 45, 67],
            "label_text_color": [255, 255, 255],
            "font_size": 12,
            "label_padding": [4, 2],
        }
    }

    out = det.draw_boxes(img.copy(), boxes, cfg)
    assert out.size == img.size

    # Verify a pixel on the rectangle border has changed color
    x0, y0, x1, y1 = boxes[0]["xyxy"]
    # sample a point on the top border
    test_x = int((x0 + x1) / 2)
    test_y = int(y0)  # top edge
    before = img.getpixel((test_x, test_y))
    after = out.getpixel((test_x, test_y))

    assert before != after, f"Expected pixel color to change at {(test_x, test_y)}"



def test_process_folder_happy_path(tmp_path):
    """End-to-end test: process one image with fake model."""
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()

    img_path = make_test_image(inp / "p.jpg")

    boxes = FakeBoxes(xyxy_list=[[10, 10, 50, 50]], confs=[0.9], clss=[0])
    res = FakeResult(boxes)
    model = FakeModel([res])

    det.process_folder(model, inp, out, conf=0.2, device="cpu", imgsz=640, cfg={})

    out_file = out / "p.jpg"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_process_folder_no_images(tmp_path, capsys):
    """Ensure process_folder handles empty input folder gracefully."""
    inp = tmp_path / "empty"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()

    model = FakeModel([])
    det.process_folder(model, inp, out, conf=0.2, device="cpu", imgsz=640, cfg={})
    captured = capsys.readouterr()

    assert "No images in" in captured.out
    assert not any(out.iterdir())
