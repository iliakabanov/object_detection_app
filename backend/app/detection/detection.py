from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont


try:
    import importlib

    fm = importlib.import_module("matplotlib.font_manager")
except Exception:
    fm = None

from .constants import (
    DEFAULT_BOX_COLOR,
    DEFAULT_BOX_THICKNESS,
    DEFAULT_LABEL_BG_COLOR,
    DEFAULT_LABEL_TEXT_COLOR,
    DEFAULT_FONT_SIZE,
    DEFAULT_LABEL_PADDING,
)


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def draw_boxes(image: Image.Image, boxes: List[dict], cfg: dict):
    draw = ImageDraw.Draw(image)

    # Font configuration (can be absolute px, fraction <=1, or string '1/50')
    font_cfg_raw = cfg.get("drawing", {}).get("font_size", DEFAULT_FONT_SIZE)

    def compute_font_px(raw, img_h):
        if isinstance(raw, str) and "/" in raw:
            try:
                n, d = raw.split("/")
                frac = float(n) / float(d)
                return max(6, int(img_h * frac))
            except Exception:
                return DEFAULT_FONT_SIZE
        try:
            val = float(raw)
            if 0 < val <= 1:
                return max(6, int(img_h * val))
            else:
                return max(6, int(val))
        except Exception:
            return DEFAULT_FONT_SIZE

    font_size_cfg = compute_font_px(font_cfg_raw, image.height)
    # make label font larger (user requested ~2x)
    try:
        font_size_cfg = int(font_size_cfg)
    except Exception:
        pass

    # Load font (try DejaVuSans, fall back to default)
    font = None
    try:
        if fm is not None:
            font = ImageFont.truetype(fm.findfont("DejaVu Sans"), font_size_cfg)
        else:
            # try a generic truetype; fall back to default
            font = ImageFont.load_default()
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    box_color = tuple(cfg.get("drawing", {}).get("box_color", DEFAULT_BOX_COLOR))
    box_thickness = int(cfg.get("drawing", {}).get("box_thickness", DEFAULT_BOX_THICKNESS))
    label_bg = tuple(cfg.get("drawing", {}).get("label_bg_color", DEFAULT_LABEL_BG_COLOR))
    label_text = tuple(cfg.get("drawing", {}).get("label_text_color", DEFAULT_LABEL_TEXT_COLOR))

    for b in boxes:
        x0, y0, x1, y1 = map(int, b["xyxy"]) if isinstance(b["xyxy"], (list, tuple)) else map(int, b["xyxy"].tolist())
        conf = b["conf"]
        # rectangle
        try:
            draw.rectangle([x0, y0, x1, y1], outline=box_color, width=box_thickness)
        except TypeError:
            for t in range(box_thickness):
                draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=box_color)

        label = f"person {conf:.2f}"
        # measure text width/height
        if font is not None:
            try:
                tw = int(draw.textlength(label, font=font))
            except Exception:
                tw = int(len(label) * getattr(font, "size", font_size_cfg) * 0.6)
            rows = label.count("\n") + 1
            th = int(getattr(font, "size", font_size_cfg) * rows)
        else:
            tw = int(len(label) * font_size_cfg * 0.6)
            rows = label.count("\n") + 1
            th = int(font_size_cfg * rows)

        padding = cfg.get("drawing", {}).get("label_padding", DEFAULT_LABEL_PADDING)
        pad_x = int(padding[0]) if isinstance(padding, (list, tuple)) and len(padding) > 0 else 4
        pad_y = int(padding[1]) if isinstance(padding, (list, tuple)) and len(padding) > 1 else 2
        label_y0 = max(0, y0 - th - pad_y * 2)
        label_x1 = x0 + tw + pad_x * 2
        draw.rectangle([x0, label_y0, label_x1, y0], fill=label_bg)
        draw.text((x0 + pad_x, label_y0 + pad_y), label, fill=label_text, font=font)

    return image


def process_folder(model, inp: Path, out: Path, conf: float, device: str, imgsz: int, cfg: dict):
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in inp.iterdir() if is_image(p)])
    if not imgs:
        print("No images in", inp)
        return
    for p in imgs:
        print("Processing", p.name)
        img = Image.open(p).convert("RGB")
        results = model.predict(source=img, device=device, imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        boxes = []
        if hasattr(r, "boxes") and r.boxes is not None:
            # Support both ultralytics Boxes (with .xyxy/.conf/.cls that implement
            # .tolist()) and simple containers where those attributes are plain lists.
            raw_xyxy = r.boxes.xyxy
            raw_conf = r.boxes.conf
            raw_cls = r.boxes.cls

            xyxy = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else raw_xyxy
            confs = raw_conf.tolist() if hasattr(raw_conf, "tolist") else raw_conf
            clss = raw_cls.tolist() if hasattr(raw_cls, "tolist") else raw_cls

            if xyxy and len(xyxy) > 0:
                for xy, c, cls in zip(xyxy, confs, clss):
                    if int(cls) == 0:  # COCO person class
                        boxes.append({"xyxy": xy, "conf": float(c), "class": int(cls)})
        out_img = draw_boxes(img, boxes, cfg)
        out_path = out / p.name
        out_img.save(out_path)
        print(f"Saved {out_path} ({len(boxes)} persons)")
