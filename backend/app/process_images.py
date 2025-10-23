#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
from ultralytics import YOLO
from detection import process_folder, draw_boxes, is_image
from detection.constants import (
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_IMGSZ,
    DEFAULT_INFERENCE_CONF,
    DEFAULT_BOX_COLOR,
    DEFAULT_BOX_THICKNESS,
    DEFAULT_LABEL_BG_COLOR,
    DEFAULT_LABEL_TEXT_COLOR,
    DEFAULT_FONT_SIZE,
    DEFAULT_LABEL_PADDING,
)
import yaml

# Web UI deps
from flask import Flask, request, send_file, render_template_string, redirect, url_for
import io
from PIL import Image
import webbrowser

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config():
    defaults = {
        "inference": {"device": DEFAULT_INFERENCE_DEVICE, "imgsz": DEFAULT_INFERENCE_IMGSZ, "conf": DEFAULT_INFERENCE_CONF},
        "drawing": {
            "box_color": DEFAULT_BOX_COLOR,
            "box_thickness": DEFAULT_BOX_THICKNESS,
            "label_bg_color": DEFAULT_LABEL_BG_COLOR,
            "label_text_color": DEFAULT_LABEL_TEXT_COLOR,
            "font_size": DEFAULT_FONT_SIZE,
            "label_padding": DEFAULT_LABEL_PADDING,
        },
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
                # merge defaults shallowly
                for k, v in defaults.items():
                    if k not in cfg:
                        cfg[k] = v
                    else:
                        for kk, vv in v.items():
                            cfg[k].setdefault(kk, vv)
                return cfg
        except Exception:
            return defaults
    return defaults


CFG = load_config()

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo12n.pt"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, default=Path("backend/input"))
    p.add_argument("-o", "--output", type=Path, default=Path("backend/output"))
    p.add_argument("-m", "--model", type=Path, default=Path(__file__).resolve().parent / "models" / "yolo12n.pt")
    p.add_argument("--mode", choices=("cli", "web"), default="web", help="Run mode: cli (process folder) or web (start local UI)")
    p.add_argument("--port", default=5000, help="Port for web UI (default 5000)")
    args = p.parse_args()

    args.input.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Attempting to download model automatically...")
        # Run download_model.sh
        download_script = Path(__file__).resolve().parents[2] / "scripts" / "download_model.sh"
        if download_script.exists():
            subprocess.run(["bash", str(download_script)], check=True)
        else:
            raise FileNotFoundError(f"Download script not found: {download_script}")

        # After downloading, verify the model exists
        if not args.model.exists():
            raise FileNotFoundError(f"Failed to download the model to {args.model}")

    print("Loading model:", args.model)
    model = YOLO(str(args.model))

    cfg = load_config()
    inf = cfg.get("inference", {})
    device = inf.get("device", DEFAULT_INFERENCE_DEVICE)
    imgsz = int(inf.get("imgsz", DEFAULT_INFERENCE_IMGSZ))
    conf_val = float(inf.get("conf", DEFAULT_INFERENCE_CONF))

    if args.mode == "cli":
        process_folder(model, args.input, args.output, conf_val, device, imgsz, cfg)
        return

    # Start simple Flask web UI
    app = Flask(__name__)

    INDEX_HTML = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Local Object Detection</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; }
          .row { display:flex; gap:1rem; }
          .col { flex:1 }
          img { max-width:100%; height:auto; }
        </style>
      </head>
      <body>
        <h2>Local Object Detection</h2>
        <p>Загрузите изображение или выберите существующее из папки input.</p>
        <form action="/process" method="post" enctype="multipart/form-data">
          <div>
            <label>Upload image: <input type="file" name="file" accept="image/*"></label>
          </div>
          <div>
            <label>Or choose existing file:
              <select name="choose">
                <option value="">-- none --</option>
                {% for f in files %}
                <option value="{{f}}">{{f}}</option>
                {% endfor %}
              </select>
            </label>
          </div>
          <div style="margin-top:1rem">
            <button type="submit">Process</button>
          </div>
        </form>
        <hr />
        <h3>How it works</h3>
        <p>Server processes the image locally and returns a processed image with bounding boxes. No internet required after model is downloaded.</p>
      </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        files = [p.name for p in sorted(args.input.glob("*")) if is_image(p)]
        return render_template_string(INDEX_HTML, files=files)

    def process_pil_image(img: Image.Image):
        # run model.predict on PIL image and draw boxes
        results = model.predict(source=img, device=device, imgsz=imgsz, conf=conf_val, verbose=False)
        r = results[0]
        boxes = []
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            clss = r.boxes.cls.tolist()
            for xy, c, cls in zip(xyxy, confs, clss):
                if int(cls) == 0:
                    boxes.append({"xyxy": xy, "conf": float(c), "class": int(cls)})
        out_img = draw_boxes(img.convert("RGB"), boxes, cfg)
        bio = io.BytesIO()
        out_img.save(bio, format="PNG")
        bio.seek(0)
        return bio

    @app.route("/process", methods=["POST"])
    def process_route():
        # uploaded file takes precedence
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            img = Image.open(f.stream).convert("RGB")
            buf = process_pil_image(img)
            return send_file(buf, mimetype="image/png", as_attachment=True, download_name="processed.png")

        chosen = request.form.get("choose", "")
        if chosen:
            p = args.input / chosen
            if p.exists() and is_image(p):
                img = Image.open(p).convert("RGB")
                buf = process_pil_image(img)
                return send_file(buf, mimetype="image/png", as_attachment=True, download_name=f"processed_{chosen}")
        return redirect(url_for("index"))

    host = "127.0.0.1"
    port = int(args.port)
    url = f"http://{host}:{port}/"
    print(f"Starting web UI at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
