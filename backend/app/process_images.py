#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
from ultralytics import YOLO
from detection import process_folder
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

    process_folder(model, args.input, args.output, conf_val, device, imgsz, cfg)


if __name__ == "__main__":
    main()
