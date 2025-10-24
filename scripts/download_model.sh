#!/bin/bash
set -e

# Download YOLOv8 model weights if missing
MODEL_DIR="/app/backend/app/models"
MODEL_FILE="$MODEL_DIR/yolo12n.pt"

# Create directory if not exists
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
else
    echo "Downloading YOLOv8 model..."
    wget -O "$MODEL_FILE" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt || {
        echo "Failed to download model. Check URL or internet access."
        exit 1
    }
    echo "Model downloaded to $MODEL_FILE"
fi
