# AI People Detector

## Overview
The AI People Detector is a web application that utilizes the Ultralytics YOLOv12-nano model for real-time people detection. The application is structured into a frontend and a backend, allowing for efficient processing and display of detection results.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/iliakabanov/object_detection_app.git
cd object_detection_app
````

### 2. Prerequisites

* Node.js and npm installed for the frontend.
* Python 3.x and pip installed for the backend.
* Docker installed for containerization (optional if using local Python).

### 3. Backend Setup

1. Navigate to the `backend` directory:

```bash
cd backend
```

2. Install the dependencies:

   * Install pyenv (recommended) and Python 3.11.4

   ```bash
   # macOS (Homebrew)
   brew install pyenv
   pyenv install 3.11.4
   pyenv local 3.11.4

   # Linux: follow https://github.com/pyenv/pyenv#installation
   ```

   * Create the virtualenv and install Python dependencies:

   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   source .venv/bin/activate
   ```

3. Download the YOLOv12-nano model:

```bash
./scripts/download_model.sh
```

4. Add images to `backend/input` and run the processor:

```bash
python app/process_images.py --input input --output output
```

Outputs will appear in `backend/output` with the same filenames, annotated with bounding boxes for detected people.

#### Config example (`backend/app/config.yaml`):

```yaml
# Rendering and inference configuration for process_images.py
inference:
  device: cpu
  imgsz: 640
  conf: 0.25

drawing:
  box_color: [255, 0, 0]      # RGB
  box_thickness: 20
  label_bg_color: [100, 100, 100]
  label_text_color: [255, 0, 0]     
  font_size: 1/50               # relative to image height
  label_padding: [40, 10]       # [horizontal, vertical] padding in pixels
```

### 4. Docker Setup (optional)

If you prefer not to install Python locally, you can use Docker.

* **Build the image:**

```bash
docker build -t ai-people-detector:local .
```

* **Run the container:**

**Linux/macOS:**

```bash
docker run --rm -p 5000:5000 \
  -v "$(pwd)/backend/input:/app/backend/app/input" \
  -v "$(pwd)/backend/output:/app/backend/app/output" \
  ai-people-detector:local \
  python backend/app/process_images.py --input /app/backend/app/input
```

**Windows (cmd/powershell):**

```bash
docker run --rm -p 5000:5000 \
  -v "%cd%\backend\input:/app/backend/app/input" \
  -v "%cd%\backend\output:/app/backend/app/output" \
  ai-people-detector:local \
  python backend/app/process_images.py --input /app/backend/app/input
```

### 5. Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 6. Troubleshooting

* If `pip install -r requirements.txt` fails with torch wheel errors:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

* If you cannot install pyenv, set `PYTHON` when running the setup script:

```bash
PYTHON=/path/to/python3.11 ./setup_venv.sh
```

### 7. Project Structure

```
├── README.md
├── backend
│   ├── app
│   │   ├── __init__.py
│   │   ├── detection
│   │   │   ├── __init__.py
│   │   │   ├── constants.py
│   │   │   └── detection.py
│   │   ├── models
│   │   └── process_images.py
│   ├── input
│   ├── output
│   ├── requirements.txt
│   └── setup_venv.sh
└── scripts
   └── download_model.sh
```

