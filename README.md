# AI People Detector

## Overview
The AI People Detector is a web application that utilizes the Ultralytics YOLOv12-nano model for real-time people detection. The application is structured into a frontend and a backend, allowing for efficient processing and display of detection results.

## Setup Instructions

### Prerequisites
- Node.js and npm installed for the frontend.
- Python 3.x and pip installed for the backend.
- Docker and Docker Compose installed for containerization.

### Backend Setup
1. Navigate to the `backend` directory:
   ```
   cd backend
   ```
2. Install the dependencies:

   1. Install pyenv (recommended) and Python 3.11.4

   ```bash
   # macOS (Homebrew)
   brew install pyenv
   pyenv install 3.11.4
   pyenv local 3.11.4
   # OR on Linux follow https://github.com/pyenv/pyenv#installation
   ```
    
   2. Create the virtualenv and install Python deps

   ```bash
   cd backend
   chmod +x setup_venv.sh
   ./setup_venv.sh
   source .venv/bin/activate
   ```

   3. Download the YOLOv12‑nano model

   ```bash
   ./scripts/download_model.sh
   ```

   4. Add images to `backend/input` and run the processor

   ```bash
   python app/process_images.py --input input --output output 
   ```

   Outputs will appear in `backend/output` with the same filenames, annotated with bounding boxes for detected people.

   Config example (`backend/app/config.yaml`):

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
   label_padding: [40, 10]    # [horizontal, vertical] padding in pixels around text
   ```

   ## Docker (platform independent)

   If you prefer not to install Python locally, use Docker. I can add a Dockerfile if you want; basic workflow:

   ```bash
   # build (example)
   docker build -t ai-people-detector:local .

   # run (mount input/output folders)
   docker run --rm -v $(pwd)/backend/input:/app/input -v $(pwd)/backend/output:/app/output ai-people-detector:local python app/process_images.py --input input --output output
   ```

   ## Troubleshooting

   - If `pip install -r requirements.txt` fails with torch wheel errors, install a CPU torch wheel first:
     ```bash
     python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
     python -m pip install -r requirements.txt
     ```
   - If you cannot install pyenv, set `PYTHON` when running the setup script to point to a Python 3.11 interpreter:
     ```bash
     PYTHON=/path/to/python3.11 ./setup_venv.sh
     ```

   ## Project Structure
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

