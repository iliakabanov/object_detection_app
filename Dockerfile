# Use Python 3.11 with Debian Bullseye (more libraries preinstalled than slim)
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies needed for building packages and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Pre-install CPU PyTorch to satisfy ultralytics dependency
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy backend folder and the rest of the project
COPY backend/ backend/
COPY . .

# Download YOLO model
RUN chmod +x scripts/download_model.sh && ./scripts/download_model.sh

# Install all Python requirements globally in the container
RUN python -m pip install -r backend/requirements.txt

# Default command to run your app
CMD ["python", "backend/app/process_images.py"]
