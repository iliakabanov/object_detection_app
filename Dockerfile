# Stage 1: Build dependencies and download model
FROM python:3.11-slim AS builder

WORKDIR /app

# Install essential system libs for OpenCV and wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first (for layer caching)
COPY backend/requirements.txt ./backend/requirements.txt

# Install dependencies
RUN python -m pip install --no-cache-dir -r backend/requirements.txt

# Copy application code (no input/ or output/)
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Download YOLO model
RUN chmod +x scripts/download_model.sh && ./scripts/download_model.sh


# Stage 2: Final runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system libs (needed for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy installed packages and app from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /app /app

# Ensure permissions
RUN chmod +x /app/scripts/download_model.sh

# Default command
CMD ["python", "backend/app/process_images.py"]

