FROM python:3.11-slim AS base

WORKDIR /app

COPY . .

RUN chmod +x backend/setup_venv.sh && \
    cd backend && \
    PYTHON=/usr/local/bin/python3 ./setup_venv.sh

ENV PATH="/app/backend/.venv/bin:$PATH"

CMD ["/app/backend/.venv/bin/python", "backend/app/process_images.py"]
