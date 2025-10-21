FROM python:3.10-slim

WORKDIR /images

# COPY pyproject.toml .

# RUN pip install -r pyproject.toml

COPY . .

CMD ["python", "-m", "src.object_detection_app.main.py"]