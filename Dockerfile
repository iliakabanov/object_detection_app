FROM python:3.10-slim

WORKDIR /images

COPY . .

# RUN pip install .


CMD ["python", "-m", "src.object_detection_app.main"]