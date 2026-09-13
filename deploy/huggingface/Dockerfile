# QueueIQ Vision API — Hugging Face Space (Docker SDK)
#
# Built and served by Hugging Face from the Space repository that
# deploy/huggingface/publish_space.py uploads. The Space listens on 7860 and
# Hugging Face terminates HTTPS in front of it.

FROM python:3.12-slim

# ultralytics installs opencv-python, which needs these libraries at import time.
# Without them YOLO never loads and the server falls back to placeholder boxes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    QUEUEIQ_PUBLIC=1 \
    QUEUEIQ_TRUST_PROXY=1
WORKDIR /home/user/app

# CPU wheels first: the free Space has no GPU, and the default x86 wheel on PyPI
# bundles several gigabytes of CUDA libraries that would never be used.
COPY --chown=user requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir --user torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt -r requirements-server.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
