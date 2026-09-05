# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------
# 1. System packages. Changes almost never, so this layer stays cached.
#    build-essential + cmake are here for two reasons: pycocotools builds
#    from source on slim images, and the C++ voxelizer backend will need
#    them later. Installing them now means no Dockerfile rewrite then.
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --------------------------------------------------------------------------
# 2. PyTorch, CPU wheels only. The assignment requires no training, and the
#    CUDA wheels would add several GB to the image for nothing.
#    Separate layer from requirements.txt because it changes even less often.
# --------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# --------------------------------------------------------------------------
# 3. Remaining Python dependencies.
# --------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# 4. Source code LAST. Any edit here must not invalidate the layers above.
# --------------------------------------------------------------------------
COPY . .
RUN pip install --no-cache-dir -e .

# Run as a non-root user matching the host UID, so files written into the
# bind-mounted workspace are owned by you and not by root.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev
USER dev

CMD ["bash"]