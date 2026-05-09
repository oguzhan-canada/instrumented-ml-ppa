FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip \
    git cmake build-essential \
    tcl-dev libreadline-dev \
    wget curl unzip \
    parallel \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── OpenROAD-flow-scripts (pinned commit for ASAP7 support) ──────────────────
ARG OPENROAD_COMMIT=4b4c5a7
RUN git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git /opt/openroad && \
    cd /opt/openroad && \
    git checkout ${OPENROAD_COMMIT} && \
    ./build_openroad.sh --minimal 2>&1 | tail -20

ENV PATH="/opt/openroad/tools/install/OpenROAD/bin:${PATH}"
ENV OPENROAD_BIN=/opt/openroad/tools/install/OpenROAD/bin/openroad
ENV YOSYS_BIN=/opt/openroad/tools/install/yosys/bin/yosys
ENV PDK_ROOT=/opt/openroad/flow/platforms

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.1.0+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch-scatter torch-sparse torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

COPY . .

# ── Data mount point ──────────────────────────────────────────────────────────
# Mount large datasets here: docker run -v /mnt/data:/data ...
ENV PPA_DATA_ROOT=/data
RUN mkdir -p /data/raw/openabc_d /data/raw/circuitnet /data/raw/openroad_runs

# ── Smoke test ────────────────────────────────────────────────────────────────
RUN python -c "import torch; import torch_geometric; import xgboost; import botorch; print('All imports OK')"
