# Dockerfile for Booster Soccer RL Training
# Optimized for Databricks on GCP with L4 GPU

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    # MuJoCo rendering dependencies
    libgl1-mesa-glx \
    libosmesa6 \
    libglfw3 \
    libglew-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install JAX with CUDA 12 support
RUN pip install "jax[cuda12]==0.4.38" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install ML dependencies
RUN pip install \
    "flax==0.10.2" \
    "optax==0.2.4" \
    "mujoco==3.2.6" \
    "mujoco-mjx==3.2.6" \
    "brax==0.12.1" \
    "numpy==2.1.3" \
    "wandb==0.19.1" \
    "mlflow==2.19.0" \
    "pynvml>=12.0.0"

# Install PyTorch with CUDA 12.6 support
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Environment variables for JAX/XLA optimization
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
ENV JAX_PREALLOCATE=false
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

# Databricks compatibility
ENV DATABRICKS_RUNTIME_VERSION=16.4

WORKDIR /workspace

# Health check script
RUN echo '#!/bin/bash\n\
python -c "import jax; print(f\"JAX devices: {jax.devices()}\")"\n\
python -c "import mujoco; print(f\"MuJoCo version: {mujoco.__version__}\")"\n\
python -c "import torch; print(f\"PyTorch CUDA: {torch.cuda.is_available()}\")"\n\
' > /usr/local/bin/healthcheck.sh && chmod +x /usr/local/bin/healthcheck.sh

CMD ["/bin/bash"]
