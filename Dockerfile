# Dockerfile for Booster Soccer RL Training
# Using NVIDIA NGC JAX image (pre-configured JAX + CUDA + cuDNN)

FROM nvcr.io/nvidia/jax:25.01-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install MuJoCo rendering dependencies (Ubuntu 24.04 package names)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libosmesa6 \
    libglfw3 \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Install additional ML dependencies (JAX/Flax already included in NGC image)
RUN pip install --no-cache-dir \
    "mujoco==3.2.6" \
    "mujoco-mjx==3.2.6" \
    "brax==0.12.1" \
    "optax==0.2.4" \
    "numpy==2.1.3" \
    "wandb==0.19.1" \
    "mlflow==2.19.0" \
    "pynvml>=12.0.0"

# Install PyTorch with CUDA 12 support
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Environment variables for JAX/XLA optimization
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
ENV JAX_PREALLOCATE=false
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

# Databricks compatibility
ENV DATABRICKS_RUNTIME_VERSION=16.4

WORKDIR /workspace

# Health check script
RUN printf '#!/bin/bash\n\
python -c "import jax; print(f\"JAX devices: {jax.devices()}\")" \n\
python -c "import mujoco; print(f\"MuJoCo version: {mujoco.__version__}\")" \n\
python -c "import torch; print(f\"PyTorch CUDA: {torch.cuda.is_available()}\")" \n\
' > /usr/local/bin/healthcheck.sh && chmod +x /usr/local/bin/healthcheck.sh

CMD ["/bin/bash"]
