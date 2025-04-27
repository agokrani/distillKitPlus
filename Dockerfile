FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set non-interactive frontend for apt-get to avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
# Using python3.11 based on modal's add_python="3.11"
RUN apt-get update && \
    apt-get install -y --no-install-recommends git python3.11 python3-pip python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
# Combine dependencies for better layer caching
RUN python3.11 -m pip install --no-cache-dir \
    accelerate \
    transformers==4.49.0 \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    datasets \
    tensorboard \
    trl==0.15.2 \
    bitsandbytes \
    tensorflow \
    h5py \
    tf-keras \
    deepspeed==0.16.4 \
    "huggingface_hub[hf_transfer]" \
    click

# Install peft from specific commit
RUN python3.11 -m pip install --no-cache-dir \
    git+https://github.com/agokrani/peft.git@fix-for-modules-to-save-for-zero3#egg=peft

# Install flash-attn
# Note: FlashAttention installation can be hardware-specific and might require compilation.
# Ensure the base image and environment are compatible.
# Using --no-build-isolation as specified in the modal script.
RUN python3.11 -m pip install --no-cache-dir flash-attn --no-build-isolation

# Copy application code
COPY scripts /app/scripts
COPY components /app/components
COPY config /app/config

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1
# Set Python path if needed, although usually not necessary when using WORKDIR
# ENV PYTHONPATH=/app

# Ensure python points to python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set default command - runs the local script.
# Assumes scripts/local/distill_logits.py is the main execution script.
# User should override the config path if needed, e.g., docker run <image> --config /path/to/my_config.json
# Or mount their config file to /app/config/default_config.json
ENTRYPOINT ["python", "scripts/local/distill_logits.py"]
CMD ["--config", "config/default_config.json"] 