FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/bin/python

WORKDIR /app
COPY . /app/

# Install PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install detectron2 + fvcore (for DensePose)
RUN pip install 'git+https://github.com/facebookresearch/fvcore.git'
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install all dependencies
RUN pip install -r requirements.txt

# Model weights auto-download on first run
ENV PYTHONPATH=CatVTON:.
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
