FROM nvcr.io/nvidia/tritonserver:23.04-py3
RUN pip install -U pip
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install diffusers==0.15.1 transformers==4.26.0 accelerate==0.15.0

