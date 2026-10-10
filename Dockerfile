FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Do not let pip resolve a different PyTorch build.
RUN pip install --no-cache-dir --no-deps \
    "transformers==5.19.0" \
    "tokenizers==0.23.2" \
    "safetensors==0.8.0"

# These are lightweight runtime dependencies normally present in the base image.
# Install only if absent, without dependency resolution against torch.
RUN pip install --no-cache-dir --no-deps \
    "huggingface-hub==1.33.0" \
    "accelerate==1.15.0" \
    "pypdf" \
    "python-docx" \
    "openpyxl"

COPY app/ /app/
COPY models/ /app/models/

RUN mkdir -p /app/corpus /app/output

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
