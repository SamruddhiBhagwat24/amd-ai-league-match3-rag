# Model Setup

This project uses two local pretrained models:

- `models/bge-small-en-v1.5/` — text embeddings
- `models/Qwen2.5-0.5B-Instruct/` — answer generation

The model weights are intentionally excluded from Git version control because of their size. Place both model directories under `models/` before building the Docker image.

The Docker build must include these model directories to support offline inference.
