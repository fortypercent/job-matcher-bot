FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# No separate torch install — we use ONNX Runtime instead
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model downloads on first run (L12 is accessible, L6 is gated)
# Pre-download at build time using the L12 model which works
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "\
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('paraphrase-multilingual-MiniLM-L12-v2')" || true

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

COPY . .

CMD ["python", "main.py"]