FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. NO model download at build time — Railway blocks HuggingFace
#    Model downloads on first run instead.

ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

COPY . .

CMD ["python", "main.py"]