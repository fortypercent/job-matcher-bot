FROM python:3.11-slim

WORKDIR /app

# gcc нужен для сборки некоторых wheel (asyncpg и др.)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# ── 1. CPU-only PyTorch ПЕРВЫМ ────────────────
# Это самая важная оптимизация: default pip install torch тянет CUDA (~800MB).
# CPU-only версия — ~150MB, экономит и размер образа и RAM.
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Остальные зависимости ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Скачиваем МАЛЕНЬКУЮ модель при сборке ──
# L6 вместо L12: 22M параметров vs 118M, ~80MB vs ~470MB
# Векторы те же 384-мерные — полностью совместимо с существующей БД.
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L6-v2')"

# ── 4. Отключаем всё лишнее для экономии RAM ──
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

COPY . .

CMD ["python", "main.py"]