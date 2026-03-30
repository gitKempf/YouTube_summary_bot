FROM python:3.12-slim

WORKDIR /app

# System deps: ffmpeg + deno (required by yt-dlp for YouTube extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl unzip \
    && curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentence-transformers langchain-neo4j rank-bm25

# Copy app code
COPY src/ src/
COPY main.py .

# Suppress OpenMP duplicate lib warnings (torch/numpy)
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
