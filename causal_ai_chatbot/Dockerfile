FROM python:3.11-slim

# System dependencies for scientific Python and R-based CI tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    r-base \
    r-base-dev \
    nodejs \
    libnode-dev \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better Docker layer caching
COPY requirements_chatbot.txt /app/requirements_chatbot.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements_chatbot.txt

# Install CPU-only PyTorch + tramdag package (same pattern as tram_dag_application)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple tramdag

# Install required R packages for DAG validator wrapper
RUN Rscript -e "install.packages(c('dagitty','igraph','dplyr','tibble','jsonlite','httr2','comets'), repos='https://cloud.r-project.org')"
RUN Rscript -e "library(V8); library(dagitty); library(comets); cat('R package check OK\n')"

# Copy application code
COPY . /app

# Ensure runtime folders exist in image
RUN python - <<'PY'
from app.config import ensure_runtime_dirs
ensure_runtime_dirs()
print("Runtime directories created.")
PY

EXPOSE 8000

CMD ["python", "chatbot_server.py"]
