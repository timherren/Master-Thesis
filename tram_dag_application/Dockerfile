# ============================================================
# Dockerfile — TRAM-DAG Causal Analysis Application
#
# Multi-arch image (AMD64 + ARM64) based on rocker/r-ver.
# Includes R, Python (via Miniforge, CPU-only PyTorch), and
# the tramdag package so the Shiny app works out of the box.
#
# Miniforge is used instead of system Python because
# reticulate requires Python built with --enable-shared.
#
# Layer order is optimised for caching:
#   1. System libs  (rarely changes)
#   2. Miniforge    (rarely changes)
#   3. R packages   (slow to compile on ARM64, rarely changes)
#   4. Conda env + Python packages  (may change more often)
#   5. App code     (changes most often)
# ============================================================
FROM rocker/r-ver:4.4.0

# ---- 1. System libraries needed by R packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libglpk-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. Miniforge (conda) — Python with --enable-shared ----
ENV CONDA_DIR=/opt/conda
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" \
      -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniforge.sh && \
    "$CONDA_DIR/bin/conda" clean -afy
ENV PATH="$CONDA_DIR/bin:$PATH"

# ---- 3. R packages (slow on ARM64 — keep this layer cached) ----
RUN R -e "install.packages(c( \
    'shiny', 'shinyFiles', 'visNetwork', 'reticulate', \
    'igraph', 'dplyr', 'jsonlite', 'httr', \
    'promises', 'future' \
  ), repos = 'https://cloud.r-project.org')"

# ---- 4. Python environment with tramdag + CPU-only PyTorch ----
RUN conda create -n tramdag python=3.11 -y && \
    conda clean -afy

RUN conda run -n tramdag pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN conda run -n tramdag pip install --no-cache-dir \
    -i https://test.pypi.org/simple \
    --extra-index-url https://pypi.org/simple \
    tramdag && \
    conda run -n tramdag pip install --no-cache-dir reportlab && \
    sed -i 's/\.applymap(/.map(/g' \
      /opt/conda/envs/tramdag/lib/python3.11/site-packages/tramdag/TramDagModel.py

# ---- 5. Application code ----
WORKDIR /app
COPY app.R .

RUN mkdir -p /root/Downloads

# ---- Environment variables ----
ENV TRAMDAG_PYTHON=/opt/conda/envs/tramdag/bin/python3
ENV LD_PRELOAD=/opt/conda/envs/tramdag/lib/libstdc++.so.6
ENV OLLAMA_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2

EXPOSE 3838

CMD ["R", "-e", "shiny::runApp('/app/app.R', host='0.0.0.0', port=3838)"]
