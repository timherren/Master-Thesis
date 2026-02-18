# ============================================================
# Dockerfile — DAG Validator Agent (R Shiny app)
# Uses rocker/r-ver which supports both AMD64 and ARM64
# ============================================================
FROM rocker/r-ver:4.4.0

# System libraries required by R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libglpk-dev \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    lmodern \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# R packages (cached in a Docker layer — only rebuilt when this list changes)
RUN R -e "install.packages(c( \
    'shiny', 'dplyr', 'tibble', 'igraph', 'httr2', \
    'visNetwork', 'dagitty', 'shinybusy', 'DT', \
    'dotenv', 'readxl', 'rmarkdown', 'zip', 'comets' \
  ), repos = 'https://cloud.r-project.org')"

# App code — this layer is rebuilt whenever you change app files
WORKDIR /app
COPY app.R .
COPY report_template.Rmd .
COPY R/ R/
# NOTE: .env is NOT copied — environment variables come from docker-compose.yml.
# For local dev outside Docker, copy .env.example to .env manually.

EXPOSE 3838

CMD ["R", "-e", "shiny::runApp('/app', host='0.0.0.0', port=3838)"]
