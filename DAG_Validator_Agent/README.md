# DAG Validator Agent

An interactive Shiny application for validating causal DAGs using conditional independence (CI) testing. The app includes a built-in LLM (via Ollama) that can propose DAG structures and interpret CI test results.

---

## Prerequisites

Install these two applications (one-time setup):

1. **Docker Desktop** — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. **Ollama** — [ollama.com/download](https://ollama.com/download)

| Platform | Docker | Ollama |
|----------|--------|--------|
| **Mac** | Download `.dmg`, drag to Applications, open once | Download `.dmg`, drag to Applications — runs automatically in background |
| **Windows** | Download installer, follow prompts, open once | Download installer, follow prompts — runs automatically as a service |
| **Linux** | [Install Docker Engine](https://docs.docker.com/engine/install/) + `sudo systemctl start docker` | `curl -fsSL https://ollama.com/install.sh \| sh` |

Make sure Docker is **running** before launching the app:
- **Mac / Windows**: Open Docker Desktop and wait for the whale icon to stop animating.
- **Linux**: Run `sudo systemctl start docker` (or enable it at boot with `sudo systemctl enable docker`).

Ollama runs in the background automatically after installation — you never need to open it manually.

---

## Quick Start

### Mac

1. Open the `RUN_APP` folder inside `shiny_dag_ci_app`
2. Double-click **`start.command`**
3. Wait for the setup to complete (first launch takes a few minutes)
4. Your browser will open automatically to `http://localhost:3838`

### Windows

1. Open the `RUN_APP` folder inside `shiny_dag_ci_app`
2. Double-click **`start.bat`**
3. Wait for the setup to complete (first launch takes a few minutes)
4. Your browser will open automatically to `http://localhost:3838`

### Linux

1. Open a terminal in the `shiny_dag_ci_app` folder
2. Run: `chmod +x RUN_APP/start.sh && ./RUN_APP/start.sh`
3. Wait for the setup to complete (first launch takes a few minutes)
4. Your browser will open automatically to `http://localhost:3838`

The start script automatically:
- Checks that Docker and Ollama are installed and running
- Starts Ollama if it is not already running
- Downloads the LLM model on first run (~700 MB, one-time)
- Builds and starts the Shiny app in Docker
- Opens your browser when ready

> **First run**: The R environment builds inside Docker (~5 min) and the LLM model downloads (~700 MB). Both are cached, so subsequent starts take only a few seconds.

---

## Stopping the App

- **Mac**: Double-click **`STOP_APP/stop.command`**
- **Windows**: Double-click **`STOP_APP\stop.bat`**
- **Linux**: Run `./STOP_APP/stop.sh`

This stops the Shiny app container. Ollama continues running in the background (no re-download needed on next start).

---

## Sample Data

A sample dataset is included in the **`Data/`** folder for testing:

- **`cd3cd28.xls`** — Sachs et al. (2005) flow cytometry dataset (1000 observations, 11 signaling proteins: Raf, Mek, PLCg, PIP2, PIP3, Erk, Akt, PKA, PKC, p38, JNK)

Upload this file in the app to try out DAG validation with a well-known causal discovery benchmark.

---

## Using the App

1. **Upload your data** — CSV or Excel file with numeric columns (or use the sample data from the `Data/` folder)
2. **Preprocessing** — optionally apply log transform or z-score standardization
3. **Define a DAG** — manually draw edges in the visual editor, upload an adjacency matrix, or let the LLM propose one
4. **Run CI tests** — the app tests all conditional independencies implied by your DAG against the data
5. **Review results** — plots show which CI statements hold and which are rejected
6. **Export a report** — download a ZIP with a PDF report, the data, and a reproducible R script

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Docker is not installed" | Install Docker Desktop and restart |
| "Docker is not running" | Open Docker Desktop and wait for it to finish starting |
| "Ollama is not installed" | Install Ollama from [ollama.com/download](https://ollama.com/download) |
| LLM features are slow | Make sure Ollama is installed natively (not only inside Docker) |
| App doesn't open after 3 minutes | Run `docker compose logs -f` in a terminal to check progress |
| Browser shows connection error | Wait a moment and refresh — the app may still be starting |

---

## For Developers

To apply code changes, edit the files and double-click the start script again — it rebuilds automatically.

Key files:
- `app.R` — main Shiny UI and server logic
- `R/` — helper modules (CI testing, DAG utilities, LLM calls, plotting)
- `report_template.Rmd` — PDF report template
- `docker-compose.yml` — container config (Shiny app only; Ollama runs natively)

### Architecture

- **Shiny app** runs inside Docker (consistent R environment across platforms)
- **Ollama** runs natively on the host machine (uses GPU for fast LLM inference) if we install LLM in docker we are forced to use CPU, which takes minutes to calculate
- The Docker container connects to the host's Ollama via `host.docker.internal:11434`
