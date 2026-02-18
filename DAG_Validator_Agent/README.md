# DAG Validator Agent

An interactive Shiny application for validating causal DAGs using conditional independence (CI) testing. The app includes a built-in LLM (via Ollama) that can propose DAG structures and interpret CI test results.

---

## Prerequisites

Install these two applications (one-time setup):

1. **Docker Desktop** [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. **Ollama** [ollama.com/download](https://ollama.com/download)

| Platform | Docker | Ollama |
|----------|--------|--------|
| **Mac** | Download `.dmg`, drag to Applications, open once | Download `.dmg`, drag to Applications, runs automatically in background |
| **Windows** | Download installer, follow prompts, open once | Download installer, follow prompts, runs automatically as a service |
| **Linux** | [Install Docker Engine](https://docs.docker.com/engine/install/) + `sudo systemctl start docker` | `curl -fsSL https://ollama.com/install.sh \| sh` |

Make sure Docker is **running** before launching the app:
- **Mac / Windows**: Open Docker Desktop and wait for the whale icon to stop animating.
- **Linux**: Run `sudo systemctl start docker` (or enable it at boot with `sudo systemctl enable docker`).

Ollama runs in the background automatically after installation, you never need to open it manually.

---

## Download

**Recommended: Clone via Terminal** (avoids macOS/Windows security warnings):

1. Open a terminal:
   - **Mac**: Press `Cmd + Space`, type **Terminal**, press Enter
   - **Windows**: Press `Win + R`, type **cmd**, press Enter
   - **Linux**: Open your terminal application
2. Choose where to download (e.g. your Desktop) and run:
   ```bash
   cd ~/Desktop
   git clone https://github.com/timherren/Master-Thesis.git
   cd Master-Thesis/DAG_Validator_Agent
   ```
3. On Mac/Linux only, make the scripts executable (run this inside `DAG_Validator_Agent/`):
   ```bash
   chmod +x RUN_APP/start.command RUN_APP/start.sh STOP_APP/stop.command STOP_APP/stop.sh
   ```

**Alternative:** Download as ZIP from GitHub → Extract → see the security notes below if scripts are blocked.

---

## Quick Start

All commands below assume you are inside the `DAG_Validator_Agent/` folder.

### Mac

1. In Finder, navigate to `DAG_Validator_Agent/RUN_APP/`
2. Double-click **`start.command`**
3. Wait for the setup to complete (first launch takes a few minutes)
4. Your browser will open automatically to `http://localhost:3838`

> **If macOS blocks the script** (only happens with ZIP downloads, not `git clone`):
> Right-click → Open instead of double-clicking. If still blocked, open Terminal, navigate to the `DAG_Validator_Agent/` folder, and run:
> ```bash
> cd ~/Desktop/Master-Thesis/DAG_Validator_Agent
> xattr -cr RUN_APP/ STOP_APP/
> ```

### Windows

1. In File Explorer, navigate to `DAG_Validator_Agent\RUN_APP\`
2. Double-click **`start.bat`**
3. Wait for the setup to complete (first launch takes a few minutes)
4. Your browser will open automatically to `http://localhost:3838`

> **If Windows SmartScreen blocks the script** (only happens with ZIP downloads, not `git clone`):
> Click "More info" → "Run anyway".

### Linux

1. Open a terminal and navigate to the `DAG_Validator_Agent/` folder:
   ```bash
   cd ~/Desktop/Master-Thesis/DAG_Validator_Agent
   ```
2. Run: `./RUN_APP/start.sh`
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

Inside the `DAG_Validator_Agent/` folder:

- **Mac**: In Finder, open `STOP_APP/` and double-click **`stop.command`**
- **Windows**: In File Explorer, open `STOP_APP\` and double-click **`stop.bat`**
- **Linux**: Run `./STOP_APP/stop.sh` from the `DAG_Validator_Agent/` folder

This stops the Shiny app container. Ollama continues running in the background (no re-download needed on next start).

---

## Sample Data

A sample dataset is included in the **`Data/`** folder for testing:

- **`cd3cd28.xls`** Sachs et al. (2005) flow cytometry dataset (1000 observations, 11 signaling proteins: Raf, Mek, PLCg, PIP2, PIP3, Erk, Akt, PKA, PKC, p38, JNK)

Upload this file in the app to try out DAG validation with a well-known causal discovery benchmark.

---

## Using the App

1. **Upload your data** CSV or Excel file with numeric columns (or use the sample data from the `Data/` folder)
2. **Preprocessing** optionally apply log transform or z-score standardization
3. **Define a DAG** manually draw edges in the visual editor, upload an adjacency matrix, or let the LLM propose one
4. **Run CI tests** the app tests all conditional independencies implied by your DAG against the data
5. **Review results** plots show which CI statements hold and which are rejected
6. **Export a report** download a ZIP with a PDF report, the data, and a reproducible R script

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| macOS blocks `start.command` (ZIP download only) | Run `xattr -cr RUN_APP/ STOP_APP/` in Terminal. Avoid this by using `git clone` instead |
| Windows SmartScreen blocks `start.bat` (ZIP download only) | Click "More info" → "Run anyway". Avoid this by using `git clone` instead |
| Linux "Permission denied" | Run `chmod +x RUN_APP/start.sh STOP_APP/stop.sh` |
| "Docker is not installed" | Install Docker Desktop and restart |
| "Docker is not running" | Open Docker Desktop and wait for it to finish starting |
| "Ollama is not installed" | Install Ollama from [ollama.com/download](https://ollama.com/download) |
| LLM features are slow | Make sure Ollama is installed natively (not only inside Docker) |
| App doesn't open after 3 minutes | Run `docker compose logs -f` in a terminal to check progress |
| Browser shows connection error | Wait a moment and refresh, the app may still be starting |

---

## For Developers

To apply code changes, edit the files and double-click the start script again, it rebuilds automatically.

Key files:
- `app.R` main Shiny UI and server logic
- `R/` helper modules (CI testing, DAG utilities, LLM calls, plotting)
- `report_template.Rmd` PDF report template
- `docker-compose.yml` container config (Shiny app only; Ollama runs natively)

### Architecture

- **Shiny app** runs inside Docker (consistent R environment across platforms)
- **Ollama** runs natively on the host machine (uses GPU for fast LLM inference). If Ollama were installed inside Docker, it would be forced to use CPU, which is significantly slower.
- The Docker container connects to the host's Ollama via `host.docker.internal:11434`
