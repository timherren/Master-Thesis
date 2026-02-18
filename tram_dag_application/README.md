# TRAM-DAG Causal Analysis Application

An interactive Shiny application for fitting TRAM-DAG models, performing interventional sampling, and computing Average Treatment Effects (ATE). The app includes a built-in LLM (via Ollama) that provides plain-language interpretations of model results.

**Demo:** [Watch the application walkthrough on YouTube (choose 4k resolution)](https://youtu.be/Vvbh9ZUrh-c)

---

## Prerequisites

Install these two applications (one-time setup):

1. **Docker Desktop**  [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. **Ollama** (optional)  [ollama.com/download](https://ollama.com/download)

| Platform | Docker | Ollama |
|----------|--------|--------|
| **Mac** | Download `.dmg`, drag to Applications, open once | Download `.dmg`, drag to Applications, runs automatically in background |
| **Windows** | Download installer, follow prompts, open once | Download installer, follow prompts, runs automatically as a service |
| **Linux** | [Install Docker Engine](https://docs.docker.com/engine/install/) + `sudo systemctl start docker` | `curl -fsSL https://ollama.com/install.sh \| sh` |

Make sure Docker is **running** before launching the app:
- **Mac / Windows**: Open Docker Desktop and wait for the whale icon to stop animating.
- **Linux**: Run `sudo systemctl start docker` (or enable it at boot with `sudo systemctl enable docker`).

Ollama runs in the background automatically after installation, you never need to open it manually. The app works fully without Ollama; AI text interpretations are just disabled.

---

## Quick Start

### Mac
1. Open the `START_APP` folder
2. Double-click **`start.command`**
3. Wait for the setup to complete (first launch takes 10–15 minutes)
4. Your browser will open automatically to `http://localhost:3838`

### Windows
1. Open the `START_APP` folder
2. Double-click **`start.bat`**
3. Wait for the setup to complete (first launch takes 10–15 minutes)
4. Your browser will open automatically to `http://localhost:3838`

### Linux
1. Open a terminal in this folder
2. Run: `chmod +x START_APP/start.sh && START_APP/start.sh`
3. Wait for the setup to complete (first launch takes 10–15 minutes)
4. Your browser will open automatically to `http://localhost:3838`

The start script automatically:
- Checks that Docker and Ollama are installed and running
- Starts Ollama if it is not already running
- Downloads the LLM model on first run (~2 GB, one-time)
- Builds and starts the Shiny app in Docker
- Opens your browser when ready

> **First run**: The R/Python environment builds inside Docker (~10–15 min) and the LLM model downloads (~2 GB). Both are cached, so subsequent starts take only a few seconds.

---

## Stopping the App

- **Mac**: Double-click **`STOP_APP/stop.command`**
- **Windows**: Double-click **`STOP_APP\stop.bat`**
- **Linux**: Run `STOP_APP/stop.sh`

This stops the Shiny app container. Ollama continues running in the background (no re-download needed on next start).

---

## Sample Data

A sample dataset is included in the **`Data/`** folder for testing:

- **`continous_3_vars_dgp_100k.csv`** — Simulated data from a linear DGP with 100,000 observations and 3 continuous variables (x1, x2, x3) connected by LinearShift edges: x1 → x2, x1 → x3, x2 → x3.

Upload this file in the app to try out TRAM-DAG model fitting and causal inference.

---

## Using the App

1. **Upload your data** — CSV file with numeric columns (or use the sample data from the `Data/` folder)
2. **Define the DAG** — Use the default DAG, upload an adjacency matrix, or draw one manually in the visual editor
3. **Fit the model** — Configure training parameters (epochs, learning rate, batch size) and click "Fit TRAM-DAG Model"
4. **Explore results** — View fit diagnostics (loss curves, parameter convergence)
5. **Sample & intervene** — Perform observational and interventional sampling (do-calculus)
6. **Compute ATE** — Select treatment/outcome variables and compute Average Treatment Effects
7. **Export** — All results are saved to the `output/` folder with reproducible R scripts

---

## Where Are My Results?

All experiment results are saved to the **`output/`** folder. Each experiment gets its own timestamped subfolder:

```
output/
└── TramDag_Experiment_20250217_143022/
    ├── configuration.json
    ├── min_max_scaling.json
    ├── fit_debug.log
    ├── scripts/
    │   ├── README.md
    │   └── reproduce_analysis.R
    ├── reproducible_package/
    │   ├── data.csv
    │   └── full_workflow.R
    ├── x1/
    │   ├── simple_intercepts_all_epochs.json
    │   ├── train_loss_hist.json
    │   └── val_loss_hist.json
    ├── x2/
    │   └── ...
    └── x3/
        └── ...
```

The experiment folder contains two subfolders for reproducing results:

- **`scripts/`** — Loads the already-fitted model from the experiment directory and reruns sampling and ATE computation. Use this to quickly explore different interventions or sample sizes without re-training. Requires the experiment folder and its model files to be present.

- **`reproducible_package/`** — A fully self-contained package that reproduces the entire analysis from scratch on any computer. Includes a copy of the dataset (`data.csv`) and a complete workflow script (`full_workflow.R`) that sets up the configuration, re-fits the model, and runs all analyses. Use this to share your work or reproduce results independently of the original experiment folder.

---

## LLM Interpretations

The app uses Ollama to provide plain-language interpretations of your model results and ATE computations. Ollama runs natively on the host (outside Docker) so it can leverage your GPU for faster inference. If Ollama were installed inside Docker, it would be forced to use CPU, which is significantly slower.

The start scripts handle everything automatically, just make sure Ollama is installed. The first launch pulls the model (~2 GB), which is cached for future runs.

### Interactive Chatbot

After fitting a model, you can ask the built-in chatbot (step 9 in the sidebar) questions about your results. The LLM is **not** a generic assistant — it has access to the actual values from your analysis and can provide data-backed answers. Specifically, each prompt includes:

- **Data summary** variable names, dimensions, per-variable mean/sd/min/max/median, and the full correlation matrix
- **DAG structure** all edges with their types, source/intermediate/sink node classification
- **Training configuration** epochs, learning rate, batch size
- **Loss history** per-variable training and validation loss trajectories (start, end, minimum, best epoch)
- **Learned parameters** final linear shift coefficients and intercept value ranges per variable
- **Negative log-likelihood (NLL)** overall model fit quality
- **Observational sampling** per-variable comparison of model-generated samples vs held-out test data (means and standard deviations)
- **ATE result** treatment/outcome variables, intervention values, computed ATE, mean outcomes under treatment and control
- **Interventional sampling** per-variable distribution shifts between treatment and control conditions

Because the LLM runs entirely offline on your machine via Ollama, your data never leaves your device, making it safe for sensitive or confidential datasets. The trade-off is that responses are slower than cloud-based models, especially without a GPU, since the model runs locally with limited compute resources.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Docker is not installed" | Install Docker Desktop and restart |
| "Docker is not running" | Open Docker Desktop and wait for it to finish starting |
| "Ollama is not installed" | Install Ollama from [ollama.com/download](https://ollama.com/download) (optional) |
| LLM interpretations are slow | Make sure Ollama is installed natively (not only inside Docker) |
| App doesn't open after 6 minutes | Run `docker compose logs -f` in a terminal to check progress |
| Browser shows connection error | Wait a moment and refresh — the app may still be starting |
| Port 3838 already in use | Stop other services on that port, or edit `docker-compose.yml` |
| Want to rebuild from scratch | Run `docker compose down && docker compose up --build` |

---

## For Developers

To apply code changes, edit the files and run the start script again — it rebuilds automatically.

Key files:
- `app.R` main Shiny UI and server logic
- `docker-compose.yml` container config (Shiny app only; Ollama runs natively)
- `Dockerfile` app image definition (R + Python/tramdag environment)

### Architecture

- **Shiny app** runs inside Docker (consistent R + Python environment across platforms)
- **Ollama** runs natively on the host machine (uses GPU for fast LLM inference)
- The Docker container connects to the host's Ollama via `host.docker.internal:11434`

### DAG Format

Adjacency matrix CSV with:
- Row and column names = variable names
- Edge codes: `0` (no edge), `ls` (LinearShift), `cs` (ComplexShift), `si` (SimpleIntercept), `ci` (ComplexIntercept)
- Must match variables in data
- Must be upper triangular (zeros below diagonal)
