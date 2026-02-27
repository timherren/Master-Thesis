# Causal AI Agent Chatbot

A conversational AI agent that guides users through causal inference workflows, replacing the Shiny UI with a ChatGPT Deep Research-style interface.

## Features

- **AI-Powered DAG Proposal**: LLM suggests causal structures from variable names
- **DAG Consistency Testing**: Validates DAGs against data using CI tests
- **Interactive Revisions**: Agent suggests DAG improvements based on test results
- **TRAM-DAG Model Fitting**: Fits probabilistic causal models
- **Causal Query Answering**: Answers association, intervention, and counterfactual questions
- **Reproducible Reports**: Generates complete analysis reports with code

## Prerequisites

Install these tools first:

1. **Python 3.11+**
2. **Ollama** (for local LLM inference) — [ollama.com/download](https://ollama.com/download)
3. **Docker Desktop** (optional, for containerized run) — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)

---

## Download

```bash
git clone https://github.com/timherren/Master-Thesis.git
cd Master-Thesis/causal_ai_chatbot
```

For Mac/Linux launcher scripts, make them executable once:

```bash
chmod +x START_APP/start.command START_APP/start.sh STOP_APP/stop.command STOP_APP/stop.sh
```

---

## Quick Start (Docker, recommended)

Inside `causal_ai_chatbot/`:

### Mac
1. Open `START_APP/`
2. Double-click `start.command`
3. Wait for first build/model pull (can take several minutes)
4. Open `http://localhost:8000`

### Windows
1. Open `START_APP\`
2. Double-click `start.bat`
3. Wait for first build/model pull (can take several minutes)
4. Open `http://localhost:8000`

### Linux
```bash
./START_APP/start.sh
```
Then open `http://localhost:8000`.

The start script checks Docker/Ollama, starts services, and pulls default decision/interpretation models if missing.

---

## Quick Start (Local Python)

```bash
cd causal_ai_chatbot
pip install -r requirements_chatbot.txt
```

If `tramdag` is missing:

```bash
pip install tramdag
```

Create `.env` in `causal_ai_chatbot/`:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL_DECISION=llama3.2:1b
LLM_MODEL_INTERPRETATION=llama3.2:latest
```

Start:

```bash
bash start_chatbot.sh
```

---

## Stop the App

- Docker mode:
  - Mac: `STOP_APP/stop.command`
  - Windows: `STOP_APP\stop.bat`
  - Linux: `./STOP_APP/stop.sh`
- Local Python mode: stop the terminal process (`Ctrl+C`)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| macOS blocks `start.command` (ZIP download only) | Run `xattr -cr START_APP/ STOP_APP/` from Terminal inside `causal_ai_chatbot/` |
| Windows SmartScreen blocks `start.bat` | Click **More info** → **Run anyway** |
| Linux "Permission denied" | Run `chmod +x START_APP/start.sh STOP_APP/stop.sh` |
| "Docker is not installed" | Install Docker Desktop and restart |
| "Docker is not running" | Open Docker Desktop and wait until it is fully started |
| Ollama model not found (404) | Pull required models: `ollama pull llama3.2:1b` and `ollama pull llama3.2:latest` |
| Port 8000 already in use | Stop the process using port 8000, or change host port in `docker-compose.yml` |
| `ModuleNotFoundError: tramdag` in local run | Install with `pip install tramdag` in the active Python environment |
| App not reachable after startup | Check logs with `docker compose logs -f` (Docker) or terminal output (local) |
| Want a clean rebuild | Run `docker compose down && docker compose up --build` |

## Architecture

See [docs/chatbot_architecture.md](docs/chatbot_architecture.md) for detailed architecture.

### Architecture at a glance

The current codebase is organized to keep runtime logic, docs, tests, and scripts clearly separated.

- `chatbot_server.py` (root): compatibility entrypoint so `python chatbot_server.py` still works
- `app/`: actual runtime package
  - `app/chatbot_server.py`: FastAPI app, `AgentOrchestrator`, tool execution
  - `app/tool_registry.py`: tool schemas exposed to LLM function-calling
  - `app/runtime_context.py`: environment/bootstrap (local LLM client, MCP client, optional R bridge)
  - `app/mcp/` + `app/mcp_wrappers/`: local MCP execution + wrapper backends
  - `app/config.py`: canonical runtime paths
  - `app/runtime/`: generated runtime artifacts (`uploads`, `reports`, `temp_plots`, MCP cache)
- `docs/`: explanatory documentation
- `tests/`: automated tests
- `scripts/`: utility scripts (e.g., environment/server checks)

### Request flow (how the agent works)

1. User sends a message from the web UI.
2. `AgentOrchestrator` loads session state and checks pending guided inputs.
3. Deterministic workflow commands are handled first (guardrails).
4. Otherwise, the LLM receives context + tool schemas from `ToolRegistry`.
5. If the LLM selects a tool, the backend executes it locally (TRAM-DAG/R wrappers as needed).
6. State is updated, results are formatted, and the response is returned to the UI.

## Documentation

- [docs/README.md](docs/README.md) - Documentation index
- [docs/QUICK_START.md](docs/QUICK_START.md) - Quick start guide
- [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Full implementation guide
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues and fixes
- [docs/MCP_ARCHITECTURE.md](docs/MCP_ARCHITECTURE.md) - MCP integration details
- [docs/GUIDED_EXPERIENCE.md](docs/GUIDED_EXPERIENCE.md) - Guided flow UX notes
- [docs/LOCAL_ANALYSIS.md](docs/LOCAL_ANALYSIS.md) - Local analysis details

## Related Applications

- [Repository overview](../README.md)
- [DAG Validator Agent](../DAG_Validator_Agent)
- [TRAM-DAG Application](../tram_dag_application)

## Workflow

1. **User asks causal question** → Agent parses and extracts variables
2. **Agent proposes DAG** → LLM suggests causal structure
3. **DAG tested** → CI tests validate against data
4. **Revisions suggested** → Agent proposes fixes for failures
5. **Model fitted** → TRAM-DAG model trained
6. **Queries answered** → Association/intervention/counterfactual
7. **Report generated** → Complete reproducible analysis

## Example Conversation

```
User: I want to understand the effect of treatment on outcome, 
      controlling for age and gender.

Agent: I'll help you analyze this causal question. Let me propose 
       an initial DAG structure...

       Proposed DAG:
       - age → treatment
       - age → outcome
       - gender → treatment
       - gender → outcome
       - treatment → outcome

       Should I test this DAG against your data?

User: Yes, please test it.

Agent: Testing DAG consistency...
       
       The DAG appears consistent with the data.
       
       Should I proceed with fitting the model?

User: Yes, fit the model.

Agent: Fitting TRAM-DAG model... This may take a few minutes.
       
       Model fitted successfully.
       
       You can now ask causal questions.

User: What is the effect of treatment=1 vs treatment=0 on outcome?

Agent: Computing Average Treatment Effect...
       
       ATE = 2.34 (95% CI: [1.89, 2.79])
       
       This means that treatment increases outcome by 2.34 units 
       on average, compared to no treatment.
```

## Files

- `chatbot_server.py` - Compatibility entrypoint (`python chatbot_server.py`)
- `app/` - Runtime application code (server, UI, config, MCP wrappers)
- `docs/` - Canonical non-entrypoint documentation
- `tests/` - Automated test suite
- `scripts/` - Utility scripts (environment and server checks)
- `requirements_chatbot.txt` - Python dependencies
- `start_chatbot.sh` - Canonical startup script

## Integration

The chatbot integrates with:
- **TRAM-DAG** (Python) - Model fitting and sampling
- **R CI Tests** - DAG consistency checking
- **Ollama (local LLM)** - tool routing, DAG proposal, and interpretation

## Development

See [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for customization and extension details.

### Run Tests

Smoke test:

```bash
python -m pytest -q tests/test_smoke.py
```

Full test suite:

```bash
python -m pytest -q tests
```

Notes:
- Test fixtures expect sample data at `data/sampled_data_1000.csv`
- Some environments may fail on full pytest runs due to native dependency issues; use targeted tests first

## License

Same as parent project.
