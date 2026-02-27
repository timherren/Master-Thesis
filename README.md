# Master Thesis: Applications for Causal Analysis

This repository contains applications developed as part of my Master's thesis on causal inference.

---

## Applications

| App | Description |
|-----|-------------|
| [DAG Validator Agent](DAG_Validator_Agent) | Interactive Shiny app for validating causal DAGs using conditional independence testing, with local LLM support via Ollama. |
| [TRAM-DAG Application](tram_dag_application) | Docker-based Shiny app for fitting TRAM-DAG causal models, computing Average Treatment Effects (ATE), and generating reproducible analysis reports. Includes an offline LLM chatbot (Ollama) for interpreting results. |
| [Causal AI Chatbot](causal_ai_chatbot) | FastAPI + web UI chatbot that orchestrates DAG proposal/testing, TRAM model fitting, sampling, and ATE workflows through guided conversational flows. |

---

## Getting Started

1. Open the app folder you want to run.
2. Follow that folder's `README.md` setup instructions.
3. Use each app's provided start scripts to launch locally.

---

See each application's folder for setup instructions, architecture details, and troubleshooting notes.
