# Causal AI Agent Chatbot Architecture

## Overview

A conversational AI agent that guides users through causal inference workflows, replacing the Shiny UI with a ChatGPT Deep Research-style interface.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Web UI)                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Chat Interface (React/HTML)                      │  │
│  │  - Message history                                │  │
│  │  - Streaming responses                            │  │
│  │  - Code blocks, plots, tables                     │  │
│  │  - Step-by-step progress indicators                │  │
│  └───────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP/WebSocket
┌───────────────────────▼─────────────────────────────────┐
│              Agent Backend (Python FastAPI)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Agent Orchestrator                               │  │
│  │  - Workflow state management                      │  │
│  │  - Step coordination                             │  │
│  │  - Tool calling                                  │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM Interface (Ollama API)                      │  │
│  │  - DAG proposal                                  │  │
│  │  - Interpretation                                │  │
│  │  - Report generation                             │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  MCP Tools (Model Context Protocol)              │  │
│  │  - propose_dag()                                 │  │
│  │  - test_dag_consistency()                        │  │
│  │  - fit_tramdag_model()                           │  │
│  │  - sample_from_model()                           │  │
│  │  - compute_ate()                                 │  │
│  │  - generate_report()                             │  │
│  └───────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│  TRAM-DAG    │ │  R CI Tests  │ │  File I/O  │
│  (Python)    │ │  (R/Subproc) │ │  (JSON)    │
└──────────────┘ └──────────────┘ └────────────┘
```

## Workflow Steps

### Step 1: User Poses Causal Question
- **Input**: Natural language question
- **Agent Action**: Parse question, extract variables, identify query type
- **Output**: Structured query object

### Step 2: Agent Proposes Initial DAG
- **Input**: Variables from question + optional expert knowledge
- **Agent Action**: Call LLM to propose DAG structure
- **Tool**: `propose_dag_from_llm()`
- **Output**: Adjacency matrix + assumptions document

### Step 3: DAG Tested Against Data
- **Input**: Proposed DAG + data
- **Agent Action**: Run CI tests
- **Tool**: `test_dag_consistency()`
- **Output**: Test results, rejected CIs, interpretation

### Step 4: Agent Proposes Revisions
- **Input**: CI test results
- **Agent Action**: LLM analyzes failures, suggests fixes
- **Tool**: `suggest_dag_revisions()`
- **Output**: Revised DAG or adjustment strategy

### Step 5: Statistical Models Fitted
- **Input**: Final DAG + data
- **Agent Action**: Fit TRAM-DAG model
- **Tool**: `fit_tramdag_model()`
- **Output**: Fitted model, diagnostics, loss history

### Step 6: Answer Pearl's Level Questions
- **Input**: Query type (association/intervention/counterfactual)
- **Agent Action**: 
  - Association: Direct from data
  - Intervention: `compute_ate()` or `sample_from_model(do_interventions)`
  - Counterfactual: `sample_from_model(predefined_latents, do_interventions)`
- **Tool**: `answer_causal_query()`
- **Output**: Answer with confidence intervals, plots

### Step 7: Generate Reproducible Report
- **Input**: All results, assumptions, scripts
- **Agent Action**: Generate markdown/HTML report
- **Tool**: `generate_report()`
- **Output**: Complete report with code, plots, explanations

## Technology Stack

### Frontend
- **Framework**: React or vanilla HTML/JS
- **Styling**: Tailwind CSS (ChatGPT-like design)
- **Communication**: WebSocket or Server-Sent Events for streaming

### Backend
- **Framework**: FastAPI (Python)
- **LLM**: Ollama local (`qwen2.5:7b-instruct` by default)
- **MCP**: Model Context Protocol for tool calling
- **State**: SQLite or in-memory for session management

### Integration
- **TRAM-DAG**: Direct Python imports
- **R CI Tests**: Subprocess calls or rpy2
- **File Storage**: Local filesystem for experiments

## Key Components

### 1. Agent Orchestrator
Manages workflow state and coordinates steps:
- Session management
- State persistence
- Error handling
- Progress tracking

### 2. MCP Tools
Wrappers around existing functions:
- `propose_dag_from_llm(vars, expert_text) -> adj_matrix`
- `test_dag_consistency(dag, data, alpha) -> results`
- `fit_tramdag_model(dag, data, params) -> model`
- `sample_from_model(model, n_samples, do_interventions) -> samples`
- `compute_ate(model, X, Y, x_treated, x_control) -> ate`
- `answer_causal_query(model, query_type, query_params) -> answer`
- `generate_report(session_id) -> report_path`

### 3. LLM Prompts
Structured prompts for each step:
- DAG proposal prompt
- CI interpretation prompt
- Revision suggestion prompt
- Report generation prompt

### 4. UI Components
- Chat message bubbles
- Code blocks with syntax highlighting
- Interactive plots (Plotly)
- Tables (DataTables)
- Progress indicators
- File download buttons

## Implementation Plan

1. **Phase 1**: Basic chat interface + LLM integration
2. **Phase 2**: MCP tools implementation
3. **Phase 3**: Workflow orchestration
4. **Phase 4**: UI polish + streaming
5. **Phase 5**: Report generation
6. **Phase 6**: Testing + documentation
