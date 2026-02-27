# Local Analysis Architecture

## Principle

**All data analysis runs locally. Agent decisions run on local Ollama.**

## What Runs Locally

### ✅ Local Analysis (No external data processing)

1. **Data Upload & Processing**
   - File reading (CSV, Excel)
   - Data validation
   - Variable extraction
   - **Location**: `upload_data()` endpoint

2. **CI Tests (Conditional Independence Tests)**
   - Runs via R subprocess calls
   - Uses `comets` and `dagitty` R packages
   - All test computations happen locally
   - **Location**: `test_dag_consistency()` method

3. **TRAM-DAG Model Fitting**
   - PyTorch model training
   - Loss computation
   - Model checkpointing
   - All happens on your local machine/GPU
   - **Location**: `fit_tramdag_model()` method

4. **Sampling & Causal Queries**
   - Intervention effects (ATE computation)
   - Counterfactual sampling
   - All sampling runs locally
   - **Location**: `sample_from_model()` method

5. **Report Generation**
   - Markdown report creation
   - Script generation
   - All file I/O is local
   - **Location**: `generate_report()` method

## What Uses the LLM (Agent Decisions Only)

### 🤖 LLM Calls (Agent Decisions)

1. **DAG Proposal** (`propose_dag_from_llm()`)
   - **Input**: Variable names (strings only)
   - **Output**: Proposed DAG structure
   - **Purpose**: Agent decision on initial causal structure
   - **No data sent**: Only variable names

2. **DAG Revision Suggestions** (`_step_4_propose_revisions()`)
   - **Input**: CI test results summary (rejected tests, p-values)
   - **Output**: Suggested DAG revisions
   - **Purpose**: Agent decision on how to revise DAG
   - **No raw data sent**: Only test result summaries

## Data Privacy

- ✅ **Your data never leaves your machine**
- ✅ **Only variable names and test summaries are sent to the LLM**
- ✅ **All model fitting, sampling, and analysis is 100% local**
- ✅ **The LLM is only used for "thinking" (agent decisions), not computation**

## Verification

To verify this architecture:

1. Check LLM calls:
   ```bash
   rg "llm_client\\.chat\\.completions\\.create" app/chatbot_server.py
   ```
   You'll find only decision-time calls, not data computation calls.

2. Check local analysis:
   - CI tests: Uses `r_python_bridge.py` → R subprocess
   - Model fitting: Uses `TramDagModel.fit()` → PyTorch (local)
   - Sampling: Uses `TramDagModel.sample()` → PyTorch (local)

3. Network monitoring:
   - In default mode (`LLM_PROVIDER=ollama`), LLM traffic stays local
   - No data uploads to external services for analysis

## Example Flow

```
User: "What is the effect of X on Y?"
  ↓
[LOCAL] Extract variables from uploaded data
  ↓
[LLM] Agent proposes DAG structure (variable names only)
  ↓
[LOCAL] Run CI tests on your data (R subprocess)
  ↓
[LLM] Agent interprets CI results and suggests revisions (test summaries only)
  ↓
[LOCAL] Fit TRAM-DAG model (PyTorch, your machine)
  ↓
[LOCAL] Compute intervention effects (sampling, your machine)
  ↓
[LOCAL] Generate report (file I/O, your machine)
```

## Summary

- **Agent (LLM)**: Decides what to do, suggests structures
- **Analysis (Local)**: Actually does the computation on your data
- **Result**: Privacy-preserving, fast, no data leakage
