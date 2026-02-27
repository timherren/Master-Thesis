# MCP Architecture Documentation

## Overview

This chatbot implements a **Model Context Protocol (MCP)** architecture where:

1. **Chat UI** provides natural language input
2. **LLM** uses function calling to select appropriate tools
3. **MCP Tools** execute locally (TRAM-DAG, R CI tests, etc.)
4. **Results** are returned to the user via chat UI

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Chat UI (Frontend)                    │
│  - Natural language input                               │
│  - User steers agent via conversation                   │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP/WebSocket
┌───────────────────────▼─────────────────────────────────┐
│              Agent Orchestrator (Backend)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM Function Calling                              │  │
│  │  - Analyzes user input                             │  │
│  │  - Selects appropriate MCP tools                   │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │  MCP Tool Registry                                 │  │
│  │  - propose_dag_from_llm                            │  │
│  │  - test_dag_consistency                            │  │
│  │  - fit_tramdag_model                               │  │
│  │  - sample_from_model                               │  │
│  │  - compute_ate                                     │  │
│  │  - generate_report                                 │  │
│  └──────────────────────┬────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌─────▼──────┐
│  TRAM-DAG    │  │  R CI Tests     │  │  File I/O  │
│  (Python)    │  │  (R/Subproc)    │  │  (JSON)    │
│  LOCAL       │  │  LOCAL          │  │  LOCAL     │
└──────────────┘  └─────────────────┘  └────────────┘
```

## Key Principles

### 1. Chat UI Steers Tool Selection

The user interacts via natural language in the chat UI. The LLM interprets the request and selects appropriate tools using function calling.

**Example:**
- User: "What is the effect of x1 on x3?"
- LLM selects: `compute_ate` tool
- Tool executes locally
- Results returned to chat UI

### 2. MCP Tool Registry

All tools are formally defined in `MCPToolRegistry` with:
- **Name**: Tool identifier
- **Description**: What the tool does (for LLM understanding)
- **Parameters**: Input schema (JSON Schema)
- **Returns**: Output description
- **Implementation**: Actual function in `MCPTools` class

### 3. Local Execution

**Critical**: All data analysis runs locally:
- TRAM-DAG model fitting (Python)
- CI tests (R subprocess)
- Sampling and ATE computation (Python)
- Report generation (Python)

**Only agent decisions use the LLM:**
- Tool selection (function calling)
- DAG proposal (LLM suggests structure)
- Result interpretation (LLM explains)

**No data is sent to external analysis services** - only variable names and summaries are used for LLM decisions.

## MCP Tools

### 1. `propose_dag_from_llm`
- **Purpose**: Agent decision - propose causal DAG structure
- **Uses LLM**: Yes (local Ollama)
- **Input**: Variable names, optional expert knowledge
- **Output**: DAG structure (adjacency matrix, edges)

### 2. `test_dag_consistency`
- **Purpose**: Test DAG against data using CI tests
- **Uses LLM**: No (runs locally via R)
- **Input**: DAG, data path, significance level
- **Output**: Consistency results, rejected CIs

### 3. `fit_tramdag_model`
- **Purpose**: Fit TRAM-DAG model to data
- **Uses LLM**: No (runs locally)
- **Input**: DAG, data path, training parameters
- **Output**: Fitted model, loss history

### 4. `sample_from_model`
- **Purpose**: Sample from fitted model (observational/interventional)
- **Uses LLM**: No (runs locally)
- **Input**: Experiment directory, sample size, interventions
- **Output**: Sampled values

### 5. `compute_ate`
- **Purpose**: Compute Average Treatment Effect
- **Uses LLM**: No (runs locally)
- **Input**: Experiment directory, treatment/outcome variables, values
- **Output**: ATE, means, standard deviations

### 6. `generate_report`
- **Purpose**: Generate PDF report with plots
- **Uses LLM**: No (runs locally)
- **Input**: Session ID, output path, report type
- **Output**: PDF report path

## Workflow Modes

The agent can operate in two modes:

### Mode 1: Workflow-Based (Current Default)
- Follows predefined steps: parse → propose → test → fit → query
- More structured, predictable
- Good for guided workflows

### Mode 2: MCP Tool-Based (New)
- LLM selects tools dynamically based on user input
- More flexible, conversational
- Good for exploratory analysis

**To enable MCP mode:**
```python
response = await orchestrator.process_message(
    user_message="What is the effect of x1 on x3?",
    use_mcp_tool_selection=True
)
```

## Example Interaction Flow

1. **User**: "I want to analyze the effect of treatment on outcome"
   - **LLM selects**: `propose_dag_from_llm`
   - **Tool executes**: Proposes DAG structure
   - **Response**: Shows proposed DAG

2. **User**: "Test this DAG against my data"
   - **LLM selects**: `test_dag_consistency`
   - **Tool executes**: Runs CI tests locally
   - **Response**: Shows test results

3. **User**: "Fit a model"
   - **LLM selects**: `fit_tramdag_model`
   - **Tool executes**: Trains TRAM-DAG model locally
   - **Response**: Shows training progress and results

4. **User**: "What is the ATE?"
   - **LLM selects**: `compute_ate`
   - **Tool executes**: Computes ATE locally
   - **Response**: Shows ATE with confidence intervals

5. **User**: "Generate a report"
   - **LLM selects**: `generate_report`
   - **Tool executes**: Creates PDF with plots locally
   - **Response**: Provides download link

## Implementation Details

### Tool Registry
```python
class MCPToolRegistry:
    TOOLS = [
        {
            "name": "tool_name",
            "description": "What it does",
            "parameters": {...},  # JSON Schema
            "returns": {...}
        },
        ...
    ]
    
    @staticmethod
    def get_tool_schemas_for_llm():
        # Convert to function-calling format
        ...
    
    @staticmethod
    def call_tool(tool_name, **kwargs):
        # Execute tool by name
        ...
```

### Agent Orchestrator
```python
class AgentOrchestrator:
    async def process_message(self, user_message, use_mcp_tool_selection=False):
        if use_mcp_tool_selection:
            # MCP mode: LLM selects tools
            return await self._process_with_mcp_tools(user_message)
        else:
            # Workflow mode: Follow predefined steps
            return await self._route_by_workflow_step(user_message)
```

## Benefits of MCP Architecture

1. **Flexibility**: User can ask questions in any order
2. **Transparency**: Clear tool definitions and execution
3. **Privacy**: All data analysis runs locally
4. **Extensibility**: Easy to add new tools
5. **Steerability**: Chat UI naturally steers agent behavior

## Future Enhancements

1. **Tool Chaining**: LLM can select multiple tools in sequence
2. **Conditional Execution**: Tools can depend on previous results
3. **Tool Validation**: Validate tool inputs before execution
4. **Tool Caching**: Cache tool results for efficiency
5. **Tool Monitoring**: Track tool usage and performance

---

**Key Takeaway**: The chat UI steers the agent, and the agent uses MCP tools to perform causal inference analysis. All computation runs locally, ensuring privacy and reproducibility.
