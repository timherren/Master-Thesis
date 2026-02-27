# Implementation Guide: Causal AI Agent Chatbot

## Overview

This guide walks you through implementing and running the Causal AI Agent Chatbot that replaces the Shiny UI with a conversational interface.

## Prerequisites

1. **Python 3.9+** with pip
2. **R 4.0+** with required packages
3. **LLM backend** (local Ollama)
4. **TRAM-DAG** installed and working

## Installation

### 1. Install Python Dependencies

```bash
cd TramDag/example_notebooks/causal_ai_chatbot
pip install -r requirements_chatbot.txt
```

### 2. Install R Dependencies

```r
# In R
install.packages(c("comets", "dagitty", "igraph", "dplyr", "tibble"))
```

### 3. Set Up Environment Variables

Create a `.env` file in the `causal_ai_chatbot` directory:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

### 4. Verify TRAM-DAG Installation

```python
python -c "from tramdag import TramDagConfig, TramDagModel; print('TRAM-DAG OK')"
```

## Running the Chatbot

### Option 1: Direct Python

```bash
cd TramDag/example_notebooks/causal_ai_chatbot
python chatbot_server.py
```

The server will start on `http://localhost:8000`

### Option 2: Using Uvicorn

```bash
cd TramDag/example_notebooks/causal_ai_chatbot
uvicorn chatbot_server:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Production Mode

```bash
uvicorn chatbot_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Usage

### 1. Open the Chat Interface

Navigate to `http://localhost:8000` in your browser.

### 2. Start a Conversation

Example conversation flow:

**User**: "I want to understand the causal effect of treatment X on outcome Y. I have data with variables: age, treatment, outcome, confounder."

**Agent**: 
- Proposes initial DAG structure
- Tests DAG consistency
- Suggests revisions if needed
- Fits TRAM-DAG model
- Answers causal queries

### 3. Workflow Steps

The agent guides you through:

1. **Question Parsing**: Extracts variables and query type
2. **DAG Proposal**: LLM proposes initial causal structure
3. **Consistency Testing**: CI tests validate DAG against data
4. **Revision Suggestions**: Agent proposes fixes for rejected CIs
5. **Model Fitting**: TRAM-DAG model is fitted
6. **Query Answering**: Answers association/intervention/counterfactual questions
7. **Report Generation**: Creates reproducible report

## API Endpoints

### HTTP Endpoints

- `GET /` - Serve chat UI
- `POST /api/chat` - Send chat message
- `POST /api/upload_data` - Upload data file
- `GET /api/generate_report?session_id=<id>` - Generate report

### WebSocket Endpoint

- `WS /ws` - Real-time chat (preferred)

## Example Queries

### Association Query
"What is the correlation between X and Y?"

### Intervention Query
"What is the effect of do(X=1) on Y?"

### Counterfactual Query
"Given that X=0, what would Y have been if X=1?"

## Customization

### Modify LLM Prompts

Edit prompts in `chatbot_server.py`:
- `propose_dag_from_llm()` - DAG proposal prompt
- `_step_4_propose_revisions()` - Revision suggestion prompt

### Add Custom Tools

Extend `MCPTools` class with new methods:

```python
@staticmethod
def my_custom_tool(param1: str, param2: int) -> Dict:
    """Custom tool description"""
    # Implementation
    return {"result": "..."}
```

### Customize UI

Edit `chatbot_ui.html`:
- Styling (Tailwind CSS classes)
- Layout
- Additional features

## Troubleshooting

### LLM Backend Errors
- For local mode, ensure Ollama is running: `ollama serve`
- Pull the configured model: `ollama pull qwen2.5:7b-instruct`
- Confirm base URL/model in `.env` (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`)

### R Integration Errors
- Ensure R is in PATH
- Check R packages installed
- Verify `r_python_bridge.py` can find R scripts

### TRAM-DAG Errors
- Verify TRAM-DAG installation
- Check experiment directory permissions
- Ensure data format is correct

### WebSocket Connection Issues
- Check firewall settings
- Try HTTP fallback (automatic)
- Verify port 8000 is available

## Architecture Notes

### State Management

Sessions are stored in-memory (`sessions` dict). For production:
- Use Redis for distributed sessions
- Use PostgreSQL for persistence
- Implement session cleanup

### MCP Tools

Tools follow Model Context Protocol pattern:
- Clear input/output types
- Error handling
- Logging

### Streaming Responses

WebSocket enables:
- Real-time updates
- Progress indicators
- Typing indicators

## Next Steps

1. **Add Data Upload**: Implement file upload endpoint
2. **Improve Query Parsing**: Use LLM to better parse causal questions
3. **Add Visualizations**: Plotly integration for DAGs and results
4. **Session Persistence**: Save sessions to database
5. **Multi-user Support**: Authentication and user management
6. **Report Templates**: Customizable report formats

## Integration with Existing Code

The chatbot integrates with:
- `app/mcp_wrappers/dag_validator_wrapper.R` - CI testing wrapper
- `app/mcp_wrappers/tram_wrapper.py` - TRAM-DAG wrapper
- Installed `tramdag` package - TRAM-DAG models
- Existing Shiny apps (can run in parallel)

## Performance Considerations

- **LLM Calls**: Cache DAG proposals for similar questions
- **Model Fitting**: Run in background threads
- **CI Tests**: Can be slow for large DAGs - add progress updates
- **WebSocket**: Use connection pooling for multiple users

## Security

- **API Keys**: Never commit `.env` file
- **File Uploads**: Validate file types and sizes
- **Session IDs**: Use secure random generation
- **CORS**: Configure properly for production

## Testing

```bash
# Test server startup
python chatbot_server.py

# Test API endpoints
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test123"}'
```

## Support

For issues or questions:
1. Check logs in console
2. Verify all dependencies installed
3. Test individual components (TRAM-DAG, R scripts)
4. Review error messages carefully
