# Quick Start: Causal AI Agent Chatbot

## What You Have

A complete conversational AI agent that replaces the Shiny UI with a ChatGPT-like interface for causal inference workflows.

## Files Created

1. **`chatbot_server.py`** - FastAPI backend with:
   - Agent orchestrator for workflow management
   - MCP tools (DAG proposal, CI testing, model fitting, sampling)
   - WebSocket and HTTP endpoints
   - Session management

2. **`chatbot_ui.html`** - Frontend chat interface with:
   - ChatGPT-like UI design
   - Real-time messaging (WebSocket)
   - File upload support
   - Code block rendering
   - Progress indicators

3. **`chatbot_architecture.md`** - Complete architecture documentation

4. **`IMPLEMENTATION_GUIDE.md`** - Detailed setup and usage guide

5. **`requirements_chatbot.txt`** - Python dependencies

6. **`start_chatbot.sh`** - Startup script

## Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
cd causal_ai_chatbot

# Install chatbot dependencies (includes TRAM-DAG dependencies)
pip install -r requirements_chatbot.txt
```

**Note**: If you get `ModuleNotFoundError: No module named 'tramdag'`, install it in the same environment (for example: `pip install tramdag`).

### 2. Set Environment Variables

Create `.env` file in this directory:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

### 3. Run

```bash
# Option 1: Use startup script
./start_chatbot.sh

# Option 2: Direct Python
python chatbot_server.py

# Option 3: Uvicorn (with auto-reload)
uvicorn chatbot_server:app --reload
```

Then open: **http://localhost:8000**

## Example Workflow

1. **Upload Data**: Click "📁 Upload Data" button, select CSV file
2. **Ask Question**: "What is the effect of treatment on outcome?"
3. **Agent Proposes DAG**: LLM suggests causal structure
4. **Agent Tests DAG**: CI tests validate against data
5. **Agent Suggests Revisions**: If needed, proposes fixes
6. **Agent Fits Model**: TRAM-DAG model training
7. **Ask Causal Queries**: Get answers to association/intervention/counterfactual questions
8. **Download Report**: Click "Download Report" for reproducible analysis

## Key Features

✅ **LLM-Powered DAG Proposal** - Uses local Ollama for fully offline decision support  
✅ **DAG Consistency Testing** - Validates DAGs using R CI tests  
✅ **Interactive Revisions** - Agent suggests improvements  
✅ **TRAM-DAG Integration** - Full model fitting and sampling  
✅ **Causal Query Answering** - Association, intervention, counterfactual  
✅ **Reproducible Reports** - Complete analysis with code  

## Architecture Highlights

- **Frontend**: HTML/JS with Tailwind CSS (ChatGPT-like design)
- **Backend**: FastAPI with WebSocket support
- **LLM**: Ollama local (`qwen2.5:7b-instruct` default)
- **Integration**: TRAM-DAG (Python) + R CI Tests (subprocess)
- **State**: In-memory sessions (upgrade to Redis/DB for production)

## Workflow Steps

The agent orchestrates these steps automatically:

1. **Parse Question** → Extract variables and query type
2. **Propose DAG** → LLM suggests causal structure  
3. **Test DAG** → CI tests validate against data
4. **Suggest Revisions** → Agent proposes fixes
5. **Fit Model** → TRAM-DAG model training
6. **Answer Queries** → Causal inference results
7. **Generate Report** → Reproducible analysis document

## Customization

### Change LLM Model

Edit `.env`:
```
# Local Ollama model
OLLAMA_MODEL=qwen2.5:14b-instruct

```

### Add Custom Tools

Extend `MCPTools` class in `chatbot_server.py`:
```python
@staticmethod
def my_custom_tool(param: str) -> Dict:
    """Custom tool"""
    return {"result": "..."}
```

### Customize UI

Edit `chatbot_ui.html` - uses Tailwind CSS for easy styling.

## Troubleshooting

**LLM Connectivity Errors**: Check Ollama is running (`ollama serve`) and model is pulled (`ollama pull qwen2.5:7b-instruct`)  
**R Integration Errors**: Ensure R packages installed (`comets`, `dagitty`)  
**TRAM-DAG Errors**: Verify installation and data format  
**WebSocket Issues**: Falls back to HTTP automatically  

## Next Steps

1. Test with your data
2. Customize prompts for your domain
3. Add visualizations (Plotly integration)
4. Deploy to production (add authentication, database)

## Integration with Existing Code

Works alongside existing Shiny apps:
- Uses same TRAM-DAG models
- Uses same R CI test functions
- Can run in parallel with Shiny apps

## Support

See `IMPLEMENTATION_GUIDE.md` for:
- Detailed setup instructions
- API documentation
- Customization guide
- Performance optimization

---

**Ready to use!** Start the server and begin your causal inference conversation. 🚀
