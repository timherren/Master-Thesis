# Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'tramdag'

**Problem**: Python can't find the tramdag module.

**Solutions**:

#### Option A: Install TRAM-DAG Dependencies
The chatbot requires TRAM-DAG's dependencies. Install them:

```bash
# From the chatbot directory
cd causal_ai_chatbot

# Install chatbot requirements (includes TRAM-DAG dependencies)
pip install -r requirements_chatbot.txt
```

#### Option B: Install TRAM-DAG in Development Mode
If TRAM-DAG is not installed as a package, install it in development mode:

```bash
# From your local tramdag source checkout
cd /path/to/tramdag
pip install -e .
```

Then run the chatbot:
```bash
python chatbot_server.py
```

### 2. Missing Dependencies (matplotlib, torch, etc.)

**Problem**: TRAM-DAG dependencies are not installed.

**Solution**:
```bash
pip install torch matplotlib scipy scikit-learn joblib statsmodels
```

Or install directly:
```bash
pip install tramdag
```

### 3. R Integration Not Available

**Problem**: `RConsistencyChecker` import fails.

**Solution**: This is optional. The chatbot will work without R integration, but DAG consistency testing won't be available. To enable:

1. Ensure R is installed and in PATH
2. Install R packages:
   ```r
   install.packages(c("comets", "dagitty", "igraph", "dplyr", "tibble"))
   ```
3. (Optional) verify `app/r_integration/r_python_bridge.py` exists

### 4. LLM Backend Not Configured

**Problem**: LLM configuration missing, or Ollama is not reachable.

**Solution**: Create a `.env` file in the chatbot directory:

```bash
# In causal_ai_chatbot/.env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

Then ensure Ollama is running and model is available:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

### 5. Port 8000 Already in Use

**Problem**: `Address already in use` error

**Solution**: Use a different port:

```bash
uvicorn chatbot_server:app --host 0.0.0.0 --port 8001
```

Or change the port in `chatbot_server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 6. Import Path Issues

**Problem**: Paths are calculated incorrectly

**Solution**: Run the test script to diagnose:

```bash
python test_imports.py
```

This will show you:
- Where Python is looking for tramdag
- Whether paths are calculated correctly
- What's missing

### 7. Data Upload Fails

**Problem**: File upload doesn't work

**Solution**: 
- Check file format (CSV, Excel)
- Ensure file is not too large
- Check `uploads/` directory permissions
- Verify file has proper headers

### 8. Model Fitting Takes Too Long

**Problem**: Model fitting is slow

**Solution**:
- Reduce epochs in the conversation
- Use smaller batch sizes
- Run on GPU if available (set device="cuda")
- Consider running fitting in background thread

## Testing Your Setup

Run the test script to verify everything works:

```bash
python test_imports.py
```

Expected output:
- TRAM-DAG import successful
- R Integration import successful (or warning if optional)

## Getting Help

If issues persist:

1. Check the error message carefully
2. Run `python test_imports.py` to diagnose
3. Verify all dependencies are installed
4. Check that paths are correct for your system
5. Ensure you're using the correct Python environment

## Quick Fixes

**All imports fail**:
```bash
# Install everything
pip install -r requirements_chatbot.txt
```

**TRAM-DAG not found**:
```bash
pip install tramdag
```

**R integration fails**:
- This is optional - chatbot works without it
- DAG consistency testing won't be available
- Other features will still work
