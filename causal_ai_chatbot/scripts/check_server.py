#!/usr/bin/env python3
"""
Quick check script to verify the server can start and identify issues
"""

import sys
from pathlib import Path

# Resolve chatbot root and add it to import path
CHATBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT_ROOT))

print("=" * 60)
print("Checking Chatbot Server Setup")
print("=" * 60)
print()

# Check imports
print("1. Checking imports...")
try:
    from chatbot_server import app, sessions, AgentOrchestrator
    print("   ✓ Server imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check LLM configuration
print("\n2. Checking LLM configuration...")
import os
from dotenv import load_dotenv
load_dotenv()
provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if provider != "ollama":
    print("   ✗ Unsupported LLM_PROVIDER. Use: LLM_PROVIDER=ollama")
else:
    print(
        "   ✓ Local Ollama configured "
        f"(decision_model={os.getenv('LLM_MODEL_DECISION') or os.getenv('LLM_MODEL', 'qwen2.5:3b-instruct')}, "
        f"interpretation_model={os.getenv('LLM_MODEL_INTERPRETATION') or os.getenv('LLM_MODEL') or os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}, "
        f"base={os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434/v1')})"
    )

# Check TRAM-DAG
print("\n3. Checking TRAM-DAG...")
try:
    from tramdag import TramDagConfig, TramDagModel, TramDagDataset
    print("   ✓ TRAM-DAG imports successful")
except Exception as e:
    print(f"   ⚠️  TRAM-DAG import failed: {e}")
    print("      Install with: pip install tramdag")

# Check R integration
print("\n4. Checking R integration...")
try:
    application_dir = CHATBOT_ROOT / "app" / "r_integration"
    if application_dir.exists():
        sys.path.insert(0, str(application_dir))
        from r_python_bridge import RConsistencyChecker
        print("   ✓ R integration available")
    else:
        print(f"   ⚠️  R integration directory not found at {application_dir}")
except Exception as e:
    print(f"   ⚠️  R integration not available: {e}")

# Check directories
print("\n5. Checking directories...")
dirs = [
    "app/runtime/uploads",
    "app/runtime/reports",
    "app/runtime/temp_plots",
    "app/runtime/mcp_artifacts_cache",
]
for d in dirs:
    dir_path = CHATBOT_ROOT / d
    if dir_path.exists():
        print(f"   ✓ {d}/ exists")
    else:
        print(f"   ⚠️  {d}/ does not exist (will be created on first use)")

# Check FastAPI app
print("\n6. Checking FastAPI app...")
try:
    routes = [r.path for r in app.routes]
    print(f"   ✓ FastAPI app initialized with {len(routes)} routes")
    print(f"   Routes: {', '.join(routes[:5])}...")
except Exception as e:
    print(f"   ✗ FastAPI app check failed: {e}")

print("\n" + "=" * 60)
print("Check Complete!")
print("=" * 60)
print("\nTo start the server, run:")
print("  python chatbot_server.py")
print("\nOr use the startup script:")
print("  bash start_chatbot.sh")
