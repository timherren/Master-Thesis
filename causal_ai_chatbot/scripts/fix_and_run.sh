#!/bin/bash
# Legacy compatibility script. Use start_chatbot.sh as the canonical entrypoint.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHATBOT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$CHATBOT_ROOT"

echo "fix_and_run.sh is deprecated. Redirecting to start_chatbot.sh..."
exec bash "$CHATBOT_ROOT/start_chatbot.sh"
