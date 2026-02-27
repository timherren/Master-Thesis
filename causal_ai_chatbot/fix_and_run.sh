#!/bin/bash
# Compatibility wrapper for the legacy script location.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec bash "$SCRIPT_DIR/scripts/fix_and_run.sh"

