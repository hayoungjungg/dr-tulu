#!/bin/bash
# Helper script to set up UV cache directory and clean old cache
# Run this once: source scripts/setup_uv_cache.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set UV cache directory to project directory
export UV_CACHE_DIR="$AGENT_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

echo "✓ UV_CACHE_DIR set to: $UV_CACHE_DIR"
echo ""
echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "  export UV_CACHE_DIR=\"$AGENT_DIR/.uv-cache\""
echo ""
echo "To clean up old cache in home directory (optional):"
echo "  rm -rf ~/.cache/uv  # or /u/hj7206/.cache/uv"

