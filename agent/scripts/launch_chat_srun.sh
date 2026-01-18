#!/bin/bash
# Quick interactive script to launch VLLM servers and run launch_chat.py
# Usage: srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 bash scripts/launch_chat_srun.sh [MODEL_NAME] [ADDITIONAL_FLAGS...]
# Example: srun ... bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B --enable-filtering
# Alternative (if resources not available): Use sbatch scripts/launch_chat_sbatch.sh instead

# Change to the agent directory (assuming script is run from project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$AGENT_DIR" || exit 1

# Set UV cache directory to project directory to avoid home directory space issues
export UV_CACHE_DIR="$AGENT_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables from .env file if it exists
if [ -f "$AGENT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$AGENT_DIR/.env"
    set +a
fi

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=== Cleaning up VLLM servers ==="
    pkill -f "vllm serve"
    wait
    echo "Cleanup complete"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Launch VLLM servers in background
echo "=== Launching VLLM servers ==="
echo "Starting DR-Tulu-8B on GPU 0 (port 30001)..."
CUDA_VISIBLE_DEVICES=0 uv run --extra vllm vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960 > logs/vllm_dr_tulu_$$.log 2>&1 &
VLLM_PID1=$!

echo "Starting Qwen3-8B on GPU 1 (port 30002)..."
CUDA_VISIBLE_DEVICES=1 uv run --extra vllm vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960 > logs/vllm_qwen_$$.log 2>&1 &
VLLM_PID2=$!

# Wait for servers to start
echo "Waiting for servers to initialize..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if both processes are still running
    if ! ps -p $VLLM_PID1 > /dev/null 2>&1 || ! ps -p $VLLM_PID2 > /dev/null 2>&1; then
        echo "✗ One or both VLLM servers failed to start"
        echo "Check logs/vllm_*.log for details"
        exit 1
    fi
    
    # Check if ports are listening (using timeout if nc is available, otherwise just check process)
    if command -v nc >/dev/null 2>&1; then
        if nc -z localhost 30001 2>/dev/null && nc -z localhost 30002 2>/dev/null; then
            echo "✓ Both VLLM servers are ready"
            echo "  - DR-Tulu-8B: http://localhost:30001"
            echo "  - Qwen3-8B: http://localhost:30002"
            break
        fi
    else
        # Fallback: just wait a bit and assume they're ready
        if [ $WAITED -ge 30 ]; then
            echo "✓ Assuming servers are ready (waited ${WAITED}s)"
            break
        fi
    fi
    
    sleep 5
    WAITED=$((WAITED + 5))
    
    # Print status every 30 seconds
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "⏳ Still waiting for servers to be ready (${WAITED}s)..."
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠ Timeout waiting for servers. They may still be initializing..."
fi

# Default model (can be overridden via command line argument)
MODEL="${1:-rl-research/DR-Tulu-8B}"

# Shift to get remaining arguments (flags like --enable-filtering, etc.)
shift

echo ""
echo "=== Launching interactive chat ==="
echo "Model: $MODEL"
echo ""

# Run launch_chat.py with skip-checks since we've already launched the servers
# Pass through all remaining arguments
uv run --extra vllm python scripts/launch_chat.py \
    --model "$MODEL" \
    --skip-checks \
    "$@"

# Cleanup will happen automatically via trap

