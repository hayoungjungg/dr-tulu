#!/bin/bash
#SBATCH --job-name=launch_chat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=0:30:00
#SBATCH --partition=all
#SBATCH --output=/n/fs/peas-lab/hayoung/dr-tulu/agent/logs/launch_chat_%j.out
#SBATCH --error=/n/fs/peas-lab/hayoung/dr-tulu/agent/logs/launch_chat_%j.err

# Set agent directory to absolute path
AGENT_DIR="/n/fs/peas-lab/hayoung/dr-tulu/agent"

# Change to the agent directory
cd "$AGENT_DIR" || {
    echo "Error: Cannot change to $AGENT_DIR"
    exit 1
}

# Create logs directory if it doesn't exist
mkdir -p "$AGENT_DIR/logs"

# Set UV cache directory to project directory to avoid home directory space issues
export UV_CACHE_DIR="$AGENT_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

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
echo "Working directory: $(pwd)"
echo "Agent directory: $AGENT_DIR"

# Ensure logs directory exists
mkdir -p "$AGENT_DIR/logs"

VLLM_LOG1="$AGENT_DIR/logs/vllm_dr_tulu_${SLURM_JOB_ID}.log"
VLLM_LOG2="$AGENT_DIR/logs/vllm_qwen_${SLURM_JOB_ID}.log"

echo "Starting DR-Tulu-8B on GPU 0 (port 30001)..."
echo "Log file: $VLLM_LOG1"
CUDA_VISIBLE_DEVICES=0 uv run --extra vllm vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960 > "$VLLM_LOG1" 2>&1 &
VLLM_PID1=$!

echo "Starting Qwen3-8B on GPU 1 (port 30002)..."
echo "Log file: $VLLM_LOG2"
CUDA_VISIBLE_DEVICES=1 uv run --extra vllm vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960 > "$VLLM_LOG2" 2>&1 &
VLLM_PID2=$!

# Wait for servers to start
echo "Waiting for servers to initialize..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if both processes are still running
    if ! ps -p $VLLM_PID1 > /dev/null 2>&1 || ! ps -p $VLLM_PID2 > /dev/null 2>&1; then
        echo "✗ One or both VLLM servers failed to start"
        echo "Check log files for details:"
        echo "  - $VLLM_LOG1"
        echo "  - $VLLM_LOG2"
        if [ -f "$VLLM_LOG1" ]; then
            echo "Last 20 lines of DR-Tulu log:"
            tail -20 "$VLLM_LOG1"
        fi
        if [ -f "$VLLM_LOG2" ]; then
            echo "Last 20 lines of Qwen log:"
            tail -20 "$VLLM_LOG2"
        fi
        exit 1
    fi
    
    # Check if ports are listening
    if nc -z localhost 30001 2>/dev/null && nc -z localhost 30002 2>/dev/null; then
        echo "✓ Both VLLM servers are ready"
        echo "  - DR-Tulu-8B: http://localhost:30001"
        echo "  - Qwen3-8B: http://localhost:30002"
        break
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

# Default model (can be overridden via command line)
MODEL="${1:-rl-research/DR-Tulu-8B}"

echo ""
echo "=== Launching interactive chat ==="
echo "Model: $MODEL"
echo ""

# NOTE: This script launches servers but chat won't be interactive with sbatch.
# For interactive chat, use: srun --pty ... bash scripts/launch_chat_srun.sh
# Or use: sbatch scripts/interactive_vllm_servers.sh, then SSH to the node and run chat manually.

echo "⚠ WARNING: sbatch runs non-interactively. The chat interface won't work here."
echo ""
echo "To interact with the chat, you have two options:"
echo ""
echo "Option 1: Use srun for interactive session:"
echo "  srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 bash scripts/launch_chat_srun.sh $MODEL"
echo ""
echo "Option 2: SSH to this node and run chat manually:"
echo "  1. Find node: squeue -u \$USER"
echo "  2. SSH to node: ssh <NODE_NAME>"
echo "  3. Run: cd $AGENT_DIR && uv run --extra vllm python scripts/launch_chat.py --model $MODEL --skip-checks"
echo ""
echo "Servers are running. Waiting 1 hour (or until cancelled)..."
echo "Press Ctrl+C to stop servers and exit."
echo ""

# Keep servers running but don't try to run chat (it won't work interactively)
wait

