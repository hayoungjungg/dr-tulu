#!/bin/bash
#SBATCH --job-name=interactive_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:2
#SBATCH --time=1:00:00
#SBATCH --partition=all
#SBATCH --output=/n/fs/peas-lab/hayoung/dr-tulu/agent/logs/vllm_servers_%j.out
#SBATCH --error=/n/fs/peas-lab/hayoung/dr-tulu/agent/logs/vllm_servers_%j.err

# Set agent directory to absolute path
AGENT_DIR="/n/fs/peas-lab/hayoung/dr-tulu/agent"

# Change to the agent directory
cd "$AGENT_DIR" || {
    echo "Error: Cannot change to $AGENT_DIR"
    exit 1
}

# Set UV cache directory to project directory to avoid home directory space issues
export UV_CACHE_DIR="$AGENT_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$AGENT_DIR/logs"

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
VLLM_LOG1="$AGENT_DIR/logs/vllm_dr_tulu_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 uv run --extra vllm vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960 > "$VLLM_LOG1" 2>&1 &
VLLM_PID1=$!

echo "Starting Qwen3-8B on GPU 1 (port 30002)..."
VLLM_LOG2="$AGENT_DIR/logs/vllm_qwen_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 uv run --extra vllm vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960 > "$VLLM_LOG2" 2>&1 &
VLLM_PID2=$!

# Wait a bit for servers to start
echo "Waiting for servers to initialize..."
sleep 10

# Check if servers are running
if ps -p $VLLM_PID1 > /dev/null && ps -p $VLLM_PID2 > /dev/null; then
    echo "✓ Both VLLM servers are running"
    echo "  - DR-Tulu-8B: http://localhost:30001"
    echo "  - Qwen3-8B: http://localhost:30002"
    echo ""
    echo "=== Interactive shell ==="
    echo "Servers are running in the background. You can now use this shell."
    echo "To check server logs: tail -f logs/vllm_*.log"
    echo ""
    echo "To launch the chat interface, run:"
    echo "  uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --skip-checks"
    echo ""
    echo "To stop servers: Press Ctrl+C or exit this shell"
    echo ""
    
    # Keep the script running and provide an interactive shell
    exec bash
else
    echo "✗ Failed to start one or both VLLM servers"
    echo "Check logs/vllm_*.log for details"
    exit 1
fi

