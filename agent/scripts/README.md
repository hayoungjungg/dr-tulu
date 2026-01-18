# Scripts Directory

This directory contains utility scripts for running, evaluating, and interacting with the DR-Tulu agent system.

## Quick Start

### Interactive Chat (Recommended)

**For true interactivity, use `srun`:**
```bash
srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 \
    bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B
```

See [HOW_TO_USE.md](HOW_TO_USE.md) for detailed instructions.

## Scripts Overview

### Interactive Scripts

- **`launch_chat.py`** - Main interactive chat launcher
  - Automatically launches MCP server and vLLM servers
  - Provides interactive chat interface
  - Supports filtering options and custom configurations
  
- **`launch_chat_srun.sh`** - Interactive wrapper for `srun`
  - Launches servers and chat in one command
  - Best for interactive sessions
  
- **`launch_chat_sbatch.sh`** - Non-interactive batch launcher
  - Launches servers but chat won't be interactive
  - Use for server-only launches, then SSH to node
  
- **`interactive_vllm_servers.sh`** - Launch servers only, get interactive shell
  - Use with `sbatch`, then SSH to node to run chat manually
  - Good when `srun` resources aren't available

- **`interactive_auto_search.py`** - Interactive CLI using auto_search workflow
  - Provides chat interface using the same search/answer agents as evaluation
  - Useful for testing the full workflow interactively

### Evaluation Scripts

- **`auto_search.sh`** - Batch evaluation on multiple tasks
  - Runs evaluation on healthbench, researchqa, genetic_diseases_qa
  - Uses local vLLM servers (launched automatically)
  - Generates and evaluates results

- **`auto_search-oai.sh`** - Batch evaluation using OpenAI models
  - Same as `auto_search.sh` but uses OpenAI API instead of local vLLM
  - Useful when GPU resources aren't available
  - Requires OpenAI API key in environment

- **`auto_search-debug-oai.sh`** - Debug version with small sample size
  - Runs only 2 examples per task for quick testing
  - Uses OpenAI models
  - Useful for debugging configuration issues

- **`evaluate.py`** - Standalone evaluation script
  - Evaluates generated JSONL result files
  - Auto-detects task type and runs appropriate evaluation
  - Can be run independently after generation

### Utility Scripts

- **`setup_uv_cache.sh`** - Sets up UV cache directory
  - Configures UV to use project directory instead of home
  - Helps avoid home directory space issues
  - Run once: `source scripts/setup_uv_cache.sh`

## Configuration

### Environment Variables

- `UV_CACHE_DIR` - Directory for UV package cache (default: `$AGENT_DIR/.uv-cache`)
- `MCP_TRANSPORT_PORT` - MCP server port (default: 8000)
- `MCP_TRANSPORT_HOST` - MCP server host (default: localhost)
- `OPENAI_API_KEY` - Required for OpenAI-based scripts

### Common Options

Most scripts support:
- `--enable-filtering` - Enable Cochrane result filtering
- `--filter-title-list-file` - Path to JSON file with filter titles
- `--filter-source-title` - Source title for content filtering
- `--filter-publication-date` - Publication date cutoff
- `--config` - Path to workflow configuration YAML

## Examples

### Run Interactive Chat
```bash
# Option 1: Direct srun (best)
srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 \
    bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B --enable-filtering

# Option 2: Two-step process
sbatch scripts/interactive_vllm_servers.sh
# Then: squeue to find node, ssh to node, run:
cd /n/fs/peas-lab/hayoung/dr-tulu/agent
uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --skip-checks
```

### Run Batch Evaluation
```bash
# Local vLLM servers
bash scripts/auto_search.sh

# OpenAI models
bash scripts/auto_search-oai.sh

# Debug with small samples
bash scripts/auto_search-debug-oai.sh
```

### Evaluate Results
```bash
python scripts/evaluate.py healthbench eval_output/auto_search_sft/healthbench.jsonl
```

## Troubleshooting

### GPU Memory Issues
- Reduce `gpu_memory_utilization` in `dr_agent/utils.py` (default: 0.80)
- Check for other processes using GPU: `nvidia-smi`
- Use smaller models or reduce `max_model_len`

### MCP Server Issues
- Check if port 8000 is available: `lsof -i :8000`
- Check logs: `tail -f logs/mcp_actions_*.log`
- Verify MCP server starts: `python -m dr_agent.mcp_backend.main --port 8000`

### vLLM Server Issues
- Check logs: `tail -f logs/vllm_*.log`
- Verify ports are listening: `nc -z localhost 30001 && nc -z localhost 30002`
- Check GPU availability: `nvidia-smi`

## File Structure

```
scripts/
├── README.md                    # This file
├── HOW_TO_USE.md               # Detailed usage guide
├── launch_chat.py              # Main chat launcher
├── launch_chat_srun.sh         # Interactive wrapper
├── launch_chat_sbatch.sh       # Batch launcher
├── interactive_vllm_servers.sh # Server-only launcher
├── interactive_auto_search.py  # Interactive workflow CLI
├── auto_search.sh              # Batch evaluation (local)
├── auto_search-oai.sh          # Batch evaluation (OpenAI)
├── auto_search-debug-oai.sh    # Debug evaluation
├── evaluate.py                 # Standalone evaluator
└── setup_uv_cache.sh           # UV cache setup
```

## Notes

- All scripts assume they're run from the agent directory root
- Logs are written to `logs/` directory
- Output files are written to `eval_output/` directory
- Scripts automatically handle server launching and cleanup

