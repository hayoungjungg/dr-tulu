# How to Use the Chat Scripts

## Quick Start

### Interactive Chat (Recommended)

**Use `srun` for true interactivity:**

```bash
srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 \
    bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B
```

**With filtering enabled:**
```bash
srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 \
    bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B --enable-filtering
```

If resources aren't immediately available, you can:
- Try again later
- Try the `all` partition instead: `--partition=all`
- Use the two-step process below

## Two-Step Process (When srun Resources Aren't Available)

**Step 1:** Launch servers and get an interactive shell:
```bash
sbatch scripts/interactive_vllm_servers.sh
```

**Step 2:** Once the job starts, find the node and connect:
```bash
# Check which node your job is running on
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N"

# SSH to that node (replace <NODE_NAME> with your actual node)
ssh <NODE_NAME>

# Once on the compute node, run the chat
cd /n/fs/peas-lab/hayoung/dr-tulu/agent
uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --skip-checks
```

## What Each Script Does

- **`launch_chat_srun.sh`** - Launches servers + chat interactively (use with `srun`)
  - Best option for interactive sessions
  - Automatically handles server startup and cleanup
  
- **`launch_chat_sbatch.sh`** - Launches servers only (non-interactive)
  - Chat won't work interactively with sbatch
  - Use for server-only launches, then SSH to node
  
- **`interactive_vllm_servers.sh`** - Launches servers only, gives you a shell
  - Use with `sbatch`, then SSH to node
  - Good when `srun` resources aren't available

- **`launch_chat.py`** - Main Python launcher
  - Can be run directly if servers are already running
  - Supports all configuration options

## Quick Reference

```bash
# Interactive (best option)
srun --pty --partition=interactive --gres=gpu:l40:2 --mem=64G --cpus-per-task=8 --time=1:00:00 \
    bash scripts/launch_chat_srun.sh rl-research/DR-Tulu-8B

# Queue servers, then SSH and run chat manually
sbatch scripts/interactive_vllm_servers.sh
# Then: squeue to find node, ssh to node, run chat command

# Direct Python launcher (if servers already running)
uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --skip-checks
```

## Advanced Options

### Filtering Options
```bash
# Enable filtering with default Cochrane titles
--enable-filtering

# Custom filter configuration
--enable-filtering \
--filter-source-title "Your source title here" \
--filter-publication-date "10 June 2025"
```

### Configuration
```bash
# Use custom config file
--config workflows/custom_config.yaml

# Override config values
--config-overrides "search_agent_max_tool_calls=15,use_browse_agent=true"
```

See `python scripts/launch_chat.py --help` for all available options.

