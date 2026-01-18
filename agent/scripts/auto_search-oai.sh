#!/bin/bash
# Batch evaluation script using OpenAI models (via auto_search_sft-oai.yaml)
#
# This script runs evaluation tasks using OpenAI models instead of local vLLM servers.
# The MCP server will be launched automatically by the workflow.
#
# Usage:
#   bash scripts/auto_search-oai.sh
#
# Configuration:
#   Edit the variables below to customize the evaluation

# Configuration
MAX_CONCURRENT=10
DATEUID=$(date +%Y%m%d)  # Auto-generate date UID
SAVE_FOLDER=eval_output/baselines-${DATEUID}-gpt4.1
MODEL=auto_search_sft

mkdir -p $SAVE_FOLDER

# Fast tasks
# TASKS="simpleqa 2wiki healthbench"
# TASKS="browsecomp bc_synthetic_depth_one_v2_verified bc_synthetic_varied_depth_o3_verified"
TASKS="dsqa"

# serper+crawl4ai+readerv2+max-tool-calls-10
for task in $TASKS; do
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples ablation \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config workflows/${MODEL}-oai.yaml \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10" \
        --output $SAVE_FOLDER/$MODEL/$task-ablation-reader-max-tool-calls-10.jsonl
done