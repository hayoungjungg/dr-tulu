#!/bin/bash
# Batch evaluation script for multiple tasks
# 
# This script runs the auto_search_sft workflow on multiple evaluation tasks
# and evaluates the results.
#
# Prerequisites:
#   - MCP server should be running (launched automatically by workflow)
#   - vLLM servers should be running (launched automatically by workflow)
#
# Usage:
#   bash scripts/auto_search.sh
#
# Configuration:
#   Edit the variables below to customize the evaluation

# Configuration
MAX_CONCURRENT=20
SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
SAVE_MODEL_NAME=auto_search_sft

mkdir -p $SAVE_FOLDER

for task in healthbench researchqa genetic_diseases_qa; do # for sqav2 and deep_research_bench, the scoring step needs additional conversion; see README for details
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10, browse_tool_name=jina" \
        --output $SAVE_FOLDER/$SAVE_MODEL_NAME/$task.jsonl

    python scripts/evaluate.py $task $SAVE_FOLDER/$SAVE_MODEL_NAME/$task.jsonl
done
