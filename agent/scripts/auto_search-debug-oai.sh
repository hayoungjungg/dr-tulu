#!/bin/bash
# Debug evaluation script using OpenAI models (small sample size for testing)
#
# This script runs a small number of examples (2 per task) for quick testing/debugging.
# Uses OpenAI models via auto_search_sft-oai.yaml config.
#
# Usage:
#   bash scripts/auto_search-debug-oai.sh
#
# Configuration:
#   Edit the variables below to customize the evaluation

# Configuration
MAX_CONCURRENT=10
DATEUID=$(date +%Y%m%d)  # Auto-generate date UID
SAVE_FOLDER=eval_output/baselines-${DATEUID}-gpt4.1
MODEL=auto_search_sft

mkdir -p $SAVE_FOLDER

TASKS="deep_research_bench healthbench sqav2 genetic_diseases_qa 2wiki webwalker simpleqa browsecomp"

# serper+crawl4ai+readerv2+max-tool-calls-10
for task in $TASKS; do
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples 2 \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config workflows/${MODEL}-oai.yaml \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10" \
        --output $SAVE_FOLDER/$MODEL/$task-5-samples-reader-max-tool-calls-10.jsonl
    
    python scripts/evaluate.py $task $SAVE_FOLDER/$MODEL/$task-5-samples-reader-max-tool-calls-10.jsonl
done