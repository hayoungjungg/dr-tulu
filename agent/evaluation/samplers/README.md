# Samplers Module

This module provides the base classes and implementations for creating samplers that can be used with the evaluation benchmarks in this repository. Samplers are responsible for generating responses to prompts and are the core component that interfaces with language models or cached responses.

## Quick Start

Use the unified evaluation script for consistent evaluation across all benchmarks:

```bash
# Run HealthBench with GPT models
python run_eval.py healthbench --run_mode gpt_completions --subset_name consensus

# Run BrowseComp with cached completions  
python run_eval.py browsecomp --run_mode cached_completions --cached_completions_path outputs/responses.json

# Run ResearchQA with custom models
python run_eval.py researchqa --run_mode gpt_completions --model gpt-4o --grader_model gpt-4-turbo --num_examples 10

# Run with MCP agents (requires rl-rag-mcp)
python run_eval.py researchqa --run_mode mcp_agents --agent_type research --num_examples 10
```

## Overview

The samplers module supports three evaluation benchmarks:
- **HealthBench**: Medical question answering evaluation
- **BrowseComp**: Web browsing and comprehension evaluation  
- **ResearchQA**: Research question answering evaluation

### Available Samplers

1. **ChatCompletionSampler**: OpenAI GPT models with chat completion API
2. **CachedCompletionSampler**: Pre-generated responses from files
3. **RLRAGMCPSampler**: RL-RAG-MCP agents with web search and tool capabilities
4. **Custom samplers**: Create your own by inheriting from `SamplerBase`

## Standardized Input/Output Formats

### Input Format (Standardized)

All benchmarks use a consistent prompt format for evaluation:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "Your question or task here..."
    }
  ],
  "prompt_id": "unique_identifier",
  "answer": "Expected answer or rubric",
  "dataset": "benchmark_name"
}
```

### Output Format (Cached Completions)

When providing cached completions, use this standardized format:

```json
{
  "prompt_id_1": {
    "answer": "This is the model's answer to question 1."
  },
  "prompt_id_2": {
    "answer": "This is the model's answer to question 2."
  }
}
```

### Evaluation Results Format

All evaluations return results in this format:

```json
{
  "score": 0.85,
  "metrics": {
    "accuracy": 0.85,
    "num_samples": 100,
    "additional_metric": 0.75
  }
}
```

## Core Types

### SamplerResponse

A `SamplerResponse` contains the output from a sampler call:

```python
@dataclass
class SamplerResponse:
    response_text: str                          # The generated text response
    actual_queried_message_list: MessageList   # The actual messages sent to the model
    response_metadata: dict[str, Any]           # Additional metadata (e.g., usage stats)
```

### MessageList

A `MessageList` is a list of message dictionaries with `role` and `content` keys:

```python
MessageList = list[dict[str, Any]]

# Example:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
```

## Creating Custom Samplers

### Base Class: SamplerBase

All samplers must inherit from `SamplerBase` and implement the `__call__` method:

```python
from samplers._types import SamplerBase, SamplerResponse, MessageList

class SamplerBase:
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        raise NotImplementedError
```

### Required Methods

#### 1. `__call__(self, message_list: MessageList) -> SamplerResponse`

**Required for all benchmarks**. This is the main interface method that:
- Takes a list of messages as input
- Returns a `SamplerResponse` with the generated text and metadata
- Must handle any preprocessing of the message list
- Should include error handling and retry logic if needed

#### 2. `_pack_message(self, role: str, content: Any) -> dict` (Optional but Recommended)

Helper method to create properly formatted message dictionaries:

```python
def _pack_message(self, role: str, content: Any) -> dict:
    return {"role": str(role), "content": content}
```

## Example Implementations

### 1. ChatCompletionSampler (GPT-4 Example)

This is the primary example of a production-ready sampler using OpenAI's API:

```python
import time
import os
from typing import Any
import openai
from openai import OpenAI, AzureOpenAI
from samplers._types import MessageList, SamplerBase, SamplerResponse

class ChatCompletionSampler(SamplerBase):
    """Sample from OpenAI's chat completion API"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        # Initialize OpenAI client (supports both OpenAI and Azure)
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            self.client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            self.client = OpenAI()
            
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any) -> dict:
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Add system message if provided
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
            
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                    
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
                
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential backoff
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec", e)
                time.sleep(exception_backoff)
                trial += 1
```

### 2. CachedCompletionSampler (Cached Responses Example)

This sampler loads pre-generated responses from files, useful for evaluation reproducibility:

```python
import json
from typing import Any
from samplers._types import MessageList, SamplerBase, SamplerResponse

class CachedCompletionSampler(SamplerBase):
    """Sample from cached completions"""
    
    def __init__(self, cached_completions_path: str, test_data_path: str = None):
        self.cached_completions = self._get_cached_completions_dict(cached_completions_path)
        
        # Auto-derive question-to-id mapping
        if test_data_path:
            self.question_to_id_dict = self._get_question_to_id_dict(test_data_path)
        else:
            self.question_to_id_dict = self._get_question_to_id_dict_from_cache()

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        cached_completion = self._retrieve_cached_completion(message_list)
        return SamplerResponse(
            response_text=cached_completion,
            response_metadata={"usage": None},
            actual_queried_message_list=message_list,
        )
```

### 3. RLRAGMCPSampler (MCP Agents)

This sampler integrates with rl-rag-mcp agents that can use tools like web search:

```python
from samplers.sampler.rl_rag_mcp_sampler import create_mcp_sampler_with_websearch

# Create MCP sampler with research agent
sampler = create_mcp_sampler_with_websearch(
    model_name="gpt-4o",
    temperature=0.1,
    max_tokens=6000,
    agent_type="research"  # or "simple", "client"
)

# Use in evaluation
response = sampler(messages)
print(response.response_text)
print(response.response_metadata["tool_calls"])  # See tool calls made
```

**Agent Types:**
- `research`: Optimized for academic questions with comprehensive search
- `simple`: General-purpose agent for basic questions
- `client`: Direct tool access without agent wrapper

**Features:**
- Web search capabilities
- Multi-step reasoning
- Tool call tracking
- Async execution with sync interface

### 4. Custom Model Sampler Template

Here's a template for creating your own sampler:

```python
from samplers._types import MessageList, SamplerBase, SamplerResponse

class CustomModelSampler(SamplerBase):
    """Template for implementing a custom model sampler"""
    
    def __init__(self, model_path: str, **model_kwargs):
        # Initialize your model here
        self.model = load_your_model(model_path, **model_kwargs)
        
    def _pack_message(self, role: str, content: Any) -> dict:
        return {"role": str(role), "content": content}
        
    def _preprocess_messages(self, message_list: MessageList) -> str:
        """Convert message list to your model's expected format"""
        # Example: concatenate messages with special tokens
        prompt = ""
        for message in message_list:
            if message["role"] == "system":
                prompt += f"<|system|>{message['content']}<|end|>\n"
            elif message["role"] == "user":
                prompt += f"<|user|>{message['content']}<|end|>\n"
            elif message["role"] == "assistant":
                prompt += f"<|assistant|>{message['content']}<|end|>\n"
        prompt += "<|assistant|>"  # prompt for response
        return prompt
    
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Preprocess the input
        formatted_prompt = self._preprocess_messages(message_list)
        
        try:
            # Generate response using your model
            response_text = self.model.generate(formatted_prompt)
            
            return SamplerResponse(
                response_text=response_text,
                response_metadata={
                    "model": self.model.name,
                    "prompt_length": len(formatted_prompt)
                },
                actual_queried_message_list=message_list,
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return SamplerResponse(
                response_text="Error generating response.",
                response_metadata={"error": str(e)},
                actual_queried_message_list=message_list,
            )
```

## Unified Evaluation Interface

Use the `run_eval.py` script for consistent evaluation across all benchmarks:

### Command Line Interface

```bash
python run_eval.py <benchmark> [options]
```

**Common Arguments:**
- `benchmark`: Choose from `healthbench`, `browsecomp`, `simpleqa`, `researchqa`
- `--run_mode`: `gpt_completions`, `mcp_agents`, or `cached_completions`
- `--num_examples`: Number of examples to evaluate (default: all)
- `--eval_save_dir`: Output directory (default: `eval_results`)
- `--cached_completions_path`: Path to cached responses JSON file

**Model Configuration:**
- `--model`: Model name for evaluation/generation (default: `gpt-4-turbo`)
- `--grader_model`: Model name for grading/judging responses (default: `gpt-4-turbo`)
- `--temperature`: Temperature for evaluation model (default: `0.0`)
- `--max_tokens`: Max tokens for evaluation model (default: `4096`)
- `--grader_max_tokens`: Max tokens for grader model (default: `2048`)

**MCP Agent Configuration:**
- `--agent_type`: Type of MCP agent (`simple`, `research`, `client`)
- `--verbose_agents`: Enable verbose output for MCP agents

**Benchmark-specific Arguments:**
- `--subset_name`: HealthBench subset (`hard`, `consensus`, `all`)
- `--data_path`: Custom dataset path for ResearchQA

### Examples

#### HealthBench Evaluation

```bash
# GPT evaluation with default models
python run_eval.py healthbench --run_mode gpt_completions --subset_name consensus --num_examples 10

# GPT evaluation with custom models
python run_eval.py healthbench --run_mode gpt_completions --subset_name consensus \
  --model gpt-4o --grader_model gpt-4-turbo --temperature 0.1

# MCP agent evaluation
python run_eval.py healthbench --run_mode mcp_agents --agent_type simple \
  --subset_name consensus --model gpt-4o --verbose_agents

# Cached completions evaluation
python run_eval.py healthbench --run_mode cached_completions --cached_completions_path outputs/healthbench_responses.json
```

#### BrowseComp Evaluation

```bash
# GPT evaluation with custom models and temperature
python run_eval.py browsecomp --run_mode gpt_completions --num_examples 50 \
  --model gpt-4-turbo --grader_model gpt-4o --temperature 0.2 --max_tokens 8192

# MCP agent evaluation (good for search tasks)
python run_eval.py browsecomp --run_mode mcp_agents --agent_type simple \
  --num_examples 25 --model gpt-4-turbo --verbose_agents

# Cached completions evaluation
python run_eval.py browsecomp --run_mode cached_completions --cached_completions_path outputs/browsecomp_responses.json
```

#### ResearchQA Evaluation

```bash
# GPT evaluation with different models for generation and grading
python run_eval.py researchqa --run_mode gpt_completions --num_examples 100 \
  --model gpt-4o --grader_model gpt-4-turbo --max_tokens 6000

# MCP research agent evaluation (recommended for academic questions)
python run_eval.py researchqa --run_mode mcp_agents --agent_type research \
  --num_examples 20 --model gpt-4o --temperature 0.1 --max_tokens 8000

# Cached completions evaluation with custom grader  
python run_eval.py researchqa --run_mode cached_completions \
  --cached_completions_path outputs/researchqa_responses.json --grader_model gpt-4o
```

#### Advanced Model Configuration Examples

```bash
# Use different models for evaluation and grading
python run_eval.py healthbench --run_mode gpt_completions \
  --model gpt-4o --grader_model gpt-4-turbo \
  --temperature 0.1 --max_tokens 8192 --grader_max_tokens 4096

# Higher temperature for more creative responses
python run_eval.py researchqa --run_mode gpt_completions \
  --model gpt-4-turbo --temperature 0.3 --max_tokens 6000 --num_examples 50

# MCP agents with verbose output
python run_eval.py browsecomp --run_mode mcp_agents --agent_type simple \
  --model gpt-4o --verbose_agents --num_examples 10
```

## MCP Agents Deep Dive

### Creating Custom MCP Samplers

```python
from samplers.sampler.rl_rag_mcp_sampler import RLRAGMCPSampler, SimpleSearchAgent, ResearchAgent
from openai import AsyncOpenAI

# Method 1: Use pre-built agents
sampler = create_mcp_sampler_with_websearch(
    model_name="gpt-4o",
    agent_type="research",
    temperature=0.1,
    verbose=True
)

# Method 2: Create custom agent
from mcp_agents.client import LLMToolClient, GenerationConfig
from mcp_agents.tool_interface.mcp_tools import WebSearchTool

openai_client = AsyncOpenAI()
llm_client = LLMToolClient(
    model_name="gpt-4o",
    client=openai_client,
    tools=[WebSearchTool()],
    generation_config=GenerationConfig(temperature=0.1, max_tokens=6000)
)

# Create custom agent class
class CustomAgent(BaseAgent):
    prompt = """You are a specialized assistant for {domain} questions.
    
    Question: {query}
    
    Please provide a detailed answer using web search when needed."""
    
custom_agent = CustomAgent(client=llm_client)
sampler = RLRAGMCPSampler(agent=custom_agent)
```

### Tool Integration

MCP agents can use various tools:

```python
# Web search tool (included by default)
from mcp_agents.tool_interface.mcp_tools import WebSearchTool

# Add multiple tools
tools = [
    WebSearchTool(),
    # Add more tools as they become available
]

llm_client = LLMToolClient(
    model_name="gpt-4o",
    client=openai_client,
    tools=tools
)
```

### Monitoring Agent Behavior

```python
# Enable verbose mode to see:
# - Tool calls made
# - Search queries
# - Reasoning steps
# - Token usage

sampler = create_mcp_sampler_with_websearch(
    model_name="gpt-4o",
    agent_type="research",
    verbose=True  # Shows detailed execution
)

response = sampler(messages)

# Inspect metadata
print("Tool calls:", response.response_metadata["tool_calls"])
print("Total tokens:", response.response_metadata["total_tokens"])
print("Tool call count:", response.response_metadata["tool_call_count"])
```

## Environment Setup

### Required Environment Variables

For OpenAI API:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

For Azure OpenAI:
```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.azure-api.net/"
export OPENAI_API_VERSION="2025-04-01-preview"
```

### Installation
Install required dependencies:
```bash
pip install openai  # For ChatCompletionSampler
pip install requests  # For any web-based samplers

# For MCP agents (ensure rl-rag-mcp directory is available)
# Dependencies should be installed in rl-rag-mcp/requirements.txt
```

## Debugging Tips

1. **Test your sampler independently** before using it in evaluations:
```python
sampler = YourCustomSampler()
test_messages = [{"role": "user", "content": "Test question"}]
response = sampler(test_messages)
print(response.response_text)
```

2. **Use small examples first** to verify your sampler works correctly:
```bash
python run_eval.py healthbench --run_mode gpt_completions --num_examples 1
python run_eval.py researchqa --run_mode mcp_agents --agent_type simple --num_examples 1 --verbose_agents
```

3. **Check the evaluation outputs** for proper formatting and content

4. **Validate your cached completions** match the expected format before evaluation

5. **For MCP agents**: Use `--verbose_agents` to see detailed execution logs

## Performance Considerations

### MCP Agents vs GPT Completions

| Aspect | GPT Completions | MCP Agents |
|--------|----------------|------------|
| Speed | Fast | Slower (due to tool calls) |
| Capabilities | Text generation only | Text + web search + tools |
| Token usage | Lower | Higher (includes tool results) |
| Quality | Good for knowledge-based tasks | Better for current info/research |
| Use cases | General evaluation | Research, fact-checking, current events |

### Recommendations

- Use **GPT completions** for fast evaluation of general knowledge
- Use **MCP agents** when questions require current information or research
- Start with small `--num_examples` when testing MCP agents
- Use `--verbose_agents` during development to understand agent behavior

This samplers module provides a flexible foundation for integrating any language model or response generation system with the evaluation benchmarks through a unified, standardized interface. 

## Advanced features (optional)

### Multimodal helpers
If your sampler needs to handle images or structured text payloads, provide helpers like:

```python
def _handle_image(self, image: str, encoding: str = "base64", format: str = "png"):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{format};{encoding},{image}"}
    }

def _handle_text(self, text: str):
    return {"type": "text", "text": text}
```

### Batching support
For efficiency, you may optionally support batched inference:

```python
def batch_call(self, message_lists: list[MessageList]) -> list[SamplerResponse]:
    # Implement batch processing for multiple inputs if your backend supports it
    raise NotImplementedError
```

## Best practices (concise)
- Include useful `response_metadata` (e.g., model name, temperature, token usage, inference time)
- Preserve the final message payload in `actual_queried_message_list`
- Prefer deterministic settings for evaluation (e.g., temperature=0.0)
- Start with small `--num_examples` to validate wiring and formats
- Validate cached completions conform to the documented schema before running 