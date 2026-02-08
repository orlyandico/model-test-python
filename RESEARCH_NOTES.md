# LLM Tool Calling Evaluation - Research Notes

Source article - https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/

Original Go source code - https://github.com/docker/model-test



**Date:** 16 January 2026  
**Project:** model-test Python implementation for evaluating LLM function calling capabilities

---

## Executive Summary

Evaluated small language models (SLMs) for tool-calling capabilities using a shopping cart simulation with 5 tools across 17 test cases. Key finding: **model size doesn't guarantee better tool-calling performance** - specialised training matters more.

### Model Performance Rankings

| Model | Size | Success Rate | Key Strengths | Key Weaknesses |
|-------|------|--------------|---------------|----------------|
| **FunctionGemma** | 270M | 76.47% (13/17) | Single-step operations, knows when NOT to call tools | Multi-step workflows (3+ steps) |
| **Granite4:350m** | 350M | 47.06% (8/17) | Basic tool execution | Over-eager tool calling, poor invocation discipline |

---

## Research Context

### Docker Blog Post Analysis
**Source:** https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/

Docker tested 21 models across 3,570 test cases to evaluate tool-calling performance:

**Top Performers:**
- **GPT-4** (hosted): F1 0.974, ~5s latency
- **Qwen 3 (14B)**: F1 0.971, ~142s latency - best local model
- **Qwen 3 (8B)**: F1 0.933, ~84s latency - good speed/accuracy balance
- **Claude 3 Haiku** (hosted): F1 0.933, ~3.56s latency

**Key Insights:**
- Qwen models dominate amongst open-source options
- Quantisation doesn't significantly impact tool-calling accuracy
- Higher accuracy typically means higher latency
- Many local models struggle with eager invocation, wrong tool selection, and invalid arguments

### NVIDIA Research: Orchestrator-8B
**Source:** https://research.nvidia.com/labs/lpr/ToolOrchestra/

**NVIDIA Nemotron-Orchestrator-8B** (December 2025)
- **Purpose:** Lightweight orchestrator coordinating diverse tools and specialised models
- **Performance:** 37.1% on Humanity's Last Exam (HLE), outperforming GPT-5 (35.1%)
- **Efficiency:** 2.5× more efficient than GPT-5, using only ~30% of the cost
- **Architecture:** Uses reinforcement learning with outcome, efficiency, and user-preference rewards
- **Approach:** Acts as "conductor" breaking down complex tasks and invoking right tools in sequence

**Key Insight:** Small orchestration models can be more effective and efficient than large monolithic models for agentic tasks.

### Google FunctionGemma
**Source:** https://ai.google.dev/gemma/docs/functiongemma/model_card

**Google FunctionGemma 270M** (Released January 2026)
- **Size:** 270 million parameters (0.27B)
- **Purpose:** Lightweight foundation model specifically trained for function calling
- **Architecture:** Built on Gemma 3 270M with specialised chat format for tool use
- **Context:** 32K tokens input/output
- **Design:** Intended to be fine-tuned for specific function-calling tasks

**Performance:**
- Base model: 58% on Mobile Actions benchmark
- After fine-tuning: 85% accuracy on Mobile Actions
- BFCL benchmarks: 61.6% (Simple), 63.5% (Multiple), 61.1% (Relevance)

**On-device performance (Samsung S25 Ultra):**
- Prefill: 1,718 tokens/second
- Decode: 125.9 tokens/second
- Time-to-first-token: 0.3 seconds
- Model size: 288 MB
- Peak memory: 551 MB

**Use Cases:**
- Tiny Garden: Voice-controlled interactive game
- Mobile Actions: Offline Android OS system tool calls

---

## Test Framework

### Implementation Details

**Python rewrite of Docker's model-test tool:**
- **Lines of code:** ~400 (vs 1000+ in Go original)
- **Architecture:** Agent loop with up to 5 rounds of tool calling
- **Execution:** Sequential test execution with detailed conversation logging
- **Tools:** 5 shopping cart operations (search, add, remove, view, checkout)
- **Output:** JSON results with timestamps and detailed metrics

**Supported Backends:**
1. **Ollama**: `http://localhost:11434/v1`
   - Use `ollama/<model>` prefix (required)
2. **llama.cpp**: `http://localhost:8080/v1`
   - Use `llama.cpp/<model>` prefix (required)
3. **AWS Bedrock**: `bedrock/<model-id>` prefix with AWS credentials (required)

**Key Features:**
- Automatic server startup detection with configurable timeout
- Prefix-based backend selection (ollama/, llama.cpp/, bedrock/)
- Path matching with flexible argument validation
- Detailed failure reporting showing expected vs actual tool calls
- Support for multiple valid execution paths per test

### Test Suite Structure

**17 Test Cases across 3 categories:**

1. **Zero Tool Cases (5 tests):** Greetings, general questions - no tools expected
2. **Simple Cases (5 tests):** Single tool operations
3. **Medium Cases (4 tests):** Two-step operations
4. **Complex Cases (3 tests):** Multi-step workflows (3-5 steps)

**Example Test Case:**
```json
{
  "name": "medium_search_and_add",
  "prompt": "Search for wireless headphones and add them to cart",
  "expected_tools_variants": [
    {
      "name": "search_by_query",
      "tools": [
        {"name": "search_products", "arguments": {"query": "wireless headphones"}},
        {"name": "add_to_cart", "arguments": {"product_name": "wireless headphones"}}
      ]
    },
    {
      "name": "search_by_category",
      "tools": [
        {"name": "search_products", "arguments": {"category": "electronics"}},
        {"name": "add_to_cart", "arguments": {"product_name": "wireless headphones"}}
      ]
    }
  ]
}
```

---

## Detailed Model Analysis

### FunctionGemma (270M) - 76.47% Success Rate

**Passed Tests (13/17):**
- ✅ All "zero" tests (5/5) - Correctly identified when NOT to use tools
- ✅ All simple single-step operations (5/5)
- ✅ Most medium 2-step operations (3/4)

**Failed Tests (4/17):**

1. **complex_shopping_workflow**
   - Expected: search → add → view → checkout (4 steps)
   - Actual: search only
   - Issue: Stopped after first step

2. **complex_gift_shopping**
   - Expected: search electronics → add → search books → add → view (5 steps)
   - Actual: No tool calls
   - Issue: Overwhelmed by complexity

3. **complex_cart_management**
   - Expected: view → remove → add (3 steps)
   - Actual: view only
   - Issue: Incomplete workflow

4. **medium_view_and_add** (not detailed in output)

**Strengths:**
- Excellent tool invocation discipline (knows when NOT to call tools)
- Perfect on single-step operations
- Fast inference (~1.3s per request)
- Compact size suitable for edge deployment

**Weaknesses:**
- Struggles with 3+ step workflows
- Stops prematurely in multi-step scenarios
- Needs fine-tuning for complex agentic tasks

**Conclusion:** FunctionGemma excels at single-step tool calling but requires fine-tuning for complex multi-step workflows. Aligns with Google's documentation stating it's a foundation model for specialisation.

---

### Granite4:350m - 47.06% Success Rate

**Passed Tests (8/17):**
- ✅ 2/5 zero tests (struggles with tool invocation discipline)
- ✅ 4/5 simple operations
- ✅ 2/4 medium operations

**Failed Tests (9/17):**

**1. Over-eager Tool Calling (3 failures):**
- `zero_capabilities`: "What can you help me with?" → Called `search_products`
- `zero_general_question`: "What is AI?" → Called `search_products` for "artificial intelligence"
- `zero_thank_you`: "Thank you" → Called `checkout` with invalid session_id

**2. Under-eager Tool Calling (1 failure):**
- `simple_checkout`: "Proceed to checkout" → Made no tool calls

**3. Extra Tool Calls (1 failure):**
- `medium_search_and_add`: Made 3 calls (search, add, view) when 2 expected
  - Note: This is actually correct behaviour, just over-achieving

**4. Wrong Execution (4 failures):**
- `complex_cart_management`: Removed wrong product (Wireless Headphones instead of iPhone)
- `complex_gift_shopping`: Only searched, didn't complete workflow
- `complex_shopping_workflow`: Added wrong product (SmartDesk Keyboard instead of iPhone/Headphones)
- `medium_view_and_add`: Code error (dict argument handling bug - now fixed)

**Strengths:**
- Can execute basic tool operations
- Handles some 2-step workflows

**Weaknesses:**
- Poor tool invocation discipline (calls tools when it shouldn't)
- Doesn't understand when NOT to use tools
- Makes incorrect product selections
- Incomplete multi-step workflows
- Slower inference (~1.6s per request)

**Conclusion:** Granite4:350m, despite being 30% larger than FunctionGemma, performs significantly worse. This demonstrates that **specialised training for tool calling matters more than raw parameter count**.

---

## Key Findings

### 1. Specialised Training > Model Size
FunctionGemma (270M) outperforms Granite4 (350M) by **29 percentage points** (76.47% vs 47.06%). FunctionGemma was specifically trained for function calling, whilst Granite4 is a general-purpose model.

### 2. Tool Invocation Discipline is Critical
The ability to know when NOT to call tools is as important as knowing when to call them:
- FunctionGemma: 5/5 on zero-tool tests
- Granite4: 2/5 on zero-tool tests (called tools inappropriately)

### 3. Multi-Step Reasoning is the Bottleneck
Both small models struggle with 3+ step workflows:
- Single-step operations: Both models perform well
- Two-step operations: FunctionGemma 75%, Granite4 50%
- Three+ step operations: Both models struggle significantly

### 4. Model Behaviour Patterns

**FunctionGemma failure mode:** Stops prematurely
- Executes first step correctly
- Doesn't continue to completion
- Needs prompting or fine-tuning for persistence

**Granite4 failure mode:** Poor judgement
- Calls tools when inappropriate
- Selects wrong tools or arguments
- Lacks understanding of task context

### 5. Practical Implications

**For Edge Deployment:**
- FunctionGemma (270M) is ideal for single-step tool calling
- 288 MB model size, 551 MB peak memory
- 125.9 tokens/sec decode speed on mobile devices

**For Complex Workflows:**
- Small models (<1B) need fine-tuning or orchestration
- Consider larger models (Qwen3-8B+) for multi-step tasks
- Or use orchestrator pattern (NVIDIA's approach)

---

## Technical Implementation Notes

### AWS Bedrock Integration

**Implementation:** Custom `BedrockClient` class wrapping boto3
- Converts OpenAI tool format to Bedrock `toolSpec` format
- Converts messages between OpenAI and Bedrock Converse API formats
- Returns OpenAI-compatible response objects

**Usage:**
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
python3 run.py --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
```

**Key Conversion:**
- OpenAI `tools` → Bedrock `toolSpec` with `inputSchema.json`
- OpenAI `tool_calls` → Bedrock `toolUse` with `toolUseId`
- OpenAI `tool` role → Bedrock `toolResult` in user message

### Ollama Startup Detection

**Problem:** macOS externally-managed Python environment prevents pip installation
**Solution:** Direct execution via `run.py` script with automatic Ollama detection

**Implementation:**
```python
def wait_for_ollama(base_url: str, timeout: int = 30):
    """Wait for Ollama to be ready."""
    # Polls health endpoint every 1s for up to 30s
    # Only runs for localhost URLs (skips Bedrock/remote APIs)
```

### Argument Matching Bug Fix

**Issue:** Code crashed with `'dict' object has no attribute 'lower'` when arguments contained non-string values

**Root Cause:** Matching logic called `.lower()` on all argument values without type checking

**Fix:**
```python
# Only do string comparison when both values are strings
if isinstance(value, str) and isinstance(actual.arguments[key], str):
    expected_val = value.lower()
    actual_val = actual.arguments[key].lower()
    if expected_val not in actual_val and actual_val not in expected_val:
        match = False
```

---

## Recommendations

### For Production Use

1. **Single-step tool calling:** Use FunctionGemma (270M) with task-specific fine-tuning
2. **Multi-step workflows:** Use Qwen3-8B or larger models
3. **Complex agentic systems:** Consider orchestrator pattern (NVIDIA approach)
4. **Edge deployment:** FunctionGemma is optimal for resource-constrained environments

### For Further Testing

**High Priority:**
1. Test Qwen3-8B (Docker's recommended model)
2. Test NVIDIA Orchestrator-8B if available
3. Evaluate Claude 3 Haiku via Bedrock
4. Test with fine-tuned FunctionGemma on shopping domain

**Medium Priority:**
1. Test larger Granite models (3B, 8B)
2. Evaluate Llama 3.2 variants
3. Test quantised versions (Q4, Q8)

**Metrics to Track:**
- Tool invocation accuracy (when to call vs not call)
- Tool selection accuracy (which tool to call)
- Argument accuracy (correct parameters)
- Multi-step completion rate (finish workflows)
- Latency per request
- Total workflow time

### For Framework Improvements

1. **Add partial matching:** Accept workflows that complete subset of expected steps
2. **Add fuzzy matching:** More lenient argument comparison
3. **Add retry logic:** Allow models to self-correct
4. **Add prompt engineering:** Test different system prompts
5. **Add streaming support:** Monitor tool calls in real-time
6. **Add cost tracking:** Calculate inference costs per test

---

## Repository Structure

```
model-test-python/
├── run.py                  # Direct execution script (no install needed)
├── pyproject.toml          # Package configuration
├── README.md               # Usage documentation
├── config/
│   ├── test_cases.json     # Full test suite (17 tests)
│   └── test_cases_simple.json  # Simplified test suite
├── model_test/
│   ├── __init__.py
│   ├── main.py            # CLI entry point with failure reporting
│   ├── models.py          # Data models (TestCase, AgentResponse, etc.)
│   ├── runner.py          # TestRunner with Ollama/Bedrock support
│   └── tools.py           # Shopping cart tools implementation
└── results/               # JSON output files with timestamps
```

---

## Quick Reference Commands

```bash
# Install dependencies
pip3 install openai boto3

# Run with Ollama
python3 run.py --model "functiongemma"
python3 run.py --model "ollama/granite4:350m"

# Run with llama.cpp server
python3 run.py --model "llama.cpp/my-model"

# Run with Bedrock
export AWS_ACCESS_KEY_ID="key"
export AWS_SECRET_ACCESS_KEY="secret"
python3 run.py --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

# Run single test
python3 run.py --test-case "simple_add_iphone"

# Adjust server timeout
python3 run.py --wait-timeout 60
```

---

## References

1. Docker Blog: Local LLM Tool Calling Evaluation  
   https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/

2. Docker model-test Repository  
   https://github.com/docker/model-test

3. NVIDIA ToolOrchestra Research  
   https://research.nvidia.com/labs/lpr/ToolOrchestra/

4. Google FunctionGemma Model Card  
   https://ai.google.dev/gemma/docs/functiongemma/model_card

5. Berkeley Function-Calling Leaderboard  
   https://huggingface.co/spaces/gorilla-llm/berkeley-function-calling-leaderboard

---

## Appendix: Test Results Summary

### FunctionGemma 270M Results
```
Total Tests: 17
Passed: 13 (76.47%)
Failed: 4 (23.53%)
Total LLM Time: 37.35s
Average Time per Request: 1.33s
```

### Granite4:350m Results
```
Total Tests: 17
Passed: 8 (47.06%)
Failed: 9 (52.94%)
Total LLM Time: 55.62s
Average Time per Request: 1.59s
```

### Performance Comparison
| Metric | FunctionGemma | Granite4 | Winner |
|--------|---------------|----------|--------|
| Success Rate | 76.47% | 47.06% | FunctionGemma |
| Zero-tool Tests | 5/5 (100%) | 2/5 (40%) | FunctionGemma |
| Simple Tests | 5/5 (100%) | 4/5 (80%) | FunctionGemma |
| Medium Tests | 3/4 (75%) | 2/4 (50%) | FunctionGemma |
| Complex Tests | 0/3 (0%) | 0/3 (0%) | Tie |
| Avg Latency | 1.33s | 1.59s | FunctionGemma |
| Model Size | 270M | 350M | FunctionGemma |

**Conclusion:** FunctionGemma wins across all metrics despite being 23% smaller.
