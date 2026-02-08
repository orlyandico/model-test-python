# Model Test - Python

A tool-calling evaluation harness for LLMs, across local and cloud backends. A Python rewrite of Docker's [model-test](https://github.com/docker/model-test) tool (originally in Go). Tests models across local and cloud backends on a shopping cart scenario with 17 test cases across 4 difficulty levels (zero-tool, simple, medium, complex),  and get F1 scores, latency numbers, and per-test pass/fail breakdowns.

## Top Results

F1 is calculated at the **individual tool-call level**, not the test-case level. A model that misses 1 tool call in a 3-tool test loses less F1 than one that misses 1 call in a 1-tool test. Extra spurious calls reduce precision.

| Rank | Model | Backend | F1 | Avg Latency |
|------|-------|---------|----|-------------|
| 1 | Claude Sonnet 4.5 | Bedrock | 0.964 | 2.89s |
| 2 | Claude Sonnet 4 | Bedrock | 0.957 | 2.48s |
| 3 | Claude Haiku 4.5 | Bedrock | 0.949 | 1.67s |
| 4 | GLM-4.7 | Vertex AI MaaS | 0.925 | 1.39s |
| 5 | Amazon Nova 2 Lite | Bedrock | 0.920 | 0.84s |
| 6 | Qwen3 8B | Ollama | 0.909 | 9.22s |
| 7 | Qwen3 1.7B | Ollama | 0.906 | 2.77s |
| 8 | Qwen3-Coder-Next (Q3_K_XL) | llama.cpp | 0.901 | 1.77s |
| 9 | GLM-4.7-flash | Ollama | 0.900 | 3.46s |
| 10 | Gemini 2.0 Flash | Vertex AI | 0.873 | 1.29s |
| 11 | Gemini 2.5 Pro | Vertex AI | 0.867 | 4.51s |
| 12 | Qwen3 4B | Ollama | 0.859 | 17.81s |
| 13 | Kimi K2 Thinking | Vertex AI MaaS | 0.856 | 1.19s |
| 14 | Nemotron-3-nano | Ollama | 0.852 | 15.22s |
| 15 | Amazon Nova Micro | Bedrock | 0.832 | 0.82s |
| 16 | Qwen3 0.6B | Ollama | 0.804 | 2.44s |
| 17 | Granite4 350M | Ollama | 0.754 | 0.57s |
| 18 | FunctionGemma | Ollama | 0.716 | 1.07s |

## Economics

The best model on general benchmarks isn't necessarily the best choice for tool calling. The F1 spread between rank 1 and rank 7 is only 5.8 points, but the cost spread is enormous.

| Model | F1 | Input $/MTok | Output $/MTok | Relative cost |
|-------|----|-------------|--------------|---------------|
| Claude Sonnet 4.5 | 0.964 | $3.00 | $15.00 | 1x (baseline) |
| Claude Haiku 4.5 | 0.949 | $1.00 | $5.00 | 3x cheaper |
| GLM-4.7 | 0.925 | $0.40 | $1.50 | 7–10x cheaper |
| Amazon Nova 2 Lite | 0.920 | $0.33 | $2.75 | ~6–9x cheaper |
| Gemini 2.0 Flash | 0.873 | $0.15 | $0.60 | 20–25x cheaper |
| Qwen3 1.7B (self-hosted) | 0.906 | ~$0.05 | ~$0.61 | see below |

**The takeaway:** For tool-calling workloads, you're paying 7–10x more for Sonnet 4.5 to gain 4 F1 points over GLM-4.7. That's a defensible trade-off if you need peak accuracy, but most agent pipelines don't. GLM-4.7 at $0.40/$1.50 per million tokens is the strongest cost/accuracy ratio among cloud models — it ranks 4th in F1 but costs a fraction of the Anthropic models above it. Nova 2 Lite is in a similar bracket at $0.33/$2.75, with the added advantage of being the fastest cloud model in the top 5 (0.84s average latency).

**What about self-hosting?** We benchmarked Qwen3 1.7B (4-bit quantized) on an NVIDIA P40, roughly equivalent to a T4 in memory-bound workloads. Measured throughput: ~103 tokens/s decode, ~1,356 tokens/s prefill. On an AWS `g4dn.xlarge` (1x T4, $0.227/hr on a 3-year RI), that works out to ~$0.05/MTok input and ~$0.61/MTok output — *if the GPU is saturated 24/7*. That's the catch: reserved instances charge whether you use them or not. At 50% utilization those per-token costs double; at 10% they're 10x higher and suddenly more expensive than the API models above.

For most tool-calling workloads, **serverless inference (pay-per-token) is the better default** unless you have sustained, predictable throughput that keeps the GPU busy. GLM-4.7 ($0.40/$1.50), Nova 2 Lite ($0.33/$2.75), Gemini 2.0 Flash ($0.15/$0.60), and Gemini 2.5 Flash ($0.30/$2.50 — not yet tested here but worth considering) all come in well under $3/MTok output with zero idle cost. Self-hosting only wins when you can guarantee high utilization — and at that point you'd also want vLLM or TensorRT-LLM with continuous batching to maximize throughput, not Ollama.

The bottom line: for tool calling, the most capable model is rarely the most cost-effective. The F1 difference between rank 1 and rank 7 is under 6 points, but the cost difference is orders of magnitude. Save the expensive frontier models for tasks that actually need their reasoning capabilities.


## Why This Exists

If you're building an agent that calls tools, you need to pick a model — but public benchmarks like [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) are dominated by cloud API models. They won't tell you how a 1.7B local model would compare, or handle a multi-step agent loop where it needs to chain tool calls based on intermediate results.

Docker's ["Local LLM Tool Calling: A Practical Evaluation"](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/) is an attempt to fill that gap: 21 local models tested across 3,570 cases with a Go harness. But it had three limitations that made it hard to use for real model selection:

1. **Local only.** The original [docker/model-test](https://github.com/docker/model-test) only tested Ollama models. If you wanted to compare a local Qwen3 against Claude or Nova on the same test suite, you couldn't. This tool adds AWS Bedrock, Google Vertex AI, and Vertex AI Model Garden backends so you can run the same 17 tests against local and cloud models side by side.

2. **Brittle evaluation.** Docker's harness used exact tool-sequence matching: if the model called the right tools in a slightly different (but valid) order, it failed. This produced false negatives on models that were actually doing the right thing. This tool adds a two-tier evaluation — fast brittle matching first, then an LLM-as-judge fallback (Claude Sonnet 4.5 via Bedrock) that evaluates semantic correctness of tool selection, sequencing, and parameters.

3. **Too few agent rounds.** The original capped the agent loop at 5 rounds.  Granted, the 5-round limit was an opinionated choice on the part of the original author. However, complex multi-step tasks (search, add multiple items, view cart, checkout) often need more iterations, especially for smaller models that take an exploratory approach. This tool gives models 10 rounds, reducing false failures from premature truncation.

## Analysis

**The stumbling block: `complex_cart_management`.** The top 5 models all passed 16 out of 17 tests, and four of them failed the same one. The prompt asks: *"show me what's in my cart, remove any duplicate items, and add one Samsung Galaxy S24."* When the model calls `view_cart`, it gets back `iPhone, quantity: 2`. Every model that failed this test looked at that result, concluded there were no duplicates — just "multiple units of the same product, not a duplicate entry" as one model put it — and skipped the `remove_from_cart` call entirely. The models distinguish between duplicate line items and a quantity of 2, and since they see a single entry, they decide no removal is needed. The user asked them to remove duplicates; the quantity of 2 *is* the duplicate; the models don't make that connection.

The exception was Haiku 4.5, which passed `complex_cart_management` in some runs but instead stumbled on `simple_add_iphone` — it searched for iPhones, found two variants (iPhone and iPhone 15), and asked the user to clarify instead of just adding the item. A different failure mode (over-cautious rather than under-reasoning), but still 16/17.

**What this tells you about model selection:**

- **Sonnet 4.5 vs Sonnet 4:** The newer model gains 0.7 F1 points but costs 17% more latency. Marginal for tool-calling workloads.
- **Haiku 4.5 is the efficiency sweet spot:** 1.5 F1 points behind Sonnet 4.5 at nearly half the latency. For high-volume tool-calling agents, this is the trade-off to beat.
- **Nova 2 Lite is the speed play:** 3.4x faster than Sonnet 4.5 with F1 still above 0.91. If your use case is latency-sensitive and can tolerate slightly less precise tool selection, this is compelling.
- **Qwen3 1.7B punches above its weight:** A 1.7B parameter local model at F1 0.906 — ahead of Gemini 2.5 Pro (0.867) and within striking distance of cloud models, at the cost of higher latency on consumer hardware.
- **FunctionGemma is fast but single-step only.** Despite being tiny and low-latency, it lands last because it can only handle single-tool test cases. On any multi-step test, it executes the first tool call and then stops — it never chains a second call based on the result. If your use case is strictly single-turn function dispatch, it works; for anything requiring an agent loop, it doesn't.
- **All top models hit the same ceiling.** The 17th test isn't a flaw in the models — it's a deliberately ambiguous scenario that tests whether a model will act on an instruction even when its own interpretation suggests the action is unnecessary. Most current models won't.



## Quick Start (No Installation)

```bash
# Install dependencies
pip3 install openai boto3 google-cloud-aiplatform

# Run with a local model via Ollama
python3 run.py --model "ollama/llama3.2"

# Run with Google Vertex AI
export GOOGLE_CLOUD_PROJECT="your-project-id"
python3 run.py --model "vertex/gemini-2.0-flash"

# Run with AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
python3 run.py --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
```

## Supported Backends

All models require a prefix to specify the backend.

| Prefix | Backend | Connects to |
|--------|---------|-------------|
| `ollama/` | Ollama server | `host:11434` (default `localhost`) |
| `llama.cpp/` | llama.cpp server | `host:8080` (default `localhost`) |
| `bedrock/` | AWS Bedrock | AWS API |
| `vertex/` | Google Vertex AI (Gemini) | Google Cloud API |
| `vertex-maas/` | Vertex AI Model Garden MaaS | Google Cloud API (OpenAI-compatible) |

### Ollama (local or remote)

```bash
# Start server in another terminal: ollama serve
python3 run.py --model "ollama/qwen3:8b"
python3 run.py --model "ollama/functiongemma:latest"
python3 run.py --model "ollama/nemotron-3-nano"
python3 run.py --model "ollama/granite4:350m"

# Connect to Ollama on a remote host
python3 run.py --model "ollama/qwen3:8b" --host myserver.local
```

By default Ollama only listens on `127.0.0.1`. To allow remote connections, set `OLLAMA_HOST=0.0.0.0` on the machine running Ollama:

```bash
# One-off
OLLAMA_HOST=0.0.0.0 ollama serve

# Or permanently via systemd
sudo systemctl edit ollama
# Add:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### llama.cpp (local)

Start `llama-server`, then run tests with the `llama.cpp/` prefix. By default llama.cpp only listens on localhost, the `--host 0.0.0.0` parameter is analogous to `OLLAMA_HOST` for ollama. The `--jinja` flag is required for tool calling support.

```bash
# Start server (example)
./llama-server \
    --model models/Qwen3-4B-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 \
    --jinja \
    --threads 10 \
    --temp 1.0 \
    --top-p 0.95 \
    --ctx-size 24576 \
    --seed 3407 \
    --fit on

# Then run tests
python3 run.py --model "llama.cpp/Qwen3-4B-UD-Q4_K_XL.gguf"

# Connect to llama.cpp on a remote host
python3 run.py --model "llama.cpp/Qwen3-4B-UD-Q4_K_XL.gguf" --host 192.168.1.100
```

### AWS Bedrock (cloud)

The model ID (string after the `bedrock/` prefix) is the Bedrock model ID or inference profile.

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1

python3 run.py --model "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
python3 run.py --model "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
python3 run.py --model "bedrock/global.amazon.nova-2-lite-v1:0"
python3 run.py --model "bedrock/us.amazon.nova-micro-v1:0"
```

### Google Vertex AI (cloud)

Vertex AI requires two different prefixes depending on whether you are calling a Google first-party model or a third-party model from the Model Garden. This is unlike AWS Bedrock, where a single `bedrock/` prefix handles both Amazon and third-party models through the same Converse API.

- **`vertex/`** — Google's own Gemini models, accessed via the native Vertex AI Generative AI SDK.
- **`vertex-maas/`** — Third-party models from the [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) (e.g., GLM, DeepSeek), served as Model-as-a-Service (MaaS) through an OpenAI-compatible API with a different endpoint and model ID format (`publisher/model`).

Both use the same GCP credentials (Application Default Credentials).

#### Vertex AI — Gemini models (`vertex/`)

```bash
# One-time setup
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com

# Set project (or use GOOGLE_CLOUD_PROJECT env var)
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # Optional, defaults to us-central1

python3 run.py --model "vertex/gemini-2.0-flash"
python3 run.py --model "vertex/gemini-2.5-flash"
python3 run.py --model "vertex/gemini-2.5-pro"
```

The project can also be picked up automatically from `gcloud config set project`.

#### Vertex AI — Model Garden MaaS (`vertex-maas/`)

Third-party models must first be enabled in your project via the Model Garden console page.

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"

python3 run.py --model "vertex-maas/zai-org/glm-4.7-maas"
python3 run.py --model "vertex-maas/deepseek-ai/deepseek-v3.1-maas"
```

The MaaS endpoint defaults to the `global` region. To use a specific region instead, set `VERTEX_MAAS_LOCATION` or `GOOGLE_CLOUD_LOCATION`.

## Command Line Options

```
--model         Model name with required prefix (REQUIRED)
--config        Test cases config file (default: "config/test_cases.json")
--test-case     Run only specified test case
--host          Hostname for Ollama/llama.cpp backends (default: localhost)
--wait-timeout  Seconds to wait for local server (default: 30)
--api-key       API key (only needed for some backends)
```

## Features

- Agent loop testing with up to 10 rounds
- 5 backends: Ollama, llama.cpp, AWS Bedrock, Google Vertex AI, Vertex AI Model Garden MaaS
- Shopping cart simulation with 5 tools (search, add, remove, view, checkout)
- 17 test cases across 4 difficulty levels (zero-tool, simple, medium, complex)
- Two-tier evaluation: brittle matching + LLM-as-judge
- JSON result output with detailed performance metrics

## Sample Output

```
============================================================
Test: medium_remove_and_add
Prompt: Remove iPhone from cart and add Samsung Galaxy instead
============================================================

--- Round 1/10 ---
LLM response time: 11.78s
Tool calls requested: 1
  [1] remove_from_cart({"product_name": "iPhone"})
      → Result: iPhone not in cart

--- Round 2/10 ---
LLM response time: 3.25s
Tool calls requested: 1
  [1] add_to_cart({"product_name": "Samsung Galaxy"})
      → Result: Added 1 Samsung Galaxy to cart

--- Round 3/10 ---
LLM response time: 2.29s
✓ Final message: The iPhone was not present in your cart, so nothing was removed.
  I've added Samsung Galaxy to your cart (1 unit). Let me know if you'd like to
  adjust the quantity or proceed with checkout!
✓ Test completed in 3 rounds

────────────────────────────────────────────────────────────
TEST RESULT: medium_remove_and_add
────────────────────────────────────────────────────────────
✅ PASSED - Matched variant: remove_then_add
   Tool calls made: 2
   Total time: 17.32s
```

Here is an example where brittle matching failed and the LLM judge was invoked:

```
============================================================
Test: complex_cart_management
Prompt: Help me organize my shopping cart. First show me what's currently in it,
  then remove any duplicate items, and add one Samsung Galaxy S24.
============================================================

--- Round 1/10 ---
LLM response time: 63.86s
Tool calls requested: 1
  [1] view_cart({})
      → Result: Cart: [{'product_name': 'iPhone', 'quantity': 2},
        {'product_name': 'Wireless Headphones', 'quantity': 1}], Total: $2199.97

--- Round 2/10 ---
LLM response time: 19.26s
Tool calls requested: 1
  [1] add_to_cart({"quantity": 1, "product_name": "Samsung Galaxy S24"})
      → Result: Added 1 Samsung Galaxy S24 to cart

--- Round 3/10 ---
LLM response time: 9.25s
✓ Final message: The shopping cart has been updated successfully:
  1. Viewed current contents: iPhone x2 and Wireless Headphones x1.
  2. No duplicate items found needing removal.
  3. Added 1 Samsung Galaxy S24 to the cart.
✓ Test completed in 3 rounds

────────────────────────────────────────────────────────────
TEST RESULT: complex_cart_management
────────────────────────────────────────────────────────────
  🤖 Brittle match failed, evaluating with LLM judge...
  📝 Judge reasoning: The model correctly called view_cart and add_to_cart with
     appropriate arguments, but it failed to address a key part of the user's
     request: 'remove any duplicate items'. The user explicitly asked for three
     actions: (1) view cart, (2) remove duplicates, and (3) add Samsung Galaxy
     S24. The model only performed actions 1 and 3, skipping the duplicate
     removal step entirely. The sequence is incomplete for the stated intent.
❌ FAILED - Tool calls did not match any expected variant

   Expected (any of these):
     Variant 1: cart_organization
       1. view_cart({})
       2. remove_from_cart({"product_name": "iPhone"})
       3. add_to_cart({"product_name": "Samsung Galaxy S24", "quantity": 1})

   Actual tool calls made:
     1. view_cart({})
     2. add_to_cart({"quantity": 1, "product_name": "Samsung Galaxy S24"})

   Total time: 92.37s
```

## Evaluation Methodology

Based on [Docker's Local LLM Tool Calling evaluation](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/).

### Metrics

| Metric | Description |
|--------|-------------|
| Tool Invocation | Did the model realize a tool was needed? |
| Tool Selection | Did it choose the right tool(s)? |
| F1 Score | Harmonic mean of precision and recall |

**F1 Score Calculation** (at individual tool level, not test case level):
- **Precision** = valid tool calls / all tool calls made
- **Recall** = correct tool calls / expected tool calls
- **F1** = 2 x (Precision x Recall) / (Precision + Recall)

### Not All-or-Nothing

We deliberately avoid requiring perfect predictions:
- Each correct tool call gets credit, even if the full sequence isn't perfect
- Focus on whether the tool selection makes sense for the intent
- Exact parameter matches are NOT required (e.g., "blue shirt" ~ "Blue T-Shirt")

### Evaluation Flow

1. **Edge cases**: No tools expected/called handled directly
2. **Brittle match**: Try exact tool name/sequence matching first (fast, no API call)
3. **LLM-as-judge**: Fall back to semantic evaluation if brittle match fails

The LLM judge evaluates:
- Tool selection appropriateness for user intent
- Sequence logic (do the tool calls make sense?)
- Parameter reasonableness (not exact matches)
- Partial credit for mostly-correct responses

### LLM Judge Requirement

**Important:** The LLM-as-judge fallback always uses **Claude Sonnet 4.5 via AWS Bedrock**, regardless of which backend you are testing. This means you need valid AWS Bedrock credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) even when testing Ollama, llama.cpp, or Vertex AI models, unless every test case passes via brittle matching alone.

- **Region**: `us-west-2`
- **Model**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`

If the judge is unavailable (e.g., no AWS credentials), tests that fail brittle matching will be marked as failed rather than evaluated semantically.

## Batch Analysis

Analyze results across multiple test runs:

```bash
python3 analyse_batch.py results/
python3 analyse_batch.py results/ --format json -o analysis.json
```

## Output

Results saved to `results/agent_test_results_<model>_<timestamp>.json`

## References

- **Original blog post:** [Local LLM Tool Calling: A Practical Evaluation](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/) (Docker, 2025)
- **Original Go implementation:** [docker/model-test](https://github.com/docker/model-test)
