# I Benchmarked 14 LLMs on Tool Calling. The Results Challenge Conventional Wisdom.

Docker recently published ["Local LLM Tool Calling: A Practical Evaluation"](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/), testing 21 models across 3,570 test cases. It is the most rigorous public evaluation of LLM function-calling I have seen. I wanted to build on their work -- extending the benchmark to cloud models, adding more local contenders, and replacing brittle string-matching with a more robust LLM-as-judge evaluation layer.

Here is what I found after 121 test runs across 14 models.

## The Setup

I rewrote Docker's [model-test](https://github.com/docker/model-test) tool in Python, keeping the same core design: a shopping cart simulation with five tools (search products, add to cart, remove from cart, view cart, checkout) tested across 17 cases of escalating difficulty. The test cases span four categories:

- **Zero-tool cases** (5 tests): Greetings, general questions -- the model should recognise no tool is needed.
- **Simple cases** (5 tests): Single tool operations.
- **Medium cases** (4 tests): Two-step sequences (search then add, view then remove).
- **Complex cases** (3 tests): Multi-step workflows requiring 3-5 coordinated tool calls.

Each test runs in an agent loop of up to 10 rounds, giving models every opportunity to complete multi-step workflows. Multiple valid execution paths are accepted per test case -- the evaluator does not enforce a single "correct" sequence.

## Beyond Brittle Matching: LLM-as-Judge

The Docker benchmark uses exact-match evaluation: either the model produced the expected tool call sequence or it did not. This is fast, but it penalises models that arrive at the right outcome through a slightly different path. A model that searches by category instead of by keyword, or calls `view_cart` as a reasonable extra step, gets scored the same as one that calls the wrong tool entirely.

I implemented a two-tier evaluation approach:

1. **Brittle match first.** Try exact tool name and sequence matching. When it passes, no API call is needed and results are deterministic.
2. **LLM-as-judge fallback.** When the brittle match fails, Claude Sonnet 4.5 evaluates the model's output semantically -- assessing whether the tool selection was appropriate for the user's intent, whether the sequence logic is sound, and whether parameters are reasonable even if not character-for-character identical.

This matters more than it might seem. Several models made defensible tool-calling decisions that would fail a strict string comparison but represent perfectly valid behaviour. The LLM-as-judge layer captures this nuance and provides partial credit where warranted.

Metrics are reported as F1 scores at the individual tool level, calculated from precision (valid tool calls / all tool calls made) and recall (correct tool calls / expected tool calls).

## The Results

Every model was run multiple times (5-22 runs depending on the model) to account for non-determinism. Results are macro-averaged across runs.

### Ranked by Tool Selection F1

| Rank | Model | Type | Selection F1 | Invocation F1 | Avg Latency | Runs |
|------|-------|------|-------------|---------------|-------------|------|
| 1 | Claude Sonnet 4.5 | Cloud (Bedrock) | 0.964 | 1.000 | 2.89s | 5 |
| 2 | Claude Sonnet 4 | Cloud (Bedrock) | 0.957 | 1.000 | 2.48s | 5 |
| 3 | Claude Haiku 4.5 | Cloud (Bedrock) | 0.949 | 1.000 | 1.67s | 5 |
| 4 | Amazon Nova 2 Lite | Cloud (Bedrock) | 0.919 | 1.000 | 0.78s | 5 |
| 5 | Qwen3 8B | Local (Ollama) | 0.909 | 1.000 | 9.22s | 8 |
| 6 | Qwen3 1.7B | Local (Ollama) | 0.904 | 1.000 | 2.76s | 9 |
| 7 | Qwen3 Coder Next | Local (llama.cpp) | 0.901 | 1.000 | 1.77s | 6 |
| 8 | GLM-4.7 Flash | Local (Ollama) | 0.900 | 0.997 | 3.46s | 13 |
| 9 | Qwen3 4B | Local (Ollama) | 0.859 | 0.972 | 17.81s | 22 |
| 10 | Nemotron-3 Nano | Local (Ollama) | 0.845 | 0.926 | 15.28s | 8 |
| 11 | Amazon Nova Micro | Cloud (Bedrock) | 0.832 | 0.975 | 0.82s | 5 |
| 12 | Qwen3 0.6B | Local (Ollama) | 0.804 | 0.893 | 2.44s | 9 |
| 13 | Granite4 350M | Local (Ollama) | 0.754 | 0.899 | 0.57s | 9 |
| 14 | FunctionGemma 270M | Local (Ollama) | 0.716 | 0.953 | 1.07s | 12 |

## What Stands Out

### Cloud models still lead, but the gap is narrow

Claude Sonnet 4.5 tops the table at F1 0.964, but the best local model -- Qwen3 8B running on commodity hardware through Ollama -- reaches 0.909. That is a 5.5-point gap. For many production use cases, particularly those with latency tolerance or data residency requirements, that gap is entirely acceptable.

### Qwen3 dominates the local tier

Across four size variants tested (0.6B, 1.7B, 4B, 8B), Qwen3 consistently outperforms its peers. The 1.7B variant is particularly notable: it achieves F1 0.904 -- within striking distance of models 4-5x its size -- with an average latency of just 2.76 seconds. Docker's original evaluation reached a similar conclusion: Qwen models dominate the open-source tool-calling landscape.

### Invocation discipline separates the tiers

The top 8 models all achieve near-perfect tool invocation F1 (0.997-1.000), meaning they almost never call a tool when they should not, and almost never fail to call one when they should. Below that threshold, models start misfiring -- calling `search_products` in response to "What is AI?" or invoking `checkout` when the user says "Thank you." This invocation discipline is arguably the most important capability for production deployments, where a false positive tool call can trigger real-world side effects.

### Model size is not destiny

FunctionGemma (270M parameters) scores higher on tool invocation (F1 0.953) than Granite4 (350M, F1 0.899), despite being 23% smaller. It knows *when* to call tools with near-perfect accuracy. Where it falls short is in multi-step *selection* -- it tends to stop after the first step of a complex workflow rather than continuing to completion. Specialised training for function calling clearly matters more than raw parameter count.

### The 4B anomaly

Qwen3 4B (F1 0.859, 17.81s latency) actually scores lower than Qwen3 1.7B (F1 0.904, 2.76s latency) in this benchmark. It is also the slowest local model tested by a significant margin. More parameters do not automatically translate to better tool-calling performance, and in this case the larger model appears to overthink multi-step sequences, degrading both accuracy and speed.

### Amazon Nova surprises

Amazon Nova 2 Lite takes fourth place overall (F1 0.919) with the fastest latency of any model above F1 0.900 at just 0.78 seconds per call. For high-throughput applications where sub-second latency matters more than the last few points of F1, this is a compelling option that has received relatively little attention in the tool-calling conversation.

## The LLM-as-Judge Difference

Switching from pure string matching to LLM-as-judge evaluation changed the scores meaningfully for several models. Models that take a reasonable but non-standard path -- searching by category instead of keyword, or adding an extra `view_cart` call to confirm state before proceeding -- are no longer penalised. This better reflects real-world usage, where the goal is to accomplish the user's intent, not to produce a specific token sequence.

The judge itself (Claude Sonnet 4.5 via Bedrock) evaluates tool selection appropriateness, sequence logic, and parameter reasonableness. It provides partial credit for mostly-correct responses rather than binary pass/fail. The two-tier approach keeps costs down: the judge is only invoked when the fast brittle match fails, which happens on roughly 20-30% of test cases depending on the model.

## Practical Takeaways

**If you need maximum accuracy and can use a cloud API:** Claude Sonnet 4.5 or Sonnet 4 via Bedrock. Perfect invocation discipline, highest selection accuracy, and consistent results across runs with near-zero variance.

**If you need a local model for tool calling:** Qwen3 8B is the safest bet. Qwen3 1.7B is the efficiency pick -- nearly as accurate at a fraction of the compute.

**If latency is your primary constraint:** Amazon Nova 2 Lite (cloud) or Granite4 350M (local) offer the fastest inference, though with meaningful accuracy trade-offs at the lower end.

**If you are building edge/mobile applications:** FunctionGemma at 270M parameters is purpose-built for this. Its invocation discipline is excellent for single-step tool calling; fine-tune it on your specific domain to close the selection gap.

**If you are evaluating models for agentic workflows:** Do not rely solely on exact-match benchmarks. An LLM-as-judge layer captures the semantic validity of tool-calling behaviour that string matching misses. The Berkeley Function-Calling Leaderboard is useful for broad comparisons, but domain-specific evaluation with semantic grading will give you a more accurate picture of real-world performance.

## Methodology Notes

All local models were run via Ollama or llama.cpp on the same hardware. Cloud models were accessed through AWS Bedrock. The test harness, configuration, and all raw results (121 JSON files across 14 models) are available for inspection. Each model was run a minimum of 5 times; high-variance models (Qwen3 4B, GLM-4.7 Flash) were run more frequently to establish stable averages. The LLM judge uses a separate model (Claude Sonnet 4.5 in us-west-2) from any model under test to avoid evaluation bias.

---

*This analysis builds on Docker's ["Local LLM Tool Calling: A Practical Evaluation"](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/) and uses a Python rewrite of their open-source [model-test](https://github.com/docker/model-test) framework. The evaluation code, test configurations, and all raw results are available on request.*
