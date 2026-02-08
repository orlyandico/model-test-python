#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .models import TestCase, ExpectedToolPath, ExpectedToolCall, InitialCartState, InitialCartItem, AgentTestResult, AgentReport
from .runner import TestRunner


def load_test_cases(config_file: str, test_case_name: str | None = None) -> list[TestCase]:
    """Load test cases from JSON file."""
    with open(config_file) as f:
        data = json.load(f)
    
    test_cases = []
    for tc in data:
        initial_cart = None
        if tc.get("initial_cart_state"):
            items = [InitialCartItem(**item) for item in tc["initial_cart_state"]["items"]]
            initial_cart = InitialCartState(items=items)
        
        variants = []
        for variant in tc.get("expected_tools_variants", []):
            tools = [ExpectedToolCall(**tool) for tool in variant["tools"]]
            variants.append(ExpectedToolPath(
                name=variant["name"],
                tools=tools,
                description=variant.get("description", "")
            ))
        
        test_case = TestCase(
            name=tc["name"],
            prompt=tc["prompt"],
            expected_tools_variants=variants,
            initial_cart_state=initial_cart
        )
        
        if test_case_name is None or test_case.name == test_case_name:
            test_cases.append(test_case)
    
    return test_cases


def run_single_test(runner: TestRunner, test_case: TestCase) -> AgentTestResult:
    """Run a single test case."""
    start = time.time()
    response, _, error = runner.run_agent_test(test_case)
    elapsed = time.time() - start

    print(f"\n{'─'*60}")
    print(f"TEST RESULT: {test_case.name}")
    print(f"{'─'*60}")

    if error:
        print(f"❌ FAILED - Error: {error}")
        print(f"   Total time: {elapsed:.2f}s\n")
        return AgentTestResult(
            test_case=test_case,
            success=False,
            response_time=elapsed,
            error_message=error
        )

    matched_path = runner.match_tool_path(response.tool_calls, test_case.expected_tools_variants, test_case.prompt)
    success = bool(matched_path) or len(test_case.expected_tools_variants) == 0

    if success:
        if matched_path:
            print(f"✅ PASSED - Matched variant: {matched_path}")
        else:
            print(f"✅ PASSED - No tool calls expected (and none made)")
        print(f"   Tool calls made: {len(response.tool_calls)}")
        print(f"   Total time: {elapsed:.2f}s\n")
    else:
        print(f"❌ FAILED - Tool calls did not match any expected variant")
        print(f"\n   Expected (any of these):")
        for i, variant in enumerate(test_case.expected_tools_variants, 1):
            print(f"     Variant {i}: {variant.name}")
            for j, tool in enumerate(variant.tools, 1):
                args_str = json.dumps(tool.arguments) if tool.arguments else "{}"
                print(f"       {j}. {tool.name}({args_str})")

        print(f"\n   Actual tool calls made:")
        if response.tool_calls:
            for i, tc in enumerate(response.tool_calls, 1):
                args_str = json.dumps(tc.arguments) if tc.arguments else "{}"
                print(f"     {i}. {tc.tool_name}({args_str})")
        else:
            print(f"     (none)")
        print(f"\n   Total time: {elapsed:.2f}s\n")

    return AgentTestResult(
        test_case=test_case,
        success=success,
        response_time=elapsed,
        response=response,
        matched_path=matched_path
    )


def print_summary(report: AgentReport):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("📈 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Tests:        {report.total_tests}")
    print(f"✅ Passed:          {report.passed_tests}")
    print(f"❌ Failed:          {report.failed_tests}")
    print(f"📊 Success Rate:    {report.passed_tests/report.total_tests*100:.2f}%")
    print(f"⏱️  Total LLM Time:  {report.total_llm_time:.2f}s")
    print(f"⏱️  Avg per Request: {report.avg_time_per_req:.2f}s")
    print("=" * 60)


def wait_for_server(base_url: str, server_name: str, timeout: int = 30):
    """Wait for a local server to be ready."""
    import urllib.request
    import urllib.error

    # Extract host from base_url
    if "/v1" in base_url:
        health_url = base_url.replace("/v1", "")
    else:
        health_url = base_url

    print(f"⏳ Waiting for {server_name} at {health_url}...")

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f"✅ {server_name} is ready")
            return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)

    print(f"❌ {server_name} not ready after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Model testing tool for function calling")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "DMR"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--config", default="config/test_cases.json")
    parser.add_argument("--test-case", default=None)
    parser.add_argument("--wait-timeout", type=int, default=30, help="Seconds to wait for Ollama")
    parser.add_argument("--host", default="localhost", help="Hostname for Ollama/llama.cpp backends (default: localhost)")

    args = parser.parse_args()

    # Create runner to determine backend type
    runner = TestRunner(args.api_key, args.base_url, args.model, host=args.host)

    # Check if a valid backend was specified
    if runner.backend_type is None:
        print("❌ Error: No valid model prefix specified\n")
        print("Usage: python3 run.py --model <prefix>/<model-name> [--host <hostname>]\n")
        print("Supported prefixes:")
        print("  ollama/<model>         - Connect to Ollama (default: localhost:11434)")
        print("  llama.cpp/<model>      - Connect to llama.cpp server (default: localhost:8080)")
        print("  bedrock/<model-id>     - Connect to AWS Bedrock")
        print("  vertex/<model-id>      - Connect to Google Vertex AI (Gemini models)")
        print("  vertex-maas/<model-id> - Connect to Vertex AI Model Garden MaaS\n")
        print("Options:")
        print("  --host <hostname>   - Set hostname for Ollama/llama.cpp (default: localhost)\n")
        print("Examples:")
        print("  python3 run.py --model 'ollama/llama3.2'")
        print("  python3 run.py --model 'ollama/qwen3:8b' --host myserver.local")
        print("  python3 run.py --model 'llama.cpp/my-model' --host 192.168.1.100")
        print("  python3 run.py --model 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'")
        print("  python3 run.py --model 'vertex/gemini-2.0-flash'")
        print("  python3 run.py --model 'vertex-maas/zai-org/glm-4.7-maas'")
        return

    model_name = runner.model

    # Wait for local servers
    if runner.backend_type == "llama.cpp":
        if not wait_for_server(runner.actual_base_url, "llama.cpp server", args.wait_timeout):
            print("\n💡 Tip: Start llama.cpp server with './server -m <model>' in another terminal")
            return
    elif runner.backend_type == "ollama":
        if not wait_for_server(runner.actual_base_url, "Ollama", args.wait_timeout):
            print("\n💡 Tip: Start Ollama with 'ollama serve' in another terminal")
            return
    
    # Load test cases
    test_cases = load_test_cases(args.config, args.test_case)
    
    if not test_cases:
        print(f"No test cases found")
        return
    
    # Setup output
    sanitized = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    Path("results").mkdir(exist_ok=True)
    output_file = f"results/agent_test_results_{sanitized}_{timestamp}.json"
    
    print(f"🚀 Starting Agent Loop Tool Efficiency Test")
    print(f"📊 Configuration:")
    print(f"   Backend: {runner.backend_type}")
    print(f"   Base URL: {runner.actual_base_url}")
    print(f"   Model: {model_name}")
    print(f"   Test Cases: {len(test_cases)}")
    print(f"   Output: {output_file}\n")

    # Run tests (runner already created earlier)
    results = []

    for test_case in test_cases:
        result = run_single_test(runner, test_case)
        results.append(result)
    
    # Generate report
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    total_llm_time = sum(r.response.llm_total_time for r in results if r.response)
    total_requests = sum(r.response.llm_requests for r in results if r.response)
    avg_time = total_llm_time / total_requests if total_requests > 0 else 0
    
    report = AgentReport(
        timestamp=datetime.now(),
        results=results,
        total_tests=len(results),
        passed_tests=passed,
        failed_tests=failed,
        total_llm_time=total_llm_time,
        avg_time_per_req=avg_time
    )
    
    # Save results
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": report.timestamp.isoformat(),
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "total_llm_time": report.total_llm_time,
            "avg_time_per_req": report.avg_time_per_req,
            "results": [
                {
                    "test_case": {
                        "name": r.test_case.name,
                        "prompt": r.test_case.prompt,
                        "expected_tools_variants": [
                            {
                                "name": v.name,
                                "description": v.description,
                                "tools": [
                                    {
                                        "name": t.name,
                                        "arguments": t.arguments
                                    }
                                    for t in v.tools
                                ]
                            }
                            for v in r.test_case.expected_tools_variants
                        ]
                    },
                    "success": r.success,
                    "response_time": r.response_time,
                    "matched_path": r.matched_path,
                    "error_message": r.error_message,
                    "response": {
                        "tool_calls": [{"name": tc.tool_name, "args": tc.arguments} for tc in r.response.tool_calls],
                        "llm_requests": r.response.llm_requests,
                        "llm_total_time": r.response.llm_total_time,
                        "final_message": r.response.final_message
                    } if r.response else None
                }
                for r in results
            ]
        }, f, indent=2)
    
    print_summary(report)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
