#!/usr/bin/env python3
"""
Batch analysis tool for model test results.

Analyzes test result JSON files to calculate precision, recall, and F1 metrics
for tool invocation and tool selection.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class MetricSet:
    """Represents precision, recall, and F1 metrics."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self):
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class RunMetrics:
    """Metrics for a single run."""
    file_path: str
    tool_invocation: MetricSet
    tool_selection: MetricSet
    average_latency_per_call: float
    test_count: int

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "tool_invocation": self.tool_invocation.to_dict(),
            "tool_selection": self.tool_selection.to_dict(),
            "average_latency_per_call": self.average_latency_per_call,
            "test_count": self.test_count,
        }


@dataclass
class ModelAnalysis:
    """Analysis results for a single model."""
    model_name: str
    batch_source: str
    tool_invocation: MetricSet
    tool_selection: MetricSet
    average_latency_per_call: float
    total_tests: int
    unique_tests: int
    total_runs: int
    result_files: List[str]
    per_run_metrics: List[RunMetrics]

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "batch_source": self.batch_source,
            "tool_invocation": self.tool_invocation.to_dict(),
            "tool_selection": self.tool_selection.to_dict(),
            "average_latency_per_call": self.average_latency_per_call,
            "total_tests": self.total_tests,
            "unique_tests": self.unique_tests,
            "total_runs": self.total_runs,
            "result_files": self.result_files,
            "per_run_metrics": [r.to_dict() for r in self.per_run_metrics],
        }


@dataclass
class BatchAnalysisReport:
    """Complete analysis report."""
    batch_directories: List[str]
    analysis_date: datetime
    models: List[ModelAnalysis]
    summary: str

    def to_dict(self):
        return {
            "batch_directories": self.batch_directories,
            "analysis_date": self.analysis_date.isoformat(),
            "models": [m.to_dict() for m in self.models],
            "summary": self.summary,
        }


def find_result_files(directory: str) -> List[str]:
    """Find all agent test result files in the directory."""
    pattern = re.compile(r'^agent_test_results_.*\.json$')
    result_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                result_files.append(os.path.join(root, file))

    return result_files


def group_files_by_model(files: List[str], batch_dirs: List[str]) -> Dict[str, Dict]:
    """Group result files by model name and track batch source."""
    model_files = {}
    # Pattern to extract model name: agent_test_results_{model}_{timestamp}.json
    pattern = re.compile(r'^agent_test_results_(.+?)_\d{8}_\d{6}\.json$')

    for file in files:
        basename = os.path.basename(file)
        matches = pattern.match(basename)

        if matches:
            model_name = matches.group(1)
        else:
            # Fallback: try to extract model name between "agent_test_results_" and timestamp
            if basename.startswith("agent_test_results_"):
                parts = basename.replace("agent_test_results_", "").split('_')
                # Take all parts before what looks like a timestamp (8 digits)
                model_parts = []
                for part in parts:
                    if re.match(r'^\d{8}$', part):
                        break
                    model_parts.append(part)
                model_name = "_".join(model_parts) if model_parts else "unknown"
            else:
                model_name = "unknown"

        # Determine batch source
        batch_source = "unknown"
        for batch_dir in batch_dirs:
            if file.startswith(batch_dir):
                batch_source = batch_dir
                break

        # Update model files info
        if model_name not in model_files:
            model_files[model_name] = {
                "files": [],
                "batch_source": batch_source
            }
        else:
            # Combine batch sources if model appears in multiple batches
            if model_files[model_name]["batch_source"] != batch_source:
                existing = model_files[model_name]["batch_source"]
                if batch_source not in existing:
                    model_files[model_name]["batch_source"] = f"{existing},{batch_source}"

        model_files[model_name]["files"].append(file)

    return model_files


def load_result_file(filename: str) -> List[Dict]:
    """Load test results from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    # Handle both old format (direct results array) and new format (report object)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        return []


def should_call_any_tool(test_case: Dict) -> bool:
    """Determine if any tool should be called for a test case."""
    variants = test_case.get("expected_tools_variants", [])
    for variant in variants:
        if len(variant.get("tools", [])) > 0:
            return True
    return False


def get_expected_tools(test_case: Dict) -> List[str]:
    """Get all expected tool names from all variants."""
    tools = []
    variants = test_case.get("expected_tools_variants", [])
    for variant in variants:
        for tool in variant.get("tools", []):
            tools.append(tool["name"])
    return tools


def get_actual_tools(response: Optional[Dict]) -> List[str]:
    """Get all actual tool names called."""
    if not response:
        return []

    tool_calls = response.get("tool_calls", [])
    # Handle both formats: direct "name" or "tool_name" field
    return [tc.get("name", tc.get("tool_name", "")) for tc in tool_calls]


def calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> MetricSet:
    """Calculate precision, recall, and F1 from confusion matrix values."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return MetricSet(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def calculate_tool_invocation_metrics(results: List[Dict]) -> MetricSet:
    """Calculate binary tool invocation metrics."""
    tp = fp = tn = fn = 0

    for result in results:
        test_case = result.get("test_case", {})

        # Handle both old format (response is dict) and new format (response nested)
        response = result.get("response")

        should_call = should_call_any_tool(test_case)

        # Handle missing response or check tool_calls
        if response:
            tool_calls = response.get("tool_calls", [])
            did_call = len(tool_calls) > 0
        else:
            did_call = False

        if should_call and did_call:
            tp += 1  # Should call and did call
        elif not should_call and not did_call:
            tn += 1  # Should not call and did not call
        elif not should_call and did_call:
            fp += 1  # Should not call but did call
        else:
            fn += 1  # Should call but did not call

    return calculate_metrics(tp, fp, tn, fn)


def get_best_matching_variant(test_case: Dict, actual_tools: List[str]) -> List[str]:
    """Find the expected variant that best matches the actual tools called.

    This uses order-independent matching to find the variant with the most
    overlap, giving the model the benefit of the doubt when multiple valid
    tool sequences exist.

    Returns the expected tool names from the best matching variant.
    If no variants exist, returns an empty list.
    """
    variants = test_case.get("expected_tools_variants", [])
    if not variants:
        return []

    # Find the variant with the most overlap with actual tools
    best_variant_tools = []
    best_match_count = -1

    for variant in variants:
        expected = [tool["name"] for tool in variant.get("tools", [])]
        # Count how many expected tools appear in actual (order-independent)
        match_count = sum(1 for tool in expected if tool in actual_tools)
        if match_count > best_match_count:
            best_match_count = match_count
            best_variant_tools = expected

    return best_variant_tools


def calculate_tool_selection_metrics(results: List[Dict]) -> MetricSet:
    """Calculate tool selection metrics at the individual tool call level.

    NOT ALL-OR-NOTHING: We deliberately avoid requiring perfect predictions.
    Each correct tool call gets credit, even if the full sequence isn't perfect.
    We focus on whether the tool selection makes sense for the intent, not
    exact parameter matches or strict ordering.

    Per Docker's methodology:
    - Precision: how often the model made valid tool calls (valid calls / all calls made)
    - Recall: how often it made the tool calls it was supposed to (correct calls / expected calls)
    - F1: harmonic mean of precision and recall

    Counting at the individual tool level (not test case level):
    - TP: A tool that was expected AND was called (by name, ignoring parameters)
    - FP: A tool that was called but NOT expected
    - FN: A tool that was expected but NOT called

    Order-independent matching: If expected is [A, B] and actual is [B, A],
    both tools are counted as correct (2 TP), not penalized for ordering.
    """
    tp = fp = tn = fn = 0

    for result in results:
        test_case = result.get("test_case", {})
        response = result.get("response")

        actual_tools = get_actual_tools(response)

        # Get the best matching variant's expected tools
        expected_tools = get_best_matching_variant(test_case, actual_tools)

        if len(expected_tools) == 0 and len(actual_tools) == 0:
            # No tools expected and none called - true negative (at test level)
            tn += 1
            continue

        # Count tool-level metrics with partial credit
        # Only compare tool names, not parameters (we don't demand exact product names, etc.)
        expected_remaining = list(expected_tools)

        for actual_tool in actual_tools:
            if actual_tool in expected_remaining:
                tp += 1  # Tool was expected and called - partial credit given
                expected_remaining.remove(actual_tool)
            else:
                fp += 1  # Tool was called but not expected

        # Any remaining expected tools that weren't called are false negatives
        fn += len(expected_remaining)

    return calculate_metrics(tp, fp, tn, fn)


def calculate_average_latency_per_llm_call(results: List[Dict]) -> float:
    """Calculate average latency per LLM call in seconds.

    This computes total LLM time divided by total LLM requests across all
    successful test results, giving the true average latency per API call.
    """
    if not results:
        return 0.0

    total_llm_time = 0.0
    total_llm_requests = 0

    for r in results:
        response = r.get("response")
        if response:  # Only count successful responses
            total_llm_time += response.get("llm_total_time", 0.0)
            total_llm_requests += response.get("llm_requests", 0)

    if total_llm_requests == 0:
        return 0.0

    return total_llm_time / total_llm_requests


def average_metric_sets(metric_sets: List[MetricSet]) -> MetricSet:
    """Average multiple MetricSets using macro-averaging.

    Computes the mean of precision, recall, and F1 across all runs,
    and sums the confusion matrix counts.
    """
    if not metric_sets:
        return MetricSet(0.0, 0.0, 0.0, 0, 0, 0, 0)

    n = len(metric_sets)
    avg_precision = sum(m.precision for m in metric_sets) / n
    avg_recall = sum(m.recall for m in metric_sets) / n
    avg_f1 = sum(m.f1 for m in metric_sets) / n

    # Sum confusion matrix counts (for reference/transparency)
    total_tp = sum(m.true_positives for m in metric_sets)
    total_fp = sum(m.false_positives for m in metric_sets)
    total_tn = sum(m.true_negatives for m in metric_sets)
    total_fn = sum(m.false_negatives for m in metric_sets)

    return MetricSet(
        precision=avg_precision,
        recall=avg_recall,
        f1=avg_f1,
        true_positives=total_tp,
        false_positives=total_fp,
        true_negatives=total_tn,
        false_negatives=total_fn,
    )


def analyze_model(model_name: str, files: List[str], batch_source: str) -> ModelAnalysis:
    """Analyze all result files for a single model using macro-averaging.

    Calculates metrics for each run separately, then averages across runs.
    This ensures each run is weighted equally regardless of test count.
    """
    per_run_metrics = []
    all_test_ids = set()
    total_tests = 0

    # Calculate metrics for each run separately
    for file in files:
        results = load_result_file(file)
        if not results:
            continue

        # Track unique test cases by their ID or name
        for result in results:
            test_case = result.get("test_case", {})
            test_id = test_case.get("id", test_case.get("name", ""))
            if test_id:
                all_test_ids.add(test_id)

        tool_invocation = calculate_tool_invocation_metrics(results)
        tool_selection = calculate_tool_selection_metrics(results)
        avg_latency = calculate_average_latency_per_llm_call(results)

        run_metrics = RunMetrics(
            file_path=file,
            tool_invocation=tool_invocation,
            tool_selection=tool_selection,
            average_latency_per_call=avg_latency,
            test_count=len(results),
        )
        per_run_metrics.append(run_metrics)
        total_tests += len(results)

    if not per_run_metrics:
        raise ValueError(f"No test results found for model {model_name}")

    # Macro-average metrics across runs
    avg_tool_invocation = average_metric_sets([r.tool_invocation for r in per_run_metrics])
    avg_tool_selection = average_metric_sets([r.tool_selection for r in per_run_metrics])
    avg_latency = sum(r.average_latency_per_call for r in per_run_metrics) / len(per_run_metrics)

    return ModelAnalysis(
        model_name=model_name,
        batch_source=batch_source,
        tool_invocation=avg_tool_invocation,
        tool_selection=avg_tool_selection,
        average_latency_per_call=avg_latency,
        total_tests=total_tests,
        unique_tests=len(all_test_ids) if all_test_ids else total_tests // len(files),
        total_runs=len(files),
        result_files=files,
        per_run_metrics=per_run_metrics,
    )


def analyze_batches(batch_dirs: List[str]) -> BatchAnalysisReport:
    """Analyze all result files across multiple batch directories."""
    all_result_files = []

    # Collect all result files
    for batch_dir in batch_dirs:
        result_files = find_result_files(batch_dir)
        all_result_files.extend(result_files)

    if not all_result_files:
        raise ValueError(f"No result files found in directories: {batch_dirs}")

    # Group files by model
    model_files = group_files_by_model(all_result_files, batch_dirs)

    # Analyze each model
    models = []
    for model_name, info in model_files.items():
        try:
            analysis = analyze_model(model_name, info["files"], info["batch_source"])
            models.append(analysis)
        except Exception as e:
            print(f"Warning: failed to analyze model {model_name}: {e}", file=sys.stderr)
            continue

    # Sort by F1 score (tool selection) descending
    models.sort(key=lambda m: m.tool_selection.f1, reverse=True)

    report = BatchAnalysisReport(
        batch_directories=batch_dirs,
        analysis_date=datetime.now(),
        models=models,
        summary=generate_summary(models),
    )

    return report


def generate_summary(models: List[ModelAnalysis]) -> str:
    """Generate a summary of the analysis."""
    if not models:
        return "No models analyzed."

    lines = ["Summary:", "--------"]

    if len(models) == 1:
        model = models[0]
        lines.append(f"Analyzed 1 model ({model.model_name}) with {model.unique_tests} unique tests across {model.total_runs} runs.")
    else:
        total_runs = sum(m.total_runs for m in models)
        best = models[0]
        lines.append(f"Analyzed {len(models)} models across {total_runs} total runs.")
        lines.append(f"Best performing model: {best.model_name} (Tool Selection F1: {best.tool_selection.f1:.3f})")

    return "\n".join(lines)


def generate_text_report(report: BatchAnalysisReport) -> str:
    """Generate a human-readable text report."""
    lines = [
        "Batch Analysis Report",
        "=====================",
        f"Batch Directories: {', '.join(report.batch_directories)}",
        f"Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Model Performance Summary:",
        "--------------------------",
    ]

    for model in report.models:
        lines.append(f"{model.model_name}:")
        if model.batch_source:
            lines.append(f"  Batch Source: {model.batch_source}")
        lines.append(f"  Runs: {model.total_runs}, Unique Tests: {model.unique_tests}")
        lines.append(f"  Average Latency per LLM Call: {model.average_latency_per_call:.2f}s")
        if model.total_runs > 1:
            lines.append("  Tool Invocation (Binary, macro-averaged):")
        else:
            lines.append("  Tool Invocation (Binary):")
        lines.append(f"    Precision: {model.tool_invocation.precision:.3f} "
                    f"({model.tool_invocation.true_positives}/"
                    f"{model.tool_invocation.true_positives + model.tool_invocation.false_positives})")
        lines.append(f"    Recall: {model.tool_invocation.recall:.3f} "
                    f"({model.tool_invocation.true_positives}/"
                    f"{model.tool_invocation.true_positives + model.tool_invocation.false_negatives})")
        lines.append(f"    F1: {model.tool_invocation.f1:.3f}")
        if model.total_runs > 1:
            lines.append("  Tool Selection (macro-averaged):")
        else:
            lines.append("  Tool Selection:")
        lines.append(f"    Precision: {model.tool_selection.precision:.3f} "
                    f"({model.tool_selection.true_positives}/"
                    f"{model.tool_selection.true_positives + model.tool_selection.false_positives})")
        lines.append(f"    Recall: {model.tool_selection.recall:.3f} "
                    f"({model.tool_selection.true_positives}/"
                    f"{model.tool_selection.true_positives + model.tool_selection.false_negatives})")
        lines.append(f"    F1: {model.tool_selection.f1:.3f}")
        # Show per-run breakdown for multiple runs
        if model.total_runs > 1:
            lines.append("  Per-run F1 scores:")
            for i, run in enumerate(model.per_run_metrics, 1):
                run_file = os.path.basename(run.file_path)
                lines.append(f"    Run {i}: Invocation={run.tool_invocation.f1:.3f}, "
                           f"Selection={run.tool_selection.f1:.3f} ({run_file})")
        lines.append("")

    if len(report.models) > 1:
        lines.append("Overall Rankings (by Tool Selection F1):")
        lines.append("-----------------------------------------")
        for i, model in enumerate(report.models, 1):
            lines.append(f"{i}. {model.model_name} (F1: {model.tool_selection.f1:.3f}, Latency: {model.average_latency_per_call:.2f}s)")
        lines.append("")

    lines.append(report.summary)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze one or more batch directories of test results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "batch_dirs",
        nargs="+",
        help="One or more batch directories to analyze (multiple directories treated as combined batch)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Validate batch directories
    for batch_dir in args.batch_dirs:
        if not os.path.exists(batch_dir):
            print(f"Error: Batch directory does not exist: {batch_dir}", file=sys.stderr)
            sys.exit(1)

    # Analyze batches
    try:
        report = analyze_batches(args.batch_dirs)
    except Exception as e:
        print(f"Error: Failed to analyze batches: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate output
    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2)
    else:
        output = generate_text_report(report)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Analysis report written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
