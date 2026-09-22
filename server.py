"""MCP server over the DreamCatcher earable sleep-event benchmark leaderboard.

The leaderboard (leaderboard.csv) has 68 columns, most of which are artifact
paths, git SHAs and optimizer state. The tools here expose only the fields a
model needs to reason about the results.
"""

import csv
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

CSV_PATH = Path(__file__).parent / "leaderboard.csv"

VALID_MODELS = ("crnn", "tinycnn", "tinycnn_cbam")
VALID_METRICS = ("test_f1", "test_acc")

INT_FIELDS = ("params", "cbam_reduction", "cbam_sa_kernel")
FLOAT_FIELDS = (
    "test_f1",
    "test_acc",
    "test_recall_macro",
    "test_balanced_acc",
    "compression_ratio",
    "model_size_mb",
    "cpu_latency_ms",
    "alpha",
    "tau",
)
CURATED_FIELDS = (
    "run_name",
    "model",
    "teacher_model",
    "test_f1",
    "test_acc",
    "test_recall_macro",
    "test_balanced_acc",
    "params",
    "compression_ratio",
    "model_size_mb",
    "cpu_latency_ms",
    "alpha",
    "tau",
    "cbam_reduction",
    "cbam_sa_kernel",
)


def _parse(field: str, raw: str) -> Any:
    """Convert a CSV cell to a typed value, or None if the cell is empty."""
    raw = raw.strip()
    if raw == "":
        return None
    if field in INT_FIELDS:
        return int(float(raw))
    if field in FLOAT_FIELDS:
        return float(raw)
    return raw


def _load_runs() -> list[dict[str, Any]]:
    """Read the leaderboard and keep only curated, non-empty fields."""
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    runs = []
    for row in rows:
        run = {}
        for field in CURATED_FIELDS:
            value = _parse(field, row.get(field, ""))
            if value is not None:
                run[field] = value
        runs.append(run)
    return runs


RUNS = _load_runs()


def _is_kd(run: dict[str, Any]) -> bool:
    return bool(run.get("teacher_model"))


def _find_teacher(teacher_model: str) -> dict[str, Any]:
    """The teacher is the non-distilled run of the teacher architecture."""
    for run in RUNS:
        if run["model"] == teacher_model and not _is_kd(run):
            return run
    raise ToolError(f"No baseline run found for teacher model '{teacher_model}'.")


mcp = MCPServer("HARBenchmark")


@mcp.tool()
def list_runs(model: str = "", kd_only: bool = False) -> list[dict[str, Any]]:
    """List training runs from the DreamCatcher earable sleep-event benchmark.

    The benchmark classifies sleep audio events into three classes (quiet,
    breathe, snore). There are 29 runs in total. Each returned run is a dict
    of curated fields; fields that do not apply to a run are omitted:
      - run_name, model, params, model_size_mb, cpu_latency_ms: every run
      - test_f1 (macro F1), test_acc (accuracy), test_recall_macro (macro
        recall), test_balanced_acc (balanced accuracy), all 0-1: every run
      - teacher_model, alpha, tau, compression_ratio: distillation runs only
      - cbam_reduction, cbam_sa_kernel: tinycnn_cbam runs only
    Higher is better: test_f1, test_acc, test_recall_macro, test_balanced_acc,
    compression_ratio. Lower is better: params, model_size_mb, cpu_latency_ms.

    Args:
        model: Filter by architecture. Must be exactly one of:
            "crnn" (the teacher, 73,411 params, a single baseline run),
            "tinycnn" (small student CNN),
            "tinycnn_cbam" (small student CNN with CBAM attention).
            Leave empty ("") to include all architectures.
        kd_only: If true, return only knowledge distillation runs, i.e. runs
            trained with a teacher (teacher_model is set, alpha and tau are
            present). Distillation is not a separate model value: a distilled
            student still has model "tinycnn" or "tinycnn_cbam". Combine with
            model to get e.g. only distilled tinycnn_cbam runs.
    """
    if model and model not in VALID_MODELS:
        raise ToolError(
            f"Unknown model '{model}'. Valid values: {', '.join(VALID_MODELS)}, or empty for all."
        )
    return [
        run
        for run in RUNS
        if (not model or run["model"] == model) and (not kd_only or _is_kd(run))
    ]


@mcp.tool()
def best_run(metric: str = "test_f1") -> dict[str, Any]:
    """Return the single best run across all 29 runs by a test-set metric.

    Higher is better for every allowed metric, so this returns the run with
    the maximum value. Ties are broken by leaderboard order. The result has
    the same curated fields as list_runs.

    Args:
        metric: Must be exactly one of:
            "test_f1" (macro F1 on the test set, 0-1, default),
            "test_acc" (accuracy on the test set, 0-1).
    """
    if metric not in VALID_METRICS:
        raise ToolError(
            f"Unknown metric '{metric}'. Valid values: {', '.join(VALID_METRICS)}."
        )
    return max(RUNS, key=lambda run: run[metric])


@mcp.tool()
def compare_to_teacher(run_name: str) -> dict[str, Any]:
    """Compare a knowledge distillation run to its teacher.

    Only applies to runs trained with distillation (runs where teacher_model
    is set; list_runs with kd_only=true returns them). Calling it on the
    teacher itself or on a student trained without distillation is an error.

    Returns:
        run_name, model, teacher_model,
        student_test_f1 and teacher_test_f1 (macro F1, 0-1),
        f1_diff_pp: student minus teacher test F1 in percentage points
            (negative means the student is worse than the teacher),
        compression_ratio: teacher params / student params (e.g. 3.1 means
            the student is 3.1x smaller).

    Args:
        run_name: Exact run_name of a distillation run as returned by
            list_runs, e.g. "p2_kd_tinycnn_a0p3_t3_seed42".
    """
    run = next((r for r in RUNS if r["run_name"] == run_name), None)
    if run is None:
        raise ToolError(
            f"No run named '{run_name}'. Use list_runs(kd_only=True) to see distillation runs."
        )
    if not _is_kd(run):
        raise ToolError(
            f"Run '{run_name}' was not trained with distillation (no teacher_model). "
            "Use list_runs(kd_only=True) to see distillation runs."
        )
    teacher = _find_teacher(run["teacher_model"])
    return {
        "run_name": run["run_name"],
        "model": run["model"],
        "teacher_model": run["teacher_model"],
        "student_test_f1": run["test_f1"],
        "teacher_test_f1": teacher["test_f1"],
        "f1_diff_pp": round((run["test_f1"] - teacher["test_f1"]) * 100, 2),
        "compression_ratio": run["compression_ratio"],
    }


if __name__ == "__main__":
    mcp.run()
