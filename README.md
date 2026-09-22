# mcp-har-benchmark

A minimal MCP server that exposes a research benchmark leaderboard to a language model. It is built on the official MCP Python SDK (`MCPServer`).

## The data

`leaderboard.csv` holds 29 training runs from a sleep event classification study on the DreamCatcher earable dataset (classes: quiet, breathe, snore). It comes from [sumeyye-agac/dreamcatcher-earable-wearer-aware-benchmark](https://github.com/sumeyye-agac/dreamcatcher-earable-wearer-aware-benchmark).

The runs cover three architectures: a CRNN teacher (`crnn`, 73,411 parameters) and two small students (`tinycnn`, `tinycnn_cbam`). Some student runs were trained with knowledge distillation from the teacher.

## The point

The leaderboard has 68 columns. Most of them are checkpoint paths, git SHAs and optimizer state. The tools return 15. The whole exercise is choosing which fields a model needs and writing docstrings that state the valid argument values and which direction is better for each metric.

## What the ablation showed

With `list_runs` described as "Get benchmark data.", the generated tool schema is identical to the one produced by the full docstring. Only the description changes. The tool stays just as callable, but a model can no longer pick a valid `model` value, know what the numbers mean, or know which direction is better. The first curation also had its own blind spot. It left out macro recall and balanced accuracy, which hid the fact that four CBAM runs were failed training runs predicting a single class. They showed only as a repeated test F1 of 0.2333 until those two fields were added back. Details in [docs/tool_description_ablation.md](docs/tool_description_ablation.md).

## Tools

- `list_runs(model="", kd_only=False)`: curated fields for matching runs, optionally filtered by architecture or to distillation runs only.
- `best_run(metric="test_f1")`: the single best run by `test_f1` or `test_acc` (higher is better).
- `compare_to_teacher(run_name)`: for a distilled run, its test F1 against the teacher's, the difference in percentage points, and the compression ratio.

## Running it

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mcp dev server.py
```

`mcp dev` opens the MCP Inspector, which needs Node 22.19 or newer. With an older Node it fails to start.
