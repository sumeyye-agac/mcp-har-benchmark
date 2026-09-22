# Tool description ablation: `list_runs`

This compares two docstrings for the same `list_runs` tool in [server.py](../server.py):

- **A (current):** names the three valid `model` values, explains `kd_only`, lists each of the 15 returned fields (out of the leaderboard's 68 columns) and when it shows up, and states which direction is better for each metric.
- **B (vague):** `"Get benchmark data."`

The function signature, body and return type stay the same. Only the docstring changes.

Everything below comes from what the SDK actually generates (`mcp` 2.2.0, `MCPServer`) and from what each docstring does or does not say. No model was run, and there is no model transcript here.

## 1. The schema is identical

Both versions were registered on an `MCPServer` and read back with `list_tools()`. Field by field:

| Tool field     | A vs B        |
|----------------|---------------|
| `name`         | identical     |
| `inputSchema`  | identical     |
| `outputSchema` | identical     |
| `description`  | **differs**   |

Side by side, as the SDK sends them:

<table>
<tr><th>A (current docstring)</th><th>B ("Get benchmark data.")</th></tr>
<tr><td>

```json
"inputSchema": {
  "properties": {
    "model": {
      "default": "",
      "title": "Model",
      "type": "string"
    },
    "kd_only": {
      "default": false,
      "title": "Kd Only",
      "type": "boolean"
    }
  },
  "title": "list_runsArguments",
  "type": "object"
},
"outputSchema": {
  "properties": {
    "result": {
      "items": {
        "additionalProperties": true,
        "type": "object"
      },
      "title": "Result",
      "type": "array"
    }
  },
  "required": ["result"],
  "title": "list_runsOutput",
  "type": "object"
}
```

</td><td>

```json
"inputSchema": {
  "properties": {
    "model": {
      "default": "",
      "title": "Model",
      "type": "string"
    },
    "kd_only": {
      "default": false,
      "title": "Kd Only",
      "type": "boolean"
    }
  },
  "title": "list_runsArguments",
  "type": "object"
},
"outputSchema": {
  "properties": {
    "result": {
      "items": {
        "additionalProperties": true,
        "type": "object"
      },
      "title": "Result",
      "type": "array"
    }
  },
  "required": ["result"],
  "title": "list_runsOutput",
  "type": "object"
}
```

</td></tr>
<tr><td>

`"description"`: the full version A docstring: 3 valid `model` values, meaning of `kd_only`, the 15 returned fields and which runs have them, and the direction of better for each metric.

</td><td>

`"description": "Get benchmark data."`

</td></tr>
</table>

**So the tool is equally callable and no longer equally usable.** Under both versions, a client can build a request that passes schema validation, because the schema accepts any string for `model` and any boolean for `kd_only`. What goes away under B is the information needed to pick arguments that mean something and to read the result.

The schema can't carry that information by itself. `model` is typed as a plain `str` with no enum, and the output is typed as a list of objects with `additionalProperties: true`, which names no fields. Everything a model knows about valid values and field meanings comes from the description.

## 2. What a calling model can and cannot know

| Question | A (current) | B ("Get benchmark data.") |
|---|---|---|
| **Can it produce a valid `model` value?** | Yes. The docstring says `model` "must be exactly one of" `"crnn"`, `"tinycnn"`, `"tinycnn_cbam"`, or `""` for all. | No. The schema says `string`, default `""`. The only valid value it can infer is the default `""` (no filter). Any other value is a guess. |
| **Does it know what `kd_only` does?** | Yes. It returns only distillation runs (runs with a teacher, plus `alpha`/`tau`). The docstring also says distillation is **not** a `model` value. | No. It only sees a boolean titled "Kd Only". It could guess "knowledge distillation" from the name, but nothing confirms it. Nothing warns it that `model="kd"` is invalid. |
| **Does it know what the returned numbers mean?** | Mostly. `test_f1` (macro F1), `test_acc` (accuracy), `test_recall_macro` (macro recall) and `test_balanced_acc` (balanced accuracy) are all 0–1. It knows which fields apply only to distillation runs (`teacher_model`, `alpha`, `tau`, `compression_ratio`) and which only to CBAM runs (`cbam_reduction`, `cbam_sa_kernel`), so it knows why they're missing elsewhere. The units of `model_size_mb` and `cpu_latency_ms` come only from the field names. | No. The output schema names no fields. It sees keys like `test_f1` or `alpha` only after calling the tool, and has to guess from the names what they measure, what scale they use, and why some runs lack them. |
| **Does it know whether higher is better?** | Yes. Higher is better for `test_f1`, `test_acc`, `test_recall_macro`, `test_balanced_acc` and `compression_ratio`. Lower is better for `params`, `model_size_mb` and `cpu_latency_ms`. | No. |
| **Does it know the scope of the data?** | Yes: sleep-event classification on DreamCatcher, 29 runs, 3 classes. | No. "Benchmark" is all it gets. |

The first draft of version A had a gap here. It named F1 and accuracy but never said which direction was better, and said nothing about latency, size or compression. The explicit statement was only in `best_run`'s docstring. Version A now has two lines stating the direction of better for every metric field.

## 3. The error message is part of the contract

The description tells the model what to send. The error message is the only feedback it gets when the value it sent is wrong. Both variants below reject `model="kd"` with the same text. The only difference is the exception class. Both were called through an in-process MCP `Client`, so the results below are what a client receives over the protocol:

| Raised in the tool | `isError` | Text the client receives |
|---|---|---|
| `ToolError("Unknown model 'kd'. Valid values: ...")` (current [server.py](../server.py)) | `true` | `Error executing tool list_runs: Unknown model 'kd'. Valid values: crnn, tinycnn, tinycnn_cbam, or empty for all.` |
| `ValueError("Unknown model 'kd'. Valid values: ...")` | `true` | `Error executing tool list_runs_plain` |

The SDK treats `ToolError` as a deliberate, caller-facing error and sends its message. It treats any other exception as a crash (`UnexpectedToolError`). The message and traceback go only to the server log, and the client gets the generic text.

That affects the two versions differently:

- **With B plus `ToolError`**, a wrong guess still gets corrected. The error message delivers the valid values the description left out, one failed call later.
- **With B plus a plain exception**, a wrong guess gets "Error executing tool" and nothing else. It can't tell a bad argument from a server bug, and nothing points it to a valid value. Only `""` is known to work.
- **With A**, the error path matters less for `model`, because the valid values are stated up front. It still catches typos and wrong casing.

## 4. The four collapsed CBAM runs

Four `tinycnn_cbam` runs are **failed training runs kept in the leaderboard rather than removed**:

| run_name | cbam_reduction | cbam_sa_kernel |
|---|---|---|
| `p1_tinycnn_cbam_rr4_sk11_seed42`  | 4  | 11 |
| `p1_tinycnn_cbam_rr8_sk11_seed42`  | 8  | 11 |
| `p1_tinycnn_cbam_rr16_sk7_seed42`  | 16 | 7  |
| `p1_tinycnn_cbam_rr16_sk11_seed42` | 16 | 11 |

All four have exactly the same metrics in `leaderboard.csv`: `test_f1` 0.233322, `test_acc` 0.53842, `test_recall_macro` and `test_balanced_acc` 0.333333, `best_epoch` 1, early-stopped at epoch 6. A macro recall of exactly 1/3 across three classes means the model predicts one class for every test sample. The F1 matches that: if one class makes up 53.842% of the test set, predicting only that class gives that class F1 = 2·0.53842 / 1.53842 ≈ 0.700 and the other two classes 0, so macro F1 ≈ 0.700 / 3 = 0.2333.

Why this matters for the tools:

- `list_runs` returns these runs like any other. Nothing in the returned fields marks them as failed, and neither docstring mentions them. A model averaging `test_f1` over `tinycnn_cbam` runs, or reasoning about which reduction/kernel settings help, will be pulled down by four runs that learned nothing.
- `best_run` isn't affected, since it takes the maximum.
- The first curation (13 fields) left out `test_recall_macro` and `test_balanced_acc`, the two fields that show the collapse most directly. With only `test_f1` and `test_acc`, a model saw a repeated 0.2333 and had no way to tell why. Both fields are now in the curated set (15 fields), so a collapsed run shows macro recall and balanced accuracy of 0.333333. That tells the model it predicted a single class. The tools still don't flag these runs as failed, and neither docstring mentions them.

## Not yet verified

The MCP Inspector could not be launched on this machine. The only Node installed is v18.16.0 (from anaconda), and `mcp dev` pulls Inspector 2.7.0, which needs Node 22.19 or newer. Installing Node with Homebrew failed with a download error (`Could not resolve host: github.com`). The tools were exercised directly through the SDK instead: `MCPServer.list_tools()` for the schemas, and an in-process `mcp.client.Client` for the calls and error responses above. Checking the tool by hand in the Inspector, with both docstring versions, has not been done.
