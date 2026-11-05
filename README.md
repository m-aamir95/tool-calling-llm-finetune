# Tool Calling LLM Fine-Tuning & Evaluation Framework

This repository contains tools, inference pipelines, and an evaluation suite designed to train and benchmark Large Language Models (LLMs) on function and tool calling tasks.

---

## Repository Layout

```text
data/responses/
├── base/       # Base Gemma inference outputs (column-oriented and records JSON)
└── finetuned/  # LoRA fine-tuned Gemma inference outputs
reports/        # Saved evaluation reports
src/
├── evaluation/ # Dataset preparation, inference clients, and evaluation scripts
└── modal_apps/
    ├── base/   # Base-model Modal inference app
    └── finetuned/ # Fine-tuned-model Modal inference app
```

`requirements.txt`, `sample.env`, and the project `README.md` remain at the repository root. Run the commands below from the repository root and pass explicit input paths because datasets and scripts are organized into subdirectories.

---

## 1. Tool Calling Evaluation Plan

Evaluating an LLM's tool calling capability requires more than standard natural language metrics (like ROUGE or BLEU) or basic text matching. A deployed model in an agentic workflow must act as a reliable runtime component: it must adhere to strict serialization protocols, trigger tools conditionally, choose the correct APIs, supply valid argument types and values, and maintain sequence precedence when multiple functions are required.

We evaluate model outputs across **6 core dimensions**:

```
+----------------------------------------------------------------------------------------------------+
|                               6-DIMENSIONAL TOOL CALLING EVALUATION                                 |
+----------------------------------+----------------------------------+------------------------------+
| 1. Syntax & Parsing Integrity    | 2. Invocation Decision           | 3. Tool Selection / Name     |
|   - Tag compliance (<tool_call>) |   - Decision Precision / Recall  |   - Exact tool name match    |
|   - Strict JSON validity         |   - F1 Need-to-Call Score        |   - Tool Set Jaccard index   |
|   - Schema structure validation  |   - Confusion Matrix (TP/FP/FN)  |   - Hallucination rate       |
+----------------------------------+----------------------------------+------------------------------+
| 4. Schema & Argument Matching    | 5. Precedence & Sequencing       | 6. Composite End-to-End      |
|   - Key Precision / Recall / F1  |   - Sequence exact match         |   - Strict E2E Accuracy      |
|   - Argument Type matching       |   - Sequence edit similarity     |   - Functional E2E Accuracy  |
|   - Exact & Normalized values    |   - Tool count match rate        |   - Sequence-only Accuracy   |
+----------------------------------+----------------------------------+------------------------------+
```

---

## 2. Metric Specifications & Mathematical Formulations

### Dimension 1: Syntax & Parsing Integrity
Measures whether the model generates syntactically valid function calls adhering to the Hermes format.
* **Tag Compliance Rate**: Percentage of model responses properly wrapped inside `<tool_call> ... </tool_call>` tags.
* **Strict JSON Parse Rate**: Percentage of payloads parseable by standard RFC 8259 JSON parsers (`json.loads`).
* **Format Error Rate**: Percentage of payloads with unrecoverable syntax errors (e.g. truncated JSON, mismatched brackets).
* **Schema Hallucination Rate**: Detects when the model accidentally outputs tool *definitions* (`{"type": "function", "parameters": ...}`) instead of invoking the tool.

### Dimension 2: Tool Invocation Decision
Evaluates whether the model correctly decides to trigger a tool versus declining or answering in plain text.
```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```
* **True Positive (TP)**: Tool was required by the user prompt, and the model invoked at least one tool.
* **False Positive (FP)**: No tool was needed, but the model called a tool.
* **False Negative (FN)**: Tool was needed, but the model answered in conversational text or failed to call.
* **True Negative (TN)**: No tool was needed, and the model answered directly without tool calls.

### Dimension 3: Tool Selection & Name Matching
Evaluates whether the model identified and selected the correct function names from the available APIs.
* **Exact Tool Name Match Rate**: Binary match indicating whether the set of called tools matches the target set: `S_pred == S_target`.
* **Tool Set Jaccard Similarity**:
  `J(S_pred, S_target) = |S_pred ∩ S_target| / |S_pred ∪ S_target|`
* **Tool Selection Precision & Recall**: Per-query precision and recall are averaged across the dataset (macro averaging), giving each query equal weight. These metrics are not micro-averaged in the current evaluator.
* **Avg Hallucinated Tools per Query**: Mean count of predicted tool names that do not exist in the target set.

### Dimension 4: Schema & Argument Matching
Evaluates whether the arguments passed to functions have the expected keys, parameter types, and values.
* **Argument Key Precision & Recall**:
  - `Key Precision = |K_pred ∩ K_target| / |K_pred|`
  - `Key Recall = |K_pred ∩ K_target| / |K_target|`
* **Argument Type Match Rate**: For matching keys, tests whether the Python / JSON datatypes match (e.g. `list`, `str`, `int`, `dict`), with numerical equivalence between `int` and `float`.
* **Argument Value Exact Match**: Fraction of parameters whose values match identically (`v_pred == v_target`).
* **Argument Value Normalized Match**: Evaluates semantic equivalence by:
  - Trimming leading/trailing whitespace and ignoring case in strings.
  - Normalizing leading periods in file extensions (e.g. `".pdf"` vs `"pdf"`).
  - Comparing list arguments via set equivalence (when item order is commutative).
  - Floating-point tolerance rounding.

### Dimension 5: Tool Precedence & Ordering (Multi-Tool Sequences)
For complex user requests that require multiple functions executed in a specific workflow (e.g., Step 1: fetch data, then Step 2: compute analytics), this dimension checks both the selected tools and their order.
* **Sequence Exact Match Rate**: The predicted ordered sequence must equal the target ordered sequence, including the number of calls. For example, `[fetch, analyze]` matches `[fetch, analyze]`, but not `[analyze, fetch]` or `[fetch]`.
* **Sequence Similarity**: An order-sensitive similarity score calculated using Python's `difflib.SequenceMatcher` ratio (not Levenshtein distance). It gives partial credit when some tool names match in order, even if the full sequences differ.
* **Tool Count Match Rate**: Fraction of samples where `len(predicted_calls) == len(target_calls)`.
* **Tool Count Divergence**: Average of `len(predicted_calls) - len(target_calls)` across samples. A negative value indicates the model tends to omit calls; a positive value indicates it tends to add calls.

For example, if the target is `[fetch, analyze, summarize]` and the model predicts `[fetch, summarize]`, exact sequence match is false, sequence similarity gives partial credit for the ordered overlap, and the tool-count difference is `2 - 3 = -1`.

The evaluator reports these sequence and count metrics over all samples; it also reports a separate sequence-match breakdown for multi-tool samples.

### Dimension 6: Composite End-to-End Success Rates
* **Strict E2E Accuracy**: Response has valid `<tool_call>` tags + strict JSON + exact tool name sequence + all argument keys and values match exactly.
* **Functional E2E Accuracy**: Correct tool sequence + normalized argument match (tolerating formatting variations like markdown blocks).
* **Tool Sequence Accuracy**: Correct tool sequence invoked regardless of parameter values.

---

## 3. Base vs. Fine-Tuned Benchmark Results

We compared the **Google Gemma-2-2B-IT base model** with its **LoRA fine-tuned version** on the same 379 Hermes function-calling examples (269 single-tool and 110 multi-tool queries). The table reports strict-mode results; changes are absolute percentage-point differences.

### Strict-Mode Benchmark Comparison

| Evaluation Metric | Base Model | LoRA Fine-Tuned | Change |
| :--- | :---: | :---: | :---: |
| Tag compliance | 51.98% | **60.69%** | **+8.71 pp** |
| Strict JSON parse rate | 49.34% | **60.69%** | **+11.35 pp** |
| Tool invocation recall | 50.26% | **60.58%** | **+10.32 pp** |
| Tool invocation F1 | 66.78% | **75.33%** | **+8.55 pp** |
| Exact tool-name match | 37.99% | **46.97%** | **+8.97 pp** |
| Tool-set Jaccard similarity | 43.45% | **52.99%** | **+9.54 pp** |
| Argument-key F1 | 38.17% | **51.13%** | **+12.95 pp** |
| Exact argument-value match | 33.07% | **45.27%** | **+12.21 pp** |
| Multi-tool sequence exact match | 45.45% | **65.45%** | **+20.00 pp** |
| Strict end-to-end accuracy | 17.94% | **27.70%** | **+9.76 pp** |
| Functional end-to-end accuracy | 18.47% | **27.97%** | **+9.50 pp** |
| Tool sequence accuracy | 35.88% | **44.33%** | **+8.44 pp** |

Fine-tuning improved every listed strict-mode metric. The largest gains were in multi-tool sequence matching and argument-key F1. Strict end-to-end accuracy increased from 17.94% to 27.70% (about a 54% relative increase), though fewer than one in three examples met the complete strict-success criteria.

**Interpretation caveat:** The fine-tuned responses include a non-standard hyphenated `<tool-call>` tag in some outputs, while the evaluator recognizes the standard `<tool_call>` tag. Those hyphenated outputs are not counted as tool calls in strict mode, so strict results reflect both model capability and adherence to the expected output protocol. In relaxed mode, the base model scores higher on tool invocation recall (82.54% vs. 60.58%), exact tool-name match (67.55% vs. 46.97%), and tool sequence accuracy (64.91% vs. 44.33%), because its alternate formats are more recoverable by the evaluator. Fine-tuning still leads on relaxed functional end-to-end accuracy (27.97% vs. 22.16%) and argument-key F1 (51.13% vs. 45.56%). Interpret both modes with the parser behavior in mind.

---

## 4. Remaining Error Modes & Next Steps

The comparison shows clear gains from fine-tuning, while highlighting areas that still need work:

1. **Output-format consistency:** Fine-tuning eliminated schema-only responses in this evaluation (10.03% for the base model versus 0% fine-tuned), but some fine-tuned outputs use `<tool-call>` instead of the required `<tool_call>` syntax. Standardizing the training targets and reinforcing the exact tag format could improve parser compatibility.
2. **Complete call and argument accuracy:** Fine-tuned strict end-to-end accuracy is 27.70%. Argument-key F1 improved substantially, but the exact argument-value match rate is 45.27%. More targeted examples and validation of required arguments may help.
3. **Multi-tool completeness:** Exact sequence match improved to 65.45% on multi-tool queries, but the model still misses or adds calls in many cases. More examples of complete ordered tool-call sequences may help address this.
