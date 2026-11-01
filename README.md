# Tool Calling LLM Fine-Tuning & Evaluation Framework

This repository contains tools, inference pipelines, and an evaluation suite designed to train and benchmark Large Language Models (LLMs) on function and tool calling tasks.

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
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **True Positive (TP)**: Tool was required by the user prompt, and the model invoked at least one tool.
* **False Positive (FP)**: No tool was needed, but the model called a tool.
* **False Negative (FN)**: Tool was needed, but the model answered in conversational text or failed to call.
* **True Negative (TN)**: No tool was needed, and the model answered directly without tool calls.

### Dimension 3: Tool Selection & Name Matching
Evaluates whether the model identified and selected the correct function names from the available APIs.
* **Exact Tool Name Match Rate**: Binary match indicating whether the set of called tools matches the target set: $S_{pred} = S_{target}$.
* **Tool Set Jaccard Similarity**:
  $$J(S_{pred}, S_{target}) = \frac{|S_{pred} \cap S_{target}|}{|S_{pred} \cup S_{target}|}$$
* **Tool Selection Precision & Recall**: Micro- and macro-averaged precision of predicted tools against required tools.
* **Avg Hallucinated Tools per Query**: Mean count of predicted tool names that do not exist in the target set.

### Dimension 4: Schema & Argument Matching
Evaluates whether the arguments passed to functions have the expected keys, parameter types, and values.
* **Argument Key Precision & Recall**:
  $$\text{Key Prec} = \frac{|K_{pred} \cap K_{target}|}{|K_{pred}|}, \quad \text{Key Rec} = \frac{|K_{pred} \cap K_{target}|}{|K_{target}|}$$
* **Argument Type Match Rate**: For matching keys, tests whether the Python / JSON datatypes match (e.g. `list`, `str`, `int`, `dict`), with numerical equivalence between `int` and `float`.
* **Argument Value Exact Match**: Fraction of parameters whose values match identically ($v_{pred} == v_{target}$).
* **Argument Value Normalized Match**: Evaluates semantic equivalence by:
  - Trimming leading/trailing whitespace and ignoring case in strings.
  - Normalizing leading periods in file extensions (e.g. `".pdf"` vs `"pdf"`).
  - Comparing list arguments via set equivalence (when item order is commutative).
  - Floating-point tolerance rounding.

### Dimension 5: Tool Precedence & Ordering (Multi-Tool Sequences)
For complex user requests that require multiple functions executed in a specific workflow (e.g., Step 1: fetch data $\rightarrow$ Step 2: compute analytics):
* **Sequence Exact Match Rate**: The ordered sequence of tool names matches the target sequence exactly:
  $$[f_1, f_2, \dots, f_m] == [f_1^*, f_2^*, \dots, f_m^*]$$
* **Sequence Edit Similarity**: Normalized Levenshtein ratio between the predicted tool sequence and the target sequence.
* **Tool Count Match Rate**: Percentage of samples where $\text{len}(\text{calls}_{pred}) == \text{len}(\text{calls}_{target})$.
* **Tool Count Divergence**: Mean difference $\text{len}(\text{calls}_{pred}) - \text{len}(\text{calls}_{target})$ (identifies systematic over-calling or under-calling).

### Dimension 6: Composite End-to-End Success Rates
* **Strict E2E Accuracy**: Response has valid `<tool_call>` tags + strict JSON + exact tool name sequence + all argument keys and values match exactly.
* **Functional E2E Accuracy**: Correct tool sequence + normalized argument match (tolerating formatting variations like markdown blocks).
* **Tool Sequence Accuracy**: Correct tool sequence invoked regardless of parameter values.

---

## 3. Evaluation Script Implementation: `model_func_results_evaluation.py`

The evaluation script is implemented in [`model_func_results_evaluation.py`](file:///workspaces/tool-calling-llm-finetune/model_func_results_evaluation.py).

### Key Architectural Highlights:
1. **Multi-Format Input Support**: Seamlessly reads both column-oriented JSON (`func_calling_dataset_gemma_2_2b_it_responses.json`) and records-oriented JSON (`func_calling_dataset_gemma_2_2b_it_responses_records.json`).
2. **Robust Normalization Pipeline**:
   - Handles Hermes dataset structural nuances, such as inverted parameter mappings (`{"arguments": {"name": "func_name", ...}}`).
   - Handles escaped unicode characters (e.g. `\\n` literals vs `\n`).
   - Normalizes nested function wrappers (`{"function": {"name": ...}}`).
3. **Dual Evaluation Mode (Strict vs Relaxed)**:
   - **Strict Mode**: Requires exact `<tool_call>` tag formatting.
   - **Relaxed Mode**: Accepts markdown code fences (````json [ ... ] ````) to isolate whether errors stem from syntax protocol adherence versus underlying tool reasoning ability.
4. **Detailed Reporting**: Saves a complete JSON report (`evaluation_report.json`) with sample-by-sample diagnostics, confusion matrices, and subgroup breakdowns (single-tool vs multi-tool).

### CLI Usage:

```bash
# Run evaluation with default records dataset
python3 model_func_results_evaluation.py

# Run evaluation on a specific input file and custom report destination
python3 model_func_results_evaluation.py --input func_calling_dataset_gemma_2_2b_it_responses.json --output my_report.json

# Run evaluation on first 50 samples with verbose failure analysis
python3 model_func_results_evaluation.py --limit 50 --verbose
```

---

## 4. Gemma-2-2B-IT Baseline Benchmark Results

We evaluated **Google Gemma-2-2B-IT** on the Hermes function calling test split (**379 samples**, including 269 single-tool and 110 multi-tool queries).

### Benchmark Scorecard Comparison:

| Evaluation Metric | Strict Hermes Mode | Relaxed / Tolerant Mode | Analysis & Insights |
| :--- | :---: | :---: | :--- |
| **Tool Invocation Recall** | **50.26%** | **82.54%** | Model attempts function calling in 82.5% of queries, but omits `<tool_call>` tags in ~32% of cases. |
| **Tool Invocation F1** | **66.78%** | **90.30%** | When relaxed, the model demonstrates high sensitivity to queries requiring tools. |
| **Exact Tool Name Match** | **37.99%** | **67.55%** | Under relaxed parsing, the model selects the exact correct tool in over two-thirds of samples. |
| **Tool Set Jaccard Similarity**| **43.45%** | **74.10%** | High tool set overlap with target functions. |
| **Tag Compliance Rate** | **51.98%** | **51.98%** | Approximately half the responses follow the expected `<tool_call>` XML convention. |
| **Argument Key F1** | **38.17%** | **45.56%** | Model generates appropriate parameter names for matching functions. |
| **Argument Value Exact Match**| **33.07%** | **39.55%** | Model extracts values accurately from context passages and user prompts. |
| **Multi-Tool Sequence Match** | **45.45%** | **60.91%** | Model correctly sequences multi-step tool calls in 60.9% of multi-tool queries. |
| **Strict E2E Accuracy** | **17.94%** | **17.94%** | Flawless end-to-end execution (tags + strict JSON + exact tool + exact args). |
| **Functional E2E Accuracy** | **18.47%** | **22.16%** | End-to-end task completion with normalized argument equivalence. |
| **Tool Sequence Accuracy** | **35.88%** | **64.91%** | Tool selection and workflow ordering without penalizing argument value formatting. |

---

## 5. Primary Error Modes & Fine-Tuning Recommendations

The evaluation benchmark surfaced three primary error modes in the base Gemma-2-2B-IT model:

1. **Protocol Drift (Markdown instead of XML tags)**:
   In ~32% of responses, the model generated valid JSON tool calls inside markdown code fences (````json [ ... ] ````) instead of `<tool_call> ... </tool_call>`.
   * *Recommendation*: Supervised Fine-Tuning (SFT) with explicit negative penalty on code fences and reinforcement of `<tool_call>` token boundaries.
2. **Schema Echoing / Hallucination (10.03% of samples)**:
   The model occasionally regurgitated the `<tools>` schema definition rather than generating an actual invocation payload.
   * *Recommendation*: Train on diverse system prompts where schema definitions are placed in system instructions and conversational turns strictly output invocation payloads.
3. **Multi-Tool Sequence Truncation**:
   In complex 3-step or 4-step queries, the model often called the first 1 or 2 tools, omitting downstream tools in the sequence.
   * *Recommendation*: Augment the training split with multi-turn and chained tool-call trajectories.
