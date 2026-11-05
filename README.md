# Gemma Tool-Calling Fine-Tuning and Evaluation

This project fine-tunes and benchmarks **Google Gemma 2 2B Instruct** for structured function/tool calling using the Hermes function-calling dataset.

## Workflow and infrastructure

- **Fine-tuning:** LoRA supervised fine-tuning with Unsloth. The included training notebook records a Kaggle-based run.
- **Inference:** Base and fine-tuned models are served in GPU-backed **Modal** containers. Modal Volumes cache model files, and Modal distributes inference batches across containers.
- **Evaluation:** A Python benchmark compares generated calls with reference calls on the same held-out examples.

## Evaluation

The evaluator measures six aspects of tool calling:

1. **Format:** tool-call tag compliance, JSON parsing, malformed output, and schema-only responses.
2. **Invocation:** whether a required tool call was made (precision, recall, and F1).
3. **Tool selection:** exact tool-set match, Jaccard overlap, per-query macro precision/recall, and extra tool names.
4. **Arguments:** key precision/recall, types, exact and normalized values.
5. **Sequences:** exact call order, sequence similarity, and whether the number of calls matches. Sequence similarity uses Python `difflib.SequenceMatcher`.
6. **End-to-end success:** strict format-and-argument correctness, normalized functional correctness, and tool-sequence correctness.

Tool selection precision and recall are averaged per query, so each query contributes equally (macro averaging). Jaccard similarity is the size of the intersection of predicted and expected tool sets divided by their union.

## Benchmark results

Base and LoRA fine-tuned outputs were evaluated on the same **379 examples** (269 single-tool and 110 multi-tool). The table shows strict-mode results; changes are percentage points.

| Metric | Base | LoRA fine-tuned | Change |
|---|---:|---:|---:|
| Tag compliance | 51.98% | **60.69%** | **+8.71 pp** |
| Tool invocation recall | 50.26% | **60.58%** | **+10.32 pp** |
| Exact tool-name match | 37.99% | **46.97%** | **+8.97 pp** |
| Argument-key F1 | 38.17% | **51.13%** | **+12.95 pp** |
| Exact argument-value match | 33.07% | **45.27%** | **+12.21 pp** |
| Multi-tool sequence exact match | 45.45% | **65.45%** | **+20.00 pp** |
| Strict end-to-end accuracy | 17.94% | **27.70%** | **+9.76 pp** |
| Functional end-to-end accuracy | 18.47% | **27.97%** | **+9.50 pp** |

Fine-tuning improved the listed strict metrics. Strict end-to-end accuracy rose by about 54% relative to the base score, but remains below 30%.

**Parsing caveat:** Some fine-tuned outputs use `<tool-call>` (hyphen), while the evaluator expects `<tool_call>` (underscore). Strict scores therefore reflect output-format adherence as well as tool-calling ability. In relaxed mode, the base model scores higher on invocation recall, exact tool-name match, and sequence accuracy because more of its alternate output formats are recoverable. The fine-tuned model scores higher on relaxed functional end-to-end accuracy (27.97% vs. 22.16%).

## Remaining challenges

- Standardize the fine-tuned output to the expected `<tool_call>` format.
- Improve complete argument accuracy; exact argument values match in 45.27% of comparisons.
- Improve completeness on multi-tool tasks; exact sequence match is 65.45%.

## Repository layout

```text
data/responses/   Base and fine-tuned model outputs
reports/          Evaluation report
src/evaluation/   Inference clients, data preparation, and evaluation
src/modal_apps/   Base and fine-tuned Modal inference apps
notebooks/        Fine-tuning notebook
```

## Key technologies

Python, Gemma 2 2B Instruct, LoRA, Unsloth, PyTorch, Hugging Face Transformers/Datasets, Modal, pandas.
