#!/usr/bin/env python3
"""
Model Function Calling Evaluation Script
========================================
Comprehensive evaluation of LLM function/tool calling performance.

Evaluates 6 core dimensions:
1. Syntax & Parsing Integrity (Strict Hermes format vs Relaxed JSON)
2. Tool Invocation Decision (Tool requirement detection: Precision / Recall / F1)
3. Tool Selection Accuracy (Name match, set Jaccard, hallucination rate)
4. Schema & Argument Matching (Key precision/recall, type match, exact/normalized value match)
5. Tool Precedence & Sequencing (Multi-tool invocation order, count accuracy, sequence edit distance)
6. Composite End-to-End Success Rates (Strict E2E vs Functional E2E)
"""

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class NormalizedToolCall:
    """Standardized representation of a single tool/function call."""
    name: str
    arguments: Dict[str, Any]
    raw_payload: str = ""
    parse_method: str = "unknown"  # 'strict_json', 'ast_literal', 'json_unescaped'


@dataclass
class ExtractionResult:
    """Container for parsed tool calls from a text turn."""
    has_tag: bool = False
    strict_calls: List[NormalizedToolCall] = field(default_factory=list)
    relaxed_calls: List[NormalizedToolCall] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    is_schema_only: bool = False  # Model output tool definition schema instead of invoking


# ==============================================================================
# 1. Parsing & Normalization Engine
# ==============================================================================

def robust_parse_payload(text: str) -> Tuple[Optional[Any], str]:
    """
    Attempts to parse text as valid JSON or Python literal dict/list.
    Handles escaped backslashes, markdown fences, and single-quoted strings.
    """
    if not text:
        return None, "empty"

    candidates = [text]
    # Check if string contains escaped newlines/quotes like r'\n' or r'\"'
    if "\\n" in text or "\\\"" in text or "\\'" in text:
        try:
            candidates.append(bytes(text, "utf-8").decode("unicode_escape"))
        except Exception:
            pass

    for cand in candidates:
        cand_clean = cand.strip().strip("\\n").strip()
        # Strip markdown fences if present
        if cand_clean.startswith("```"):
            cand_clean = re.sub(r"^```(?:json|xml)?\s*", "", cand_clean)
            cand_clean = re.sub(r"\s*```$", "", cand_clean).strip()

        # 1. Standard strict JSON
        try:
            return json.loads(cand_clean), "strict_json"
        except Exception:
            pass

        # 2. Python ast.literal_eval fallback (Hermes dataset sometimes uses single quotes)
        try:
            val = ast.literal_eval(cand_clean)
            if isinstance(val, (dict, list)):
                return val, "ast_literal"
        except Exception:
            pass

    return None, "failed"


def normalize_to_tool_call(obj: Any, raw_str: str = "", method: str = "") -> Optional[NormalizedToolCall]:
    """
    Normalizes diverse JSON / dict formats into a unified NormalizedToolCall:
    - Standard: {"name": "func_name", "arguments": {...}}
    - Hermes inverted: {"arguments": {"name": "func_name", ...}}
    - Function wrapper: {"function": {"name": "func_name", ...}, "parameters": ...}
    - Type/Function: {"type": "function", "function": {"name": "func_name", ...}}
    """
    if not isinstance(obj, dict):
        return None

    # Standard format: {"name": ..., "arguments": ...}
    if "name" in obj:
        name = str(obj["name"]).strip()
        args = obj.get("arguments", {})
        if isinstance(args, str):
            parsed_args, _ = robust_parse_payload(args)
            args = parsed_args if isinstance(parsed_args, dict) else {"_raw": args}
        elif not isinstance(args, dict):
            args = {"_raw": args}
        return NormalizedToolCall(name=name, arguments=args, raw_payload=raw_str, parse_method=method)

    # Hermes inverted format: {"arguments": {"name": "func_name", "param1": ...}}
    if "arguments" in obj and isinstance(obj["arguments"], dict) and "name" in obj["arguments"]:
        args_copy = dict(obj["arguments"])
        name = str(args_copy.pop("name")).strip()
        return NormalizedToolCall(name=name, arguments=args_copy, raw_payload=raw_str, parse_method=method)

    # Function wrapper: {"function": {"name": ...}}
    if "function" in obj and isinstance(obj["function"], dict) and "name" in obj["function"]:
        name = str(obj["function"]["name"]).strip()
        args = obj.get("parameters", obj.get("arguments", {}))
        return NormalizedToolCall(name=name, arguments=args if isinstance(args, dict) else {}, raw_payload=raw_str, parse_method=method)

    return None


def extract_tool_calls_from_text(text: str, is_target: bool = False) -> ExtractionResult:
    """
    Extracts tool calls from raw response / conversation text.
    - Strict: Extracts exclusively from `<tool_call>...</tool_call>` tags.
    - Relaxed: Falls back to markdown code blocks or JSON arrays if tags are absent.
    """
    res = ExtractionResult()
    if not text:
        return res

    # 1. Search for Hermes <tool_call> tags
    xml_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    res.has_tag = len(xml_matches) > 0

    for m in xml_matches:
        obj, method = robust_parse_payload(m)
        if obj is None:
            res.format_errors.append(m)
            continue

        if isinstance(obj, list):
            for item in obj:
                norm = normalize_to_tool_call(item, raw_str=m, method=method)
                if norm:
                    res.strict_calls.append(norm)
                else:
                    res.format_errors.append(str(item))
        elif isinstance(obj, dict):
            norm = normalize_to_tool_call(obj, raw_str=m, method=method)
            if norm:
                res.strict_calls.append(norm)
            else:
                res.format_errors.append(str(obj))
        else:
            res.format_errors.append(str(obj))

    res.relaxed_calls = list(res.strict_calls)

    # 2. Relaxed extraction fallback (for model responses that omit <tool_call> tags)
    if not is_target and not res.relaxed_calls:
        # Detect if model echoed tool schema definitions (common failure mode)
        if re.search(r'```(?:json)?\s*\[?\s*\{\s*"(?:type|function)"\s*:\s*(?:"function"|\{)', text):
            res.is_schema_only = True

        code_blocks = re.findall(r'```(?:json|xml)?\s*(.*?)\s*```', text, re.DOTALL)
        for b in code_blocks:
            obj, method = robust_parse_payload(b)
            if obj is None:
                continue

            if isinstance(obj, list):
                for item in obj:
                    norm = normalize_to_tool_call(item, raw_str=b, method=method)
                    if norm:
                        res.relaxed_calls.append(norm)
            elif isinstance(obj, dict):
                norm = normalize_to_tool_call(obj, raw_str=b, method=method)
                if norm:
                    res.relaxed_calls.append(norm)

    return res


# ==============================================================================
# 2. Metric Evaluators
# ==============================================================================

def normalize_value(val: Any) -> Any:
    """Normalizes string / numeric / list values for semantic comparison."""
    if isinstance(val, str):
        cleaned = val.strip().lower()
        # Normalize leading period in file extensions (e.g. '.pdf' vs 'pdf')
        if cleaned.startswith(".") and len(cleaned) > 1 and cleaned[1:].isalnum():
            return cleaned[1:]
        return cleaned
    if isinstance(val, (int, float)):
        # Normalize 100.0 to 100
        return int(val) if float(val).is_integer() else round(float(val), 6)
    if isinstance(val, list):
        return sorted([str(normalize_value(x)) for x in val])
    if isinstance(val, dict):
        return {str(k).strip(): normalize_value(v) for k, v in val.items()}
    return val


def evaluate_single_call_arguments(pred_args: Dict[str, Any], target_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates argument matching for a single function call:
    - Key precision, recall, F1
    - Type match rate
    - Value exact match & normalized match
    """
    t_keys = set(target_args.keys())
    p_keys = set(pred_args.keys())

    if not t_keys and not p_keys:
        return {
            "key_precision": 1.0,
            "key_recall": 1.0,
            "key_f1": 1.0,
            "type_match_rate": 1.0,
            "val_exact_match_rate": 1.0,
            "val_norm_match_rate": 1.0,
            "exact_args_match": True,
            "norm_args_match": True,
        }

    shared_keys = t_keys & p_keys
    k_prec = len(shared_keys) / len(p_keys) if p_keys else 0.0
    k_rec = len(shared_keys) / len(t_keys) if t_keys else 0.0
    k_f1 = (2 * k_prec * k_rec / (k_prec + k_rec)) if (k_prec + k_rec) > 0 else 0.0

    type_matches = 0
    val_exact_matches = 0
    val_norm_matches = 0

    for k in shared_keys:
        pv = pred_args[k]
        tv = target_args[k]

        # Type match (allow int/float equivalence)
        if type(pv) is type(tv) or (isinstance(pv, (int, float)) and isinstance(tv, (int, float))):
            type_matches += 1

        # Exact match
        if pv == tv:
            val_exact_matches += 1
            val_norm_matches += 1
        else:
            # Normalized match
            npv = normalize_value(pv)
            ntv = normalize_value(tv)
            if npv == ntv:
                val_norm_matches += 1

    total_target_keys = len(t_keys) if t_keys else 1
    t_rate = type_matches / total_target_keys
    exact_rate = val_exact_matches / total_target_keys
    norm_rate = val_norm_matches / total_target_keys

    exact_match = (p_keys == t_keys) and (val_exact_matches == len(t_keys))
    norm_match = (p_keys == t_keys) and (val_norm_matches == len(t_keys))

    return {
        "key_precision": k_prec,
        "key_recall": k_rec,
        "key_f1": k_f1,
        "type_match_rate": t_rate,
        "val_exact_match_rate": exact_rate,
        "val_norm_match_rate": norm_rate,
        "exact_args_match": exact_match,
        "norm_args_match": norm_match,
    }


def evaluate_sample(
    sample_id: int,
    target_raw: Any,
    model_response_str: str,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Evaluates one sample conversation turn against all 6 metric dimensions.
    """
    # 1. Parse target tools
    target_text = target_raw.get("value") if isinstance(target_raw, dict) else str(target_raw or "")
    target_extraction = extract_tool_calls_from_text(target_text, is_target=True)
    target_calls = target_extraction.strict_calls or target_extraction.relaxed_calls

    # 2. Parse model response
    model_extraction = extract_tool_calls_from_text(model_response_str, is_target=False)
    pred_calls = model_extraction.strict_calls if strict_mode else model_extraction.relaxed_calls

    tool_required = len(target_calls) > 0
    tool_invoked = len(pred_calls) > 0

    # Dimension 1: Syntax & Parsing Integrity
    tag_compliance = model_extraction.has_tag
    strict_json_parse = len(model_extraction.strict_calls) > 0 and len(model_extraction.format_errors) == 0
    has_format_errors = len(model_extraction.format_errors) > 0

    # Dimension 2: Tool Invocation Decision
    # Confusion matrix elements
    tp = int(tool_required and tool_invoked)
    fp = int((not tool_required) and tool_invoked)
    fn = int(tool_required and (not tool_invoked))
    tn = int((not tool_required) and (not tool_invoked))

    # Dimension 3: Tool Selection & Name Matching
    target_names = [c.name for c in target_calls]
    pred_names = [c.name for c in pred_calls]

    t_set = set(target_names)
    p_set = set(pred_names)

    if not t_set and not p_set:
        tool_jaccard = 1.0
        tool_prec = 1.0
        tool_rec = 1.0
        tool_name_match = True
        hallucinated_tools_count = 0
    else:
        shared = t_set & p_set
        union = t_set | p_set
        tool_jaccard = len(shared) / len(union) if union else 0.0
        tool_prec = len(shared) / len(p_set) if p_set else 0.0
        tool_rec = len(shared) / len(t_set) if t_set else 0.0
        tool_name_match = (t_set == p_set)
        hallucinated_tools_count = len(p_set - t_set)

    # Dimension 4 & 5: Precedence & Sequencing
    is_multi_tool = len(target_calls) > 1
    seq_exact_match = (target_names == pred_names)
    tool_count_diff = len(pred_calls) - len(target_calls)
    tool_count_match = (len(pred_calls) == len(target_calls))

    # Sequence alignment similarity
    matcher = SequenceMatcher(None, target_names, pred_names)
    seq_similarity = matcher.ratio()

    # Dimension 6: Argument Evaluation (Aligned by call index or matching name)
    args_evals = []
    min_len = min(len(target_calls), len(pred_calls))
    for i in range(min_len):
        # Compare positional calls
        arg_res = evaluate_single_call_arguments(pred_calls[i].arguments, target_calls[i].arguments)
        args_evals.append(arg_res)

    if args_evals:
        avg_key_prec = sum(e["key_precision"] for e in args_evals) / len(args_evals)
        avg_key_rec = sum(e["key_recall"] for e in args_evals) / len(args_evals)
        avg_key_f1 = sum(e["key_f1"] for e in args_evals) / len(args_evals)
        avg_type_match = sum(e["type_match_rate"] for e in args_evals) / len(args_evals)
        avg_val_exact = sum(e["val_exact_match_rate"] for e in args_evals) / len(args_evals)
        avg_val_norm = sum(e["val_norm_match_rate"] for e in args_evals) / len(args_evals)
        all_args_exact = all(e["exact_args_match"] for e in args_evals) and tool_count_match
        all_args_norm = all(e["norm_args_match"] for e in args_evals) and tool_count_match
    else:
        avg_key_prec = 1.0 if not tool_required else 0.0
        avg_key_rec = 1.0 if not tool_required else 0.0
        avg_key_f1 = 1.0 if not tool_required else 0.0
        avg_type_match = 1.0 if not tool_required else 0.0
        avg_val_exact = 1.0 if not tool_required else 0.0
        avg_val_norm = 1.0 if not tool_required else 0.0
        all_args_exact = (not tool_required)
        all_args_norm = (not tool_required)

    # Composite End-to-End Metrics
    # Strict E2E: Tag present + Strict JSON + Exact sequence + Exact arguments
    strict_e2e_success = (
        tag_compliance and
        seq_exact_match and
        all_args_exact and
        not has_format_errors
    )

    # Functional E2E: Sequence exact match + Normalized arguments match (even if relaxed tags)
    functional_e2e_success = (
        seq_exact_match and
        all_args_norm
    )

    # Tool Name Only E2E: Correct tools in correct order regardless of argument values
    tool_sequence_success = seq_exact_match

    return {
        "sample_id": sample_id,
        "tool_required": tool_required,
        "tool_invoked": tool_invoked,
        "tag_compliance": tag_compliance,
        "strict_json_parse": strict_json_parse,
        "has_format_errors": has_format_errors,
        "is_schema_only_failure": model_extraction.is_schema_only,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "target_names": target_names,
        "pred_names": pred_names,
        "tool_jaccard": tool_jaccard,
        "tool_precision": tool_prec,
        "tool_recall": tool_rec,
        "tool_name_match": tool_name_match,
        "hallucinated_tools_count": hallucinated_tools_count,
        "is_multi_tool": is_multi_tool,
        "tool_count_match": tool_count_match,
        "tool_count_diff": tool_count_diff,
        "seq_exact_match": seq_exact_match,
        "seq_similarity": seq_similarity,
        "avg_key_precision": avg_key_prec,
        "avg_key_recall": avg_key_rec,
        "avg_key_f1": avg_key_f1,
        "avg_type_match": avg_type_match,
        "avg_val_exact": avg_val_exact,
        "avg_val_norm": avg_val_norm,
        "all_args_exact": all_args_exact,
        "all_args_norm": all_args_norm,
        "strict_e2e_success": strict_e2e_success,
        "functional_e2e_success": functional_e2e_success,
        "tool_sequence_success": tool_sequence_success,
    }


# ==============================================================================
# 3. Aggregation & Benchmark Runner
# ==============================================================================

def load_evaluation_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Loads dataset from either column-oriented dict or records list."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Column-oriented format
        first_col = next(iter(data.values()))
        indices = sorted(list(first_col.keys()), key=lambda x: int(x) if x.isdigit() else x)
        records = [
            {col: data[col][idx] for col in data.keys()}
            for idx in indices
        ]
        return records

    raise ValueError(f"Unsupported JSON structure in {file_path}")


def aggregate_metrics(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall statistics, macro/micro metrics, and failure breakdowns."""
    n = len(sample_results)
    if n == 0:
        return {}

    # Confusion matrix
    total_tp = sum(r["tp"] for r in sample_results)
    total_fp = sum(r["fp"] for r in sample_results)
    total_fn = sum(r["fn"] for r in sample_results)
    total_tn = sum(r["tn"] for r in sample_results)

    decision_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    decision_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    decision_f1 = (2 * decision_prec * decision_rec / (decision_prec + decision_rec)) if (decision_prec + decision_rec) > 0 else 0.0

    # Multi-tool subset
    multi_samples = [r for r in sample_results if r["is_multi_tool"]]
    single_samples = [r for r in sample_results if not r["is_multi_tool"]]

    def avg(key: str, subset=sample_results) -> float:
        return sum(r[key] for r in subset) / len(subset) if subset else 0.0

    return {
        "dataset_summary": {
            "total_samples": n,
            "single_tool_queries": len(single_samples),
            "multi_tool_queries": len(multi_samples),
            "queries_requiring_tool": sum(1 for r in sample_results if r["tool_required"]),
            "queries_model_invoked": sum(1 for r in sample_results if r["tool_invoked"]),
        },
        "dimension_1_syntax_and_parsing": {
            "tag_compliance_rate": avg("tag_compliance"),
            "strict_json_parse_rate": avg("strict_json_parse"),
            "format_error_rate": sum(1 for r in sample_results if r["has_format_errors"]) / n,
            "schema_hallucination_rate": sum(1 for r in sample_results if r["is_schema_only_failure"]) / n,
        },
        "dimension_2_invocation_decision": {
            "precision": decision_prec,
            "recall": decision_rec,
            "f1_score": decision_f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "true_negatives": total_tn,
        },
        "dimension_3_tool_selection": {
            "exact_tool_name_match_rate": avg("tool_name_match"),
            "tool_set_jaccard_similarity": avg("tool_jaccard"),
            "tool_selection_precision": avg("tool_precision"),
            "tool_selection_recall": avg("tool_recall"),
            "avg_hallucinated_tools_per_query": avg("hallucinated_tools_count"),
        },
        "dimension_4_argument_matching": {
            "key_precision": avg("avg_key_precision"),
            "key_recall": avg("avg_key_recall"),
            "key_f1": avg("avg_key_f1"),
            "type_match_rate": avg("avg_type_match"),
            "value_exact_match_rate": avg("avg_val_exact"),
            "value_normalized_match_rate": avg("avg_val_norm"),
            "all_arguments_exact_match_rate": avg("all_args_exact"),
            "all_arguments_normalized_match_rate": avg("all_args_norm"),
        },
        "dimension_5_precedence_and_sequencing": {
            "all_queries_sequence_exact_match": avg("seq_exact_match"),
            "all_queries_sequence_similarity": avg("seq_similarity"),
            "tool_count_match_rate": avg("tool_count_match"),
            "avg_tool_count_divergence": avg("tool_count_diff"),
            "multi_tool_subset": {
                "count": len(multi_samples),
                "sequence_exact_match": avg("seq_exact_match", multi_samples),
                "sequence_similarity": avg("seq_similarity", multi_samples),
                "tool_count_match": avg("tool_count_match", multi_samples),
            },
            "single_tool_subset": {
                "count": len(single_samples),
                "tool_name_match": avg("tool_name_match", single_samples),
            }
        },
        "dimension_6_composite_success": {
            "strict_end_to_end_accuracy": avg("strict_e2e_success"),
            "functional_end_to_end_accuracy": avg("functional_e2e_success"),
            "tool_sequence_accuracy": avg("tool_sequence_success"),
        }
    }


def print_scorecard(metrics: Dict[str, Any], mode_label: str) -> None:
    """Prints a clean ASCII scorecard to the terminal."""
    ds = metrics["dataset_summary"]
    d1 = metrics["dimension_1_syntax_and_parsing"]
    d2 = metrics["dimension_2_invocation_decision"]
    d3 = metrics["dimension_3_tool_selection"]
    d4 = metrics["dimension_4_argument_matching"]
    d5 = metrics["dimension_5_precedence_and_sequencing"]
    d6 = metrics["dimension_6_composite_success"]

    sep = "=" * 76
    sub_sep = "-" * 76

    print(f"\n{sep}")
    print(f" LLM FUNCTION CALLING EVALUATION BENCHMARK ({mode_label})")
    print(sep)
    print(f" Dataset Size: {ds['total_samples']} samples | Single-tool: {ds['single_tool_queries']} | Multi-tool: {ds['multi_tool_queries']}")
    print(f" Invocations: Required in {ds['queries_requiring_tool']} | Model Invoked in {ds['queries_model_invoked']}")
    print(sub_sep)

    print(" 1. SYNTAX & PARSING INTEGRITY")
    print(f"    - Tag Compliance (<tool_call>):          {d1['tag_compliance_rate'] * 100:6.2f}%")
    print(f"    - Strict JSON Parse Rate:                {d1['strict_json_parse_rate'] * 100:6.2f}%")
    print(f"    - Format Error Rate:                     {d1['format_error_rate'] * 100:6.2f}%")
    print(f"    - Schema Hallucination (Outputting def): {d1['schema_hallucination_rate'] * 100:6.2f}%")

    print("\n 2. TOOL INVOCATION DECISION")
    print(f"    - Precision:                             {d2['precision'] * 100:6.2f}%")
    print(f"    - Recall:                                {d2['recall'] * 100:6.2f}%")
    print(f"    - F1 Score:                              {d2['f1_score'] * 100:6.2f}%")
    print(f"    - Confusion Matrix (TP / FP / FN / TN):  {d2['true_positives']} / {d2['false_positives']} / {d2['false_negatives']} / {d2['true_negatives']}")

    print("\n 3. TOOL SELECTION / NAME MATCHING")
    print(f"    - Exact Tool Name Match Rate:            {d3['exact_tool_name_match_rate'] * 100:6.2f}%")
    print(f"    - Tool Set Jaccard Similarity:           {d3['tool_set_jaccard_similarity'] * 100:6.2f}%")
    print(f"    - Selection Precision:                   {d3['tool_selection_precision'] * 100:6.2f}%")
    print(f"    - Selection Recall:                      {d3['tool_selection_recall'] * 100:6.2f}%")
    print(f"    - Avg Hallucinated Tools / Query:        {d3['avg_hallucinated_tools_per_query']:6.2f}")

    print("\n 4. ARGUMENT SCHEMA & VALUE MATCHING")
    print(f"    - Argument Key Precision:                {d4['key_precision'] * 100:6.2f}%")
    print(f"    - Argument Key Recall:                   {d4['key_recall'] * 100:6.2f}%")
    print(f"    - Argument Key F1:                       {d4['key_f1'] * 100:6.2f}%")
    print(f"    - Argument Type Match Rate:              {d4['type_match_rate'] * 100:6.2f}%")
    print(f"    - Argument Value Exact Match:            {d4['value_exact_match_rate'] * 100:6.2f}%")
    print(f"    - Argument Value Normalized Match:       {d4['value_normalized_match_rate'] * 100:6.2f}%")
    print(f"    - All Arguments Exact Match (per query): {d4['all_arguments_exact_match_rate'] * 100:6.2f}%")
    print(f"    - All Arguments Norm Match (per query):  {d4['all_arguments_normalized_match_rate'] * 100:6.2f}%")

    print("\n 5. TOOL PRECEDENCE & SEQUENCING")
    print(f"    - Overall Sequence Exact Match:          {d5['all_queries_sequence_exact_match'] * 100:6.2f}%")
    print(f"    - Sequence Edit Similarity:              {d5['all_queries_sequence_similarity'] * 100:6.2f}%")
    print(f"    - Tool Count Match Rate:                 {d5['tool_count_match_rate'] * 100:6.2f}%")
    print(f"    - Multi-tool Sequence Exact Match:       {d5['multi_tool_subset']['sequence_exact_match'] * 100:6.2f}% (N={d5['multi_tool_subset']['count']})")
    print(f"    - Single-tool Name Exact Match:          {d5['single_tool_subset']['tool_name_match'] * 100:6.2f}% (N={d5['single_tool_subset']['count']})")

    print("\n 6. END-TO-END COMPOSITE SUCCESS")
    print(f"    - STRICT E2E ACCURACY:                   {d6['strict_end_to_end_accuracy'] * 100:6.2f}%")
    print(f"    - FUNCTIONAL E2E ACCURACY:               {d6['functional_end_to_end_accuracy'] * 100:6.2f}%")
    print(f"    - TOOL SEQUENCE ACCURACY:                {d6['tool_sequence_accuracy'] * 100:6.2f}%")
    print(sep)


# ==============================================================================
# 4. Main CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tool Calling Performance for LLM inferences")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("func_calling_dataset_gemma_2_2b_it_responses_records.json"),
        help="Path to responses JSON file (records or column format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("evaluation_report.json"),
        help="Path to save detailed JSON evaluation report"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit evaluation to first N samples"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample failure analysis"
    )
    args = parser.parse_args()

    # Fallback to column format if records file is missing
    input_path = args.input
    if not input_path.exists():
        fallback = Path("func_calling_dataset_gemma_2_2b_it_responses.json")
        if fallback.exists():
            input_path = fallback
        else:
            print(f"Error: Input file {args.input} does not exist.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading responses from {input_path}...")
    records = load_evaluation_dataset(input_path)
    if args.limit:
        records = records[:args.limit]
    print(f"Evaluating {len(records)} records...")

    # Run dual evaluation: Strict Hermes vs Relaxed Format
    strict_sample_results = []
    relaxed_sample_results = []

    for idx, r in enumerate(records):
        t_raw = r.get("target_tools") or r.get("target_tools_calls")
        m_resp = r.get("model_responses", "")

        strict_res = evaluate_sample(idx, t_raw, m_resp, strict_mode=True)
        relaxed_res = evaluate_sample(idx, t_raw, m_resp, strict_mode=False)

        strict_sample_results.append(strict_res)
        relaxed_sample_results.append(relaxed_res)

        if args.verbose and not strict_res["strict_e2e_success"]:
            print(f"[FAIL #{idx}] Target: {strict_res['target_names']} | Pred: {strict_res['pred_names']}")

    strict_metrics = aggregate_metrics(strict_sample_results)
    relaxed_metrics = aggregate_metrics(relaxed_sample_results)

    # Print scorecards
    print_scorecard(strict_metrics, "STRICT HERMES FORMAT")
    print_scorecard(relaxed_metrics, "RELAXED FORMAT (Tolerates Markdown Codeblocks)")

    # Save detailed JSON report
    report = {
        "benchmark_meta": {
            "input_file": str(input_path),
            "samples_evaluated": len(records),
        },
        "strict_metrics": strict_metrics,
        "relaxed_metrics": relaxed_metrics,
        "sample_evaluations": {
            "strict": strict_sample_results,
            "relaxed": relaxed_sample_results,
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed evaluation report to {args.output}")


if __name__ == "__main__":
    main()

