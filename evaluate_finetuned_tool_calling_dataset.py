import modal
import pandas as pd
from datasets import load_dataset
from typing import Dict

# Modal app and class names matching llm_inference_finetuned_with_modal.py
APP_NAME = "llm-t4-finetuned-inference"
CLASS_NAME = "GemmaFinetunedInference"

model_downloader = modal.Function.from_name(
    APP_NAME,
    "download_model"
)

model_inference_engine = modal.Cls.from_name(
    APP_NAME,
    CLASS_NAME
)


def extract_system_and_user_messages_only(ds):
    """
    The Hermes dataset contains:
    system (tool definitions & instructions) + human (user prompt) + gpt (tool call / model response) ...

    Gemma models do not natively support a separate 'system' role in apply_chat_template.
    System instructions and tool definitions are prepended directly into the first user turn,
    matching both the baseline evaluation and the Kaggle Unsloth training script.
    """
    transformed_ds = []
    target_tools = []
    raw_conversations = []
    for d in ds:
        convs = d["conversations"]
        system_content = convs[0]["value"] if len(convs) > 0 and convs[0]["from"] == "system" else ""
        user_content = convs[1]["value"] if len(convs) > 1 and convs[1]["from"] == "human" else (convs[0]["value"] if convs else "")

        combined_content = f"{system_content}\n\n{user_content}".strip() if system_content else user_content

        transformed_ds.append([
            {"role": "user", "content": combined_content}
        ])

        target_tools.append(convs[2] if len(convs) > 2 else None)
        raw_conversations.append(convs)

    return transformed_ds, target_tools, raw_conversations


if __name__ == "__main__":
    # Load dataset with the exact same seed & split as baseline and training
    print("Loading Hermes function calling test split...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling")

    ds_train_test = ds["train"].train_test_split(
        test_size=0.2,
        seed=42
    )

    system_user_messages, target_tools, raw_conversations = extract_system_and_user_messages_only(ds_train_test["test"])
    total = len(system_user_messages)
    print(f"Loaded {total} test samples.")

    # ----------------------------------------------------------------
    # Build ordered batches upfront
    # ----------------------------------------------------------------
    batch_size = 4
    batches: list[list[list[Dict[str, str]]]] = []
    batch_meta: list[dict] = []   # keeps target_tools / raw_convs aligned per batch

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batches.append(system_user_messages[start:end])
        batch_meta.append({
            "target_tools": target_tools[start:end],
            "raw_conversations": raw_conversations[start:end],
            "messages": system_user_messages[start:end],
        })

    num_batches = len(batches)
    print(f"Submitting all {num_batches} batches concurrently via Modal .map() "
          f"(up to {num_batches} containers may launch in parallel)...")

    # ----------------------------------------------------------------
    # Ensure model weights are present on the volume before inference
    # ----------------------------------------------------------------
    model_downloader.remote()

    # ----------------------------------------------------------------
    # Parallel dispatch: Modal distributes all batches across available
    # containers simultaneously and streams results back in input order.
    # ----------------------------------------------------------------
    engine = model_inference_engine()
    all_responses: list[list[str]] = list(engine.run.map(batches))

    print(f"All {num_batches} batches completed. Assembling results...")

    # ----------------------------------------------------------------
    # Stitch results back in original sample order
    # ----------------------------------------------------------------
    batch_dfs = []
    for meta, responses in zip(batch_meta, all_responses):
        batch_df = pd.DataFrame({
            "raw_conversations": meta["raw_conversations"],
            "system_user_messages": meta["messages"],
            "target_tools": meta["target_tools"],
            "model_responses": responses,
        })
        batch_dfs.append(batch_df)

    evaluation_df = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else pd.DataFrame()

    output_path = "func_calling_dataset_gemma_lora_responses_via_fine_tuned.json"
    evaluation_df.to_json(output_path, indent=2)
    print(f"Saved {len(evaluation_df)} evaluated rows to {output_path}")

