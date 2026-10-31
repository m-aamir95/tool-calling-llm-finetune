import modal
from datasets import load_dataset
from typing import Dict

model_downloader = modal.Function.from_name(
    "llm-t4-inference-experimentation",
    "download_model"
)

model_inference_engine = modal.Cls.from_name(
    "llm-t4-inference-experimentation",
    "GemmaInference" # Class name to be used
)


"""
 messages_sample = [
            [{
                "role": "user",
                "content": "What is the weather in Karachi in October and November?",
            }],

            [{
                "role": "user",
                "content": "What is 2+2",
            }]
        ]
"""
def go_for_the_evaluation_boy(messages : list[list[Dict[str, str]]]) -> list[str]:

    # Make sure that the model is is downloaded on the Modal volumes
    model_downloader.remote()

    return model_inference_engine().run.remote(messages)


def extract_system_and_user_messages_only(ds):
    """
    The Hermes dataset contains:
    system (tool definitions & instructions) + human (user prompt) + gpt (tool call / model response) ...

    Gemma models do not natively support a separate 'system' role in apply_chat_template.
    Instead, system instructions are prepended directly into the first user turn.
    """
    transformed_ds = []
    for d in ds:
        convs = d["conversations"]
        system_content = convs[0]["value"] if len(convs) > 0 and convs[0]["from"] == "system" else ""
        user_content = convs[1]["value"] if len(convs) > 1 and convs[1]["from"] == "human" else (convs[0]["value"] if convs else "")

        # Prepend system instructions to the first user turn
        combined_content = f"{system_content}\n\n{user_content}".strip() if system_content else user_content

        transformed_ds.append([
            {"role": "user", "content": combined_content}
        ])

    return transformed_ds

if __name__ == "__main__":

    # Downloading the dataset
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling")

    ds_train_test = ds["train"].train_test_split(
        test_size=0.2,
        seed=42
    )

    # Note on Gemma role mapping:
    # Gemma's default chat template only supports alternating 'user' and 'assistant' roles:
    # - "system" -> prepended into the first "user" message
    # - "human"  -> "user"
    # - "gpt"    -> "assistant"
    # - "tool"   -> wrapped in <tool_response> inside a "user" message
    system_user_messages = extract_system_and_user_messages_only(ds_train_test["test"])

    # print(system_user_messages[0])
    # print("="*20)
    # for x in ds_train_test["test"][111]["conversations"]:
        # print("="*20)
        # print(x)
    # print(ds_train_test["test"][0]["conversations"])

    model_responses = go_for_the_evaluation_boy(messages=system_user_messages[:5])

    for mr in model_responses:
        print("="*20)
        print(mr)