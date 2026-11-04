import os
from pathlib import Path
import modal

# ==============================================================================
# Model & Storage Configuration
# ==============================================================================
# Base model used during Unsloth training
BASE_MODEL_ID = "google/gemma-2-2b-it"

# Fine-tuned LoRA Adapter repository pushed from Kaggle/Unsloth
LORA_MODEL_ID = "m-aamir2979/gemma-2-2b-bnb-4bit-fineTuned-tool-calling-adapter"

# Shared Modal volume mount point and directories
VOLUME_DIR = Path("/models")
BASE_MODEL_DIR = VOLUME_DIR / "gemma-2-2b-it"
LORA_DIR = VOLUME_DIR / "gemma-2-2b-it-lora-adapter"

# ==============================================================================
# Container Image Setup
# ==============================================================================
# Includes PyTorch, Transformers, Hugging Face Hub, and PEFT for LoRA merging
image_to_use = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "datasets",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "peft",          # Required for loading and fusing LoRA adapters
        "bitsandbytes",  # 4-bit / 8-bit support if needed
    )
    .pip_install_from_requirements("requirements.txt")
)

# Re-use existing volume so the base Gemma model doesn't need to be redownloaded
model_volume = modal.Volume.from_name("gemma-models", create_if_missing=True)

# Standalone Modal App for the Fine-Tuned Model
app = modal.App(
    name="llm-t4-finetuned-inference",
    image=image_to_use,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


# ==============================================================================
# Downloader: Fetches Base Model and LoRA Adapter into Volume
# ==============================================================================
@app.function(volumes={str(VOLUME_DIR): model_volume})
def download_model():
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # 1. Download Base Model (skip if already cached from previous base deployment)
    if not (BASE_MODEL_DIR / "config.json").exists():
        print(f"Downloading base model {BASE_MODEL_ID} into {BASE_MODEL_DIR}...")
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            local_dir=str(BASE_MODEL_DIR),
            token=hf_token,
        )
    else:
        print(f"Base model {BASE_MODEL_ID} already cached at {BASE_MODEL_DIR}.")

    # 2. Download LoRA Adapter weights
    print(f"Downloading LoRA adapter {LORA_MODEL_ID} into {LORA_DIR}...")
    snapshot_download(
        repo_id=LORA_MODEL_ID,
        local_dir=str(LORA_DIR),
        token=hf_token,
    )

    model_volume.commit()
    print("Download complete and volume committed successfully!")


# ==============================================================================
# Scaling Configuration
# ==============================================================================
# Maximum number of T4 GPU containers that Modal can spin up concurrently.
# Each container loads and holds the full merged model in its own GPU memory.
# With 379 test samples and batch_size=5 → 76 batches. Modal will distribute
# these across up to MAX_CONTAINERS containers in parallel.
MAX_CONTAINERS = 5

# Seconds to keep an idle warm container alive between batch calls.
# Prevents repeated cold-starts during the evaluation loop.
SCALEDOWN_WINDOW_SECONDS = 90  # Seconds


# ==============================================================================
# Inference Engine: Merges LoRA into Base Model on Startup for Fast Inference
# ==============================================================================
@app.cls(
    gpu="T4",
    volumes={str(VOLUME_DIR): model_volume},
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class GemmaFinetunedInference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # 1. Load Tokenizer
        # Prefer tokenizer from the adapter repo since Unsloth saves updated chat templates & tokens there
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(LORA_DIR), local_files_only=True
            )
            print("Loaded fine-tuned tokenizer from adapter repository.")
        except Exception as e:
            print(f"Fallback to base tokenizer due to: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(BASE_MODEL_DIR), local_files_only=True
            )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Model in float16
        print(f"Loading base model from {BASE_MODEL_DIR}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

        # 3. Attach LoRA Adapter and merge into base model weights
        print(f"Attaching LoRA adapter from {LORA_DIR}...")
        lora_model = PeftModel.from_pretrained(
            base_model,
            str(LORA_DIR),
            local_files_only=True,
        )

        print("Fusing LoRA weights into base model (merge_and_unload)...")
        self.model = lora_model.merge_and_unload()
        self.model.eval()
        print("Model successfully merged and ready for low-latency inference on T4!")

    @modal.method()
    def run(self, messages: list[list[dict[str, str]]]) -> list[str]:
        """
        Runs batch inference on a list of conversation histories.
        Each conversation is a list of {"role": str, "content": str} dictionaries.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Matched to Unsloth training notebook:
        # greedy decoding (do_sample=False) and max_new_tokens=512 for structured JSON tool calls
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )

        input_width = inputs["input_ids"].shape[-1]
        return self.tokenizer.batch_decode(
            output[:, input_width:],
            skip_special_tokens=True,
        )


# ==============================================================================
# Standalone CLI Test Runner
# ==============================================================================
@app.local_entrypoint()
def main():
    print("Ensuring model and adapter are downloaded on Modal volume...")
    download_model.remote()


    # Testing tool prompt, this is the same prompt from kaggle
    # Define your tool definitions (matching your training data format)
    tools_definition = """
    You have access to the following tools:
    <tool>
    {
    "name": "get_current_weather",
    "description": "Get the current weather for a given location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {"type": "string", "city": "The Name of the City"}
        },
        "required": ["location"]
    }
    }
    </tool>
    """

    system_content = f"""You are a function calling AI. You MUST output a tool call whenever a user query can be answered using the available tools. Do not write conversational filler text; output only the tool invocation block.
    {tools_definition}

    Please tell me the current weather for the following cities.
    1. Karachi 
    2. Rawalpindi
    3. Murree

    Also I need to know the weather for capital of south korea, I am not able to recall the name though
    """


    
    test_messages = [
        [{
            "role": "user", "content" : "Please tell me about the person on Moon, tell me short story about them."
        }],
        [{
            "role": "user",
            "content": "What is 2 + 2?",
        }],
        [
            {"role" : "user", "content" : system_content}
        ]
    ]

    print("\nRunning inference test on deployed fine-tuned model...")
    responses = GemmaFinetunedInference().run.remote(test_messages)

    print("\n--- Model Responses ---")
    for msg, response in zip(test_messages, responses):
        print(f"Prompt: {msg[0]['content'][:100]}...")
        print(f"Response:\n{response}")
        print("=" * 60)

