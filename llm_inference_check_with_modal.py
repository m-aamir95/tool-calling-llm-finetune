import os
from pathlib import Path
import modal

MODEL_ID = "google/gemma-2-2b-it"
MODEL_DIR = Path("/models/gemma-2-2b-it")

# Create image with hf_transfer for hyper-fast model downloads
image_to_use = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "datasets",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
    )
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_volume = modal.Volume.from_name("gemma-models", create_if_missing=True)

app = modal.App(
    name="llm-t4-inference-experimentation",
    image=image_to_use,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


@app.function(volumes={str(MODEL_DIR): model_volume})
def download_model():
    from huggingface_hub import snapshot_download

    # Fetch token safely from environment (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Downloading model {MODEL_ID} into volume...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        token=hf_token,
    )
    model_volume.commit()
    print("Download complete and volume committed!")


@app.cls(
    gpu="T4",
    volumes={str(MODEL_DIR): model_volume},
)
class GemmaInference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        # Use float16 for Nvidia T4 (bfloat16 is inefficient on T4)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

    @modal.method()
    def run(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
        )

        # Slice output to strip the input prompt tokens and return only generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_len:]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


@app.local_entrypoint()
def main():
    download_model.remote()

    messages = [
        {
            "role": "user",
            "content": "What is the weather in Karachi in October and November?",
        }
    ]

    response = GemmaInference().run.remote(messages)
    print("\n--- Model Response ---")
    print(response)