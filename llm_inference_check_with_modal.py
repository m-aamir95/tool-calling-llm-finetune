import modal
from dotenv import load_dotenv

load_dotenv()

# CHANGED: Lowered python_version to "3.11" or "3.12" to support PyTorch
image_to_use = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", 
    "transformers", 
    "accelerate"
).pip_install_from_requirements("requirements.txt")

app = modal.App(
    name="llm-t4-inference-experimentation",
    image=image_to_use,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)

@app.function(gpu="T4")
def run_llm_inference():

    import torch
    print(torch.cuda.is_available())
    # Your GPU execution code goes here
    return "This is coming from cloud GPU"

@app.local_entrypoint()
def main():
    print("Called Modal bhai jaan, for the inference")
    result = run_llm_inference.remote() 
    print(result)
