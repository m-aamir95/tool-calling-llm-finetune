import modal


model_downloader = modal.Function.from_name(
    "llm-t4-inference-experimentation",
    "download_model"
)

model_inference_engine = modal.Cls.from_name(
    "llm-t4-inference-experimentation",
    "GemmaInference" # Class name to be used
)



def go_for_the_evaluation_boy():

    # Make sure that the model is is downloaded on the Modal volumes
    model_downloader.remote()
    
    messages = [
            [{
                "role": "user",
                "content": "What is the weather in Karachi in October and November?",
            }],

            [{
                "role": "user",
                "content": "What is 2+2",
            }]
        ]

    responses = model_inference_engine().run.remote(messages)

    print("\n--- Model Response ---")

    for response in responses:
        print(response)
        print("*"*10)


if __name__ == "__main__":


    go_for_the_evaluation_boy()