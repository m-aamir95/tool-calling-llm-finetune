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
    
    # messages = [
    #         [{
    #             "role": "user",
    #             "content": "What is the weather in Karachi in October and November?",
    #         }],

    #         [{
    #             "role": "user",
    #             "content": "What is 2+2",
    #         }]
    #     ]

    messages = [
    # 1. Weather & Geography
    [{"role": "user", "content": "What is the weather in Karachi in October and November?"}],
    [{"role": "user", "content": "What is the capital city of France?"}],
    [{"role": "user", "content": "Name three oceans in the world."}],
    [{"role": "user", "content": "Which continent is Egypt located in?"}],

    # 2. Basic Math & Logic
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is 15 divided by 3?"}],
    [{"role": "user", "content": "Is 17 an even or odd number?"}],
    [{"role": "user", "content": "What comes next in the pattern: 2, 4, 6, 8, __?"}],

    # 3. Simple Definitions & Facts
    [{"role": "user", "content": "What color do you get when you mix blue and yellow?"}],
    [{"role": "user", "content": "What is the main function of the heart?"}],
    [{"role": "user", "content": "How many days are in a leap year?"}],
    [{"role": "user", "content": "What planet is closest to the Sun?"}],

    # 4. Text Processing & Formatting
    [{"role": "user", "content": "Translate 'Hello, how are you?' into Spanish."}],
    [{"role": "user", "content": "Capitalize all words in this sentence: 'the cat sat on the mat'."}],
    [{"role": "user", "content": "Count the number of words in this sentence: 'I love learning new things.'"}],
    [{"role": "user", "content": "What is the opposite of the word 'heavy'?"}],

    # 5. Short General Knowledge & Everyday Queries
    [{"role": "user", "content": "Name two ingredients used to make a simple cheese sandwich."}],
    [{"role": "user", "content": "Why do humans need to drink water?"}],
    [{"role": "user", "content": "What sound does a dog make?"}],
    [{"role": "user", "content": "List three common primary colors."}],
    [{"role": "user", "content": "How many hours are there in one day?"}],
    [{"role": "user", "content": "What season comes right after winter?"}],
]

    responses = model_inference_engine().run.remote(messages)

    print("\n--- Model Response ---")

    for message, response in zip(messages,responses):
        question = message[0]["content"]

        output = """
        Question: <question>
        Model Says: <answer>
        """

        output = output.replace("<question>", question)
        output = output.replace("<answer>", response)

        print(output)
        print("*"*20)


if __name__ == "__main__":


    go_for_the_evaluation_boy()