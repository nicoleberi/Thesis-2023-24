import tiktoken
import json
import openai
import setup

openai.api_key = setup.openai_api_key
openai.organization = setup.openai_organization_id

def measure_accuracy(data):
    """
    input is a list of examples
    output is stored in accuracies.json where each key is the example
    and the value is a dictionary with keys equal to N (# of context words)
    and values equal to accuracy at that particular N value.
    {"ex_1": {N_1: accuracy_1, N_2: accuracy_2}, "ex_2": {N_1: accuracy_1, ...}, ...}
    """
    for example in data:
        example_words = example.split()
        example_acc = {}
        for i in range(0, len(example_words) - 1):
            context_words = example_words[0:i+1]
            context = ' '.join(context_words)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "user", "content": context}
                ],
                temperature=0,
                max_tokens=3
            )
            gpt_output = response['choices'][0]['message']['content']
            gpt_output_trunc = gpt_output.split()[0]    # get only first word of output
            print("Input: ", context)
            print("Output: ", gpt_output)
            accuracy = 1 if gpt_output_trunc == example_words[i+1] else 0
            example_acc[i+1] = accuracy

        with open("output_data/accuracies.json", "r") as file:
            current = json.load(file)

        current.update({example: example_acc})

        with open("output_data/accuracies.json", "w") as file:
            json.dump(current, file)


# helper function
def get_input_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_words = [encoding.decode_single_token_bytes(token) for token in tokens]
    print("Length: ", len(token_words))
    print("Tokens: ", token_words)

get_input_tokens("Canada is a country in", "gpt-3.5-turbo")
measure_accuracy(["Canada is a"])