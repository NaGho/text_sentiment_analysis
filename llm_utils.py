import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print('device = ', device)


def generate_prompt(input):
    return f"""Please perform Sentiment Classification task.
            Given the sentence, assign a sentiment label
            from ['negative', 'positive']. Return label only
            without any other text.
            Sentence: Oh , and more entertaining, too.
            Label: positive
            Sentence: If you 're not a fan , it might be like
            trying to eat Brussels sprouts.
            Label: negative
            Sentence: {input}.
            Label: """


def generate_text_from_prompt(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def LLM_predict(X_train, y_train, X_test):
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        # device_map="auto"
        low_cpu_mem_usage=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_sentiments = [
        generate_text_from_prompt(model, tokenizer, generate_prompt(txt)) for txt in X_test
    ]
    print(output_sentiments)
    return np.where(np.array(output_sentiments)=='positive', 1, 0)


    
    