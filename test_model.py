import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
messages = [
    {
        "role": "system",
        "content": "Your name is 'ChinchillAI'",
    },
    {"role": "user", "content": "What is your name?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_k=20, top_p=0.9)
answer = str(outputs[0]["generated_text"])
index = answer.find("<|assistant|>")
if index != -1:
    final_answer = answer[index + len("<|assistant|>"):].strip()
    print(final_answer)