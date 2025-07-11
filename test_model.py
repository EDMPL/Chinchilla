import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
messages = [
    {
        "role": "system",
        "content": "You are ChinchillaAI, a friendly and privacy-respecting chatbot who loves protecting privacy",
    },
    {"role": "user", "content": "Hi!"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.5,
    top_k=10,
    top_p=0.95,
    repetition_penalty=1.2
)

# Decode and print
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)