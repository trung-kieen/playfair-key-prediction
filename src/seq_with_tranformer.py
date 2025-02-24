from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorflow as tf

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="tf").input_ids

output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
