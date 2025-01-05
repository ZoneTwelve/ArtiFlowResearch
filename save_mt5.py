from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/mt5-small"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save tokenizer and model directly without accessing .module
model.save_pretrained("./my_mt5")
tokenizer.save_pretrained("./my_mt5")


