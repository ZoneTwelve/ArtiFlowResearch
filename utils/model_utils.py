# utils/model_utils.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ModelHandler:
    def __init__(self, model_name="google/mt5-small", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def save_model(self, save_directory):
        """Save the model and tokenizer to the specified directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")

    def load_model(self, model_path):
        """Load the model and tokenizer from the specified directory."""
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        print(f"Model and tokenizer loaded from {model_path}")
