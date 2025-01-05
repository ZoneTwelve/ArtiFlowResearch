import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
import numpy as np

class ArticleTitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        title = self.data[idx]["title"]

        # Tokenize input (article) and output (title)
        input_encoding = self.tokenizer(
            article, truncation=True, padding="max_length", max_length=self.max_input_length, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            title, truncation=True, padding="max_length", max_length=self.max_output_length, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

# Load the dataset
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Train-Test Split
def split_dataset(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size)
    return train_data, val_data

# Reward Function (dummy implementation, replace with BLEU/ROUGE or custom logic)
def compute_reward(predictions, references):
    # This is a simple reward function; you can use ROUGE, BLEU, or any custom metric
    if predictions == references:
        return 1.0  # High reward for correct predictions
    else:
        return -1.0  # Negative reward for incorrect predictions

# Training Function with RL
def train_model_rl(model, dataloader, optimizer, device, tokenizer, reward_fn):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Generate predictions
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        decoded_labels = [tokenizer.decode(lbl, skip_special_tokens=True) for lbl in labels]

        # Compute reward for each prediction
        rewards = np.array([reward_fn(pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)])

        # RL loss: combine the standard loss with the reward (negative reward means bad predictions)
        rl_loss = loss - torch.mean(torch.tensor(rewards, dtype=torch.float32).to(device))

        # Backward pass
        rl_loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Validation Function
def validate_model_rl(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Main Training Script with RL
def main():
    # Paths and configurations
    dataset_path = "dataset.jsonl"
    success_path = "success.jsonl"
    failed_path = "failed.jsonl"
    
    model_name = "google/mt5-small"
    batch_size = 24
    epochs = 24
    learning_rate = 5e-5
    max_input_length = 512
    max_output_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Load dataset")
    # Load and split dataset
    data = load_dataset(dataset_path)
    success_data = load_dataset(success_path)
    failed_data = load_dataset(failed_path)
    train_data, val_data = split_dataset(data)

    print("Load model")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    print("Prepare datasets")
    # Prepare datasets and dataloaders
    train_dataset = ArticleTitleDataset(train_data, tokenizer, max_input_length, max_output_length)
    val_dataset = ArticleTitleDataset(val_data, tokenizer, max_input_length, max_output_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print("Optimizer")
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Ready to train")
    # Training loop
    for epoch in range(epochs):
        train_loss = train_model_rl(model, train_dataloader, optimizer, device, tokenizer, compute_reward)
        val_loss = validate_model_rl(model, val_dataloader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    # Save the trained model
    model.save_pretrained("title_generation_model")
    tokenizer.save_pretrained("title_generation_model")

if __name__ == "__main__":
    main()
