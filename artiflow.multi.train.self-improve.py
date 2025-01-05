import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch import nn
from tqdm import tqdm
import os
import pandas as pd

class ArticleCategoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        categories = self.data[idx]["categories"]

        # Format the output string as a concatenation of <cate> tags for each category
        output = ''.join([f"<cate>{category}</cate>" for category in categories])

        # Tokenize input (article) and output (category tags)
        input_encoding = self.tokenizer(
            article, truncation=True, padding="max_length", max_length=self.max_input_length, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            output, truncation=True, padding="max_length", max_length=self.max_output_length, return_tensors="pt"
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

# Training Function
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    # Add tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", ncols=100, dynamic_ncols=True)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Update progress bar description with the loss
        progress_bar.set_postfix(loss=loss.item(), refresh=True)

    return total_loss / len(dataloader)

# Validation Function
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0

    # Add tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Validation", ncols=100, dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Update progress bar description with the loss
            progress_bar.set_postfix(loss=loss.item(), refresh=True)

    return total_loss / len(dataloader)

# Save model checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)

# Load model checkpoint
def load_checkpoint(model, optimizer, checkpoint_dir, filename):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    else:
        return 0, None

# Main Training Script
def main():
    # Paths and configurations
    dataset_path = "./processed_dataset.json"  # Update path to your dataset
    model_name = "./mt5-small"
    batch_size = 14
    epochs = 24
    learning_rate = 5e-5
    max_input_length = 512
    max_output_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoints-202501051213"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a dataframe to record training metrics
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    # Distributed setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Load and split dataset
    data = load_dataset(dataset_path)
    train_data, val_data = split_dataset(data)

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Prepare datasets and dataloaders
    train_dataset = ArticleCategoryDataset(train_data, tokenizer, max_input_length, max_output_length)
    val_dataset = ArticleCategoryDataset(val_data, tokenizer, max_input_length, max_output_length)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Check if resuming from checkpoint
    checkpoint_filename = "checkpoint.pth"
    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_dir, checkpoint_filename)

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss = validate_model(model, val_dataloader, device)

        # Record metrics
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        }])], ignore_index=True)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint after every epoch
        save_checkpoint(epoch + 1, model, optimizer, val_loss, checkpoint_dir, checkpoint_filename)

        # Save metrics to CSV after every epoch
        metrics_df.to_csv("training_metrics.csv", index=False)

    # Save the final trained model (Only on rank 0)
    if local_rank == 0:
        model.module.save_pretrained("category_generation_model")
        tokenizer.save_pretrained("category_generation_model")

if __name__ == "__main__":
    main()

