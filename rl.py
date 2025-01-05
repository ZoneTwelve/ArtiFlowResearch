#!/usr/bin/env python
# train_rl_titlegen.py

import argparse
import json
import re
import csv
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split


################################################################################
# 1) Argument Parser
################################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an mT5 model with a simple RL reward for <title>...</title> outputs."
    )
    parser.add_argument("--model_name", type=str, default="google/mt5-small",
                        help="Name of the pretrained T5/mT5 model on HuggingFace.")
    parser.add_argument("--dataset_path", type=str, default="dataset.jsonl",
                        help="Path to the dataset in JSON Lines format.")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=24,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Max token length for input (article).")
    parser.add_argument("--max_output_length", type=int, default=128,
                        help="Max token length for output (title).")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Weight for RL reward component.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use as validation set.")
    parser.add_argument("--save_csv", action="store_true",
                        help="If given, will save epoch metrics to a CSV file.")
    parser.add_argument("--output_dir", type=str, default="title_generation_model",
                        help="Directory to save the trained model and tokenizer.")
    return parser.parse_args()


################################################################################
# 2) Dataset and Helpers
################################################################################
class ArticleTitleDataset(Dataset):
    """
    Simple PyTorch Dataset to hold (article, title) pairs for T5/mT5 training.
    """
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

        # Tokenize input (article)
        input_encoding = self.tokenizer(
            article,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        # Tokenize output (title)
        target_encoding = self.tokenizer(
            title,
            truncation=True,
            padding="max_length",
            max_length=self.max_output_length,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }


def load_dataset(file_path):
    """
    Load dataset from a JSONL (json lines) file:
    Each line is a JSON object with at least:
      {
        "article": "...",
        "title": "..."
      }
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_dataset(data, test_size=0.2):
    """
    Splits data into train/validation sets.
    """
    train_data, val_data = train_test_split(data, test_size=test_size)
    return train_data, val_data


################################################################################
# 3) RL Components (Regex Reward)
################################################################################
# We want the model to produce strings in the format "<title> ... </title>"
title_pattern = r"<title>(.*?)</title>"

def compute_reward(generated_text: str, pattern: str) -> float:
    """
    Returns 1.0 if `generated_text` fully matches <title>...</title> pattern,
    else 0.0.
    """
    match = re.fullmatch(pattern, generated_text.strip())
    return 1.0 if match else -10.0


################################################################################
# 4) Training and Validation (with RL)
################################################################################
def train_model_rl(model, dataloader, tokenizer, optimizer,
                   device, alpha=0.1, max_output_length=128):
    """
    One epoch of training that combines supervised cross-entropy loss
    with a simple RL penalty/bonus for matching <title>...</title>.
    
    Args:
        alpha: Weight of the RL reward term. Larger => more RL influence.
    Returns:
        avg_ce_loss (float): average cross-entropy loss
        avg_reward (float): average reward for the epoch
    """
    model.train()
    total_ce_loss = 0.0
    total_reward = 0.0
    num_batches = len(dataloader)

    # We use a tqdm progress bar to show training status
    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(loop):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # (1) Usual supervised forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        ce_loss = outputs.loss  # cross-entropy from T5ForConditionalGeneration

        # (2) Generate predictions (with model in eval mode for deterministic generation)
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_output_length
            )
        model.train()

        # (3) Compute RL reward for each sample in the batch, then average
        generated_texts = [
            tokenizer.decode(g_ids, skip_special_tokens=True)
            for g_ids in generated_ids
        ]
        batch_reward = sum(
            compute_reward(txt, title_pattern) for txt in generated_texts
        ) / len(generated_texts)

        total_reward += batch_reward

        # (4) Combine the losses: naive RL => loss = CE_loss - alpha * reward
        loss = ce_loss - alpha * batch_reward
        loss.backward()
        optimizer.step()

        # Update metrics
        total_ce_loss += ce_loss.item()

        # Update progress bar info
        loop.set_postfix({
            "CE_Loss": ce_loss.item(),
            "Batch_Reward": batch_reward
        })

    avg_ce_loss = total_ce_loss / num_batches
    avg_reward = total_reward / num_batches
    return avg_ce_loss, avg_reward


def validate_model_rl(model, dataloader, tokenizer, device, max_output_length=128):
    model.eval()
    total_ce_loss = 0.0
    total_reward = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Cross-entropy
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            ce_loss = outputs.loss
            total_ce_loss += ce_loss.item()

            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_output_length
            )
            generated_texts = [
                tokenizer.decode(g_ids, skip_special_tokens=True)
                for g_ids in generated_ids
            ]
            batch_reward = sum(
                compute_reward(txt, title_pattern) for txt in generated_texts
            ) / len(generated_texts)

            total_reward += batch_reward

            loop.set_postfix({
                "CE_Loss": ce_loss.item(),
                "Batch_Reward": batch_reward
            })

    avg_ce_loss = total_ce_loss / num_batches
    avg_reward = total_reward / num_batches
    return avg_ce_loss, avg_reward


################################################################################
# 5) Main Training Script
################################################################################
def main():
    args = parse_args()

    # Print arguments
    print("===== Training Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load dataset and split
    print("Loading dataset...")
    data = load_dataset(args.dataset_path)
    train_data, val_data = split_dataset(data, test_size=args.test_size)

    # 2) Load tokenizer and model
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # 3) Prepare DataLoaders
    print("Preparing DataLoaders...")
    train_dataset = ArticleTitleDataset(
        train_data,
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    val_dataset = ArticleTitleDataset(
        val_data,
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4) Optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 5) Training loop
    print("Starting training...")
    # If saving CSV stats
    csv_file = None
    csv_writer = None
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_file = open(os.path.join(args.output_dir, "training_logs.csv"), mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "train_reward", "val_loss", "val_reward"])

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_ce_loss, train_reward = train_model_rl(
            model, train_dataloader, tokenizer, optimizer,
            device=device,
            alpha=args.alpha,
            max_output_length=args.max_output_length
        )
        val_ce_loss, val_reward = validate_model_rl(
            model, val_dataloader, tokenizer,
            device=device,
            max_output_length=args.max_output_length
        )

        print(f"[Epoch {epoch+1}] Train CE Loss: {train_ce_loss:.4f}, "
              f"Train Reward: {train_reward:.4f} | "
              f"Val CE Loss: {val_ce_loss:.4f}, Val Reward: {val_reward:.4f}")

        # Save stats to CSV if needed
        if csv_writer:
            csv_writer.writerow([epoch+1, train_ce_loss, train_reward, val_ce_loss, val_reward])
            csv_file.flush()

    # 6) Save the trained model and tokenizer
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if csv_file:
        csv_file.close()

    print("Training complete.")


if __name__ == "__main__":
    main()

