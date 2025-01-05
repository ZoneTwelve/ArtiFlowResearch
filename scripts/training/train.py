# scripts/training/train.py

import argparse
import os
from utils.data_utils import load_jsonl, split_data
from utils.model_utils import ModelHandler
from utils.training_utils import train_epoch, validate
from utils.rl_utils import compute_reward

def main(args):
    # Load dataset
    print("Loading dataset...")
    data = load_jsonl(args.dataset_path)
    train_data, val_data = split_data(data, test_size=args.test_size)

    # Initialize model handler
    model_handler = ModelHandler(model_name=args.model_name, device=args.device)
    model = model_handler.model
    tokenizer = model_handler.tokenizer

    # Prepare datasets and dataloaders
    print("Preparing datasets...")
    from torch.utils.data import Dataset

    class ArticleTitleDatasetCustom(Dataset):
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

    train_dataset = ArticleTitleDatasetCustom(train_data, tokenizer, args.max_input_length, args.max_output_length)
    val_dataset = ArticleTitleDatasetCustom(val_data, tokenizer, args.max_input_length, args.max_output_length)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        if args.rl:
            train_loss = train_epoch(
                model, train_dataloader, optimizer, args.device,
                tokenizer=tokenizer,
                rl=True,
                reward_fn=lambda g, r: compute_reward(g, r),
                alpha=args.alpha,
                max_output_length=args.max_output_length
            )
            val_loss, val_reward = validate(
                model, val_dataloader, args.device,
                tokenizer=tokenizer,
                rl=True,
                reward_fn=lambda g, r: compute_reward(g, r),
                max_output_length=args.max_output_length
            )
            print(f"Train Loss: {train_loss:.4f}, Train Reward: {val_reward:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Reward: {val_reward:.4f}")
        else:
            train_loss = train_epoch(
                model, train_dataloader, optimizer, args.device
            )
            val_loss = validate(
                model, val_dataloader, args.device
            )
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Save final model
    model_handler.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mT5 model with optional RL.")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pretrained model name or path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSONL file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input token length.")
    parser.add_argument("--max_output_length", type=int, default=128, help="Max output token length.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Train-validation split ratio.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--output_dir", type=str, default="title_generation_model", help="Directory to save the final model.")
    parser.add_argument("--rl", action='store_true', help="Enable Reinforcement Learning training.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for RL reward.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (cpu/cuda).")

    args = parser.parse_args()
    main(args)
