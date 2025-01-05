import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5Tokenizer
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import os

class ArticleTitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        input_encoding = self.tokenizer(
            article, truncation=True, padding="max_length", max_length=self.max_input_length, return_tensors="pt"
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
        }

# Load the dataset
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Reward function
def calculate_reward(generated_title, good_samples, bad_samples):
    if generated_title in good_samples:
        return 1.0  # High reward for good titles
    elif generated_title in bad_samples:
        return -1.0  # Penalty for bad titles
    else:
        return 0.0  # Neutral reward for unknown titles

# Sampling function
def generate_title(model, tokenizer, input_ids, attention_mask, max_output_length, device):
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,
        num_beams=1,
        do_sample=True
    )
    print(f"Generated token IDs: {output[0]}")
    return tokenizer.decode(output[0], skip_special_tokens=True)


# RL training function
def train_with_reinforcement(model, dataloader, optimizer, tokenizer, good_samples, bad_samples, max_output_length, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Progress", leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Generate titles
        generated_titles = [
            generate_title(model, tokenizer, input_ids[i:i+1], attention_mask[i:i+1], max_output_length, device)
            for i in range(len(input_ids))
        ]

        # Compute rewards
        rewards = torch.tensor(
            [calculate_reward(title, good_samples, bad_samples) for title in generated_titles],
            dtype=torch.float32, device=device
        )

        # Calculate loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        log_probs = -F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1), reduction='none')
        log_probs = log_probs.view(len(input_ids), -1).mean(dim=1)  # Average log-probabilities per sample

        policy_loss = -(log_probs * rewards).mean()  # Policy gradient loss
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        total_loss += policy_loss.item()

    return total_loss / len(dataloader)

# Save model and optimizer state
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Main RL training script
def main():
    # Paths and configurations
    success_path = "success.jsonl"
    failed_path = "failed.jsonl"
    model_name = "google/mt5-small"
    batch_size = 8
    epochs = 10
    learning_rate = 5e-4
    max_input_length = 512
    max_output_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    good_samples = load_dataset(success_path)
    bad_samples = load_dataset(failed_path)

    # Prepare training data
    articles = [{"article": sample["article"]} for sample in good_samples + bad_samples]
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    dataset = ArticleTitleDataset(articles, tokenizer, max_input_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train with reinforcement learning
    for epoch in range(epochs):
        rl_loss = train_with_reinforcement(model, dataloader, optimizer, tokenizer, good_samples, bad_samples, max_output_length, device)
        print(f"Epoch {epoch + 1}/{epochs}, RL Loss: {rl_loss:.4f}")

        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch + 1, rl_loss)

    # Final model save
    model.save_pretrained("rl_title_generation_model")
    tokenizer.save_pretrained("rl_title_generation_model")

if __name__ == "__main__":
    main()

