# utils/training_utils.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, tokenizer=None, rl=False, reward_fn=None, alpha=0.1, max_output_length=128):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_reward = 0

    progress_bar = tqdm(dataloader, desc="Training", ncols=100, dynamic_ncols=True)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", None)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        if rl and reward_fn and labels is not None:
            # Generate predictions
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_output_length
                )
            # Decode
            generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            # Compute rewards
            rewards = torch.tensor([reward_fn(g, r) for g, r in zip(generated_texts, decoded_labels)], dtype=torch.float32).to(device)
            # Adjust loss
            loss = loss - alpha * rewards.mean()

        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device, tokenizer=None, rl=False, reward_fn=None, max_output_length=128):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    total_reward = 0

    progress_bar = tqdm(dataloader, desc="Validation", ncols=100, dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", None)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            if rl and reward_fn and labels is not None:
                # Generate predictions
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_output_length
                )
                # Decode
                generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

                # Compute rewards
                rewards = [reward_fn(g, r) for g, r in zip(generated_texts, decoded_labels)]
                total_reward += sum(rewards) / len(rewards)

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), reward=rewards[0])
            else:
                progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    if rl:
        avg_reward = total_reward / len(dataloader)
        return avg_loss, avg_reward
    return avg_loss
