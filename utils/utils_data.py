# utils/data_utils.py

import json
from sklearn.model_selection import train_test_split

def load_jsonl(file_path):
    """Load data from a JSON Lines file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSON Lines file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def split_data(data, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data
