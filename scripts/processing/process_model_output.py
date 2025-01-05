# scripts/processing/process_model_output.py

import argparse
import json
from utils.data_utils import load_jsonl, save_jsonl

def process_dataset(input_file, output_file, pattern="<title>{}</title>"):
    data = load_jsonl(input_file)
    processed_data = []
    for item in data:
        title = item.get('title', '')
        if '$title' not in title and title != '':
            item['title'] = pattern.format(title)
            processed_data.append(item)
    save_jsonl(processed_data, output_file)
    print(f"Processed data saved to {output_file}")

def main(args):
    process_dataset(args.input_file, args.output_file, pattern=args.pattern)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model output for training")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the processed JSONL file")
    parser.add_argument("--pattern", type=str, default="<title>{}</title>", help="Pattern to wrap the title")
    args = parser.parse_args()
    main(args)
