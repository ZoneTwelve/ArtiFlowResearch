import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import argparse

# Sampling function for title generation
def generate_title(model, tokenizer, article, max_output_length, device):
    input_ids = tokenizer(article, truncation=True, padding="max_length", max_length=512, return_tensors="pt").input_ids.to(device)
    attention_mask = tokenizer(article, truncation=True, padding="max_length", max_length=512, return_tensors="pt").attention_mask.to(device)
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,
        num_beams=1,  # Sampling strategy
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Inference script for title generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file (in JSONL format)")
    parser.add_argument("--max_output_length", type=int, default=128, help="Maximum output length for generated titles")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model on")
    
    args = parser.parse_args()

    # Load model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    # Load test dataset
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    # Generate titles for each article in the test dataset
    for idx, sample in enumerate(test_data):
        #article = sample["article"]
        article = sample["text"]
        generated_title = generate_title(model, tokenizer, article, args.max_output_length, device)
        print(f"Article {idx + 1}:")
        print(f"Article: {article[:100]}")
        #print(f"Some excpet title: {sample['title'][:100]}")
        print(f"Generated Title: {generated_title}")
        print("-" * 50)

if __name__ == "__main__":
    main()

