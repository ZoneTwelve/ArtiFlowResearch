import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import argparse
import re
from tqdm import tqdm  # For progress bar

# Process function to parse generated titles
def process(generated_title, format='str'):
    # Regex to capture text inside <title>{text}</title>
    titles = re.findall(r"<title>(.*?)</title>", generated_title)

    if format == 'str':
        return titles[0] if titles else ""
    elif format == 'list':
        return titles
    return ""

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Inference script for title generation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (in JSONL format)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--max-output-length", type=int, default=128, help="Maximum output length for generated titles")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model on")

    args = parser.parse_args()

    # Load model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.model_path, use_fast=True)  # Use fast tokenizer for speedup

    # Load test dataset
    with open(args.input, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    # Increase batch size but carefully manage GPU memory usage
    batch_size = 64  # Keep a smaller batch size to avoid memory issues
    results = []
    length_data = len(test_data)

    # Use tqdm to display progress bar
    for idx in tqdm(range(0, len(test_data), batch_size), desc="Processing Batches", total=(length_data // batch_size) + 1):
        batch = test_data[idx:idx + batch_size]
        articles = [sample['text'] for sample in batch]

        # Manually tokenize and prepare inputs for inference
        inputs = tokenizer(articles, padding=True, truncation=True, return_tensors="pt", max_length=args.max_output_length).to(device)

        # Run inference on the batch
        with torch.no_grad():  # Disable gradient calculations for inference
            outputs = model.generate(
                **inputs,
                max_length=args.max_output_length,
                num_beams=1,  # Reduce beams for faster inference
                do_sample=True,
                top_k=50,  # Can experiment with this for speed vs. quality tradeoff
                top_p=0.95
            )

        # Decode the outputs and process each generated title
        for i, sample in enumerate(batch):
            generated_title = tokenizer.decode(outputs[i], skip_special_tokens=True)
            processed_title = process(generated_title, format='str')
            results.append({
                'article': sample['text'],
                'title': processed_title
            })

    # Save results to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Inference completed and saved to {args.output}")

if __name__ == "__main__":
    main()

