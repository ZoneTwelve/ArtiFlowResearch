# scripts/inference/gen.py

import argparse
import json
from transformers import pipeline
from utils.model_utils import ModelHandler
from utils.data_utils import load_jsonl
from utils.rl_utils import compute_reward
from tqdm import tqdm
import re

def process(generated_title, pattern=r"<title>(.*?)</title>", format='str'):
    """Extracts the title from the generated text."""
    titles = re.findall(pattern, generated_title)
    if format == 'str':
        return titles[0] if titles else ""
    elif format == 'list':
        return titles
    return ""

def main(args):
    # Initialize model handler
    model_handler = ModelHandler(model_name=args.model_path, device=args.device)
    model = model_handler.model
    tokenizer = model_handler.tokenizer

    # Load input data
    print("Loading input data...")
    input_data = load_jsonl(args.input)

    # Setup inference pipeline
    print("Setting up inference pipeline...")
    inference_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if args.device == "cuda" else -1
    )

    # Inference
    print("Starting inference...")
    results = []
    batch_size = args.batch_size
    total = len(input_data)

    for i in tqdm(range(0, total, batch_size), desc="Processing Batches", total=(total // batch_size) + 1):
        batch = input_data[i:i+batch_size]
        texts = [sample['text'] for sample in batch]

        # Run inference
        outputs = inference_pipeline(texts, max_length=args.max_output_length, num_beams=1, do_sample=True)

        # Process outputs
        for j, sample in enumerate(batch):
            generated_title = outputs[j]['generated_text']
            processed_title = process(generated_title)
            results.append({
                'article': sample['text'],
                'title': processed_title
            })

    # Save output
    print(f"Saving output to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for title generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (in JSONL format)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for inference.")
    parser.add_argument("--max_output_length", type=int, default=128, help="Maximum output length for generated titles")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model on")
    args = parser.parse_args()
    main(args)
