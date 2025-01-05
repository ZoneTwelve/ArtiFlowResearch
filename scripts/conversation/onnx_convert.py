# scripts/conversion/onnx_convert.py

import argparse
import os
import torch
import onnx
import onnxruntime
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_mt5 import MT5OptimizationOptions
from utils.model_utils import ModelHandler
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def main(args):
    # Initialize model handler
    model_handler = ModelHandler(model_name=args.model_path, device='cpu')  # ONNX export on CPU
    model = model_handler.model
    tokenizer = model_handler.tokenizer

    # Prepare dummy input
    dummy_text = "Translate English to German: The house is wonderful."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Export to ONNX
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        args.onnx_output_path,
        opset_version=args.opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,
    )
    print(f"Model exported to {args.onnx_output_path}")

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(args.onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Optimize ONNX model if requested
    if args.optimize:
        if not args.optimized_model_path:
            raise ValueError("Optimized model path must be provided if optimization is enabled.")
        print("Optimizing ONNX model...")
        opt_options = MT5OptimizationOptions()
        optimized_model = optimizer.optimize_model(
            args.onnx_output_path,
            model_type='mt5',
            optimization_options=opt_options
        )
        optimized_model.save_model_to_file(args.optimized_model_path)
        print(f"Optimized ONNX model saved to {args.optimized_model_path}")

    # Run inference with ONNX model
    print("Running inference with ONNX model...")
    ort_session = onnxruntime.InferenceSession(args.onnx_output_path)
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    translated_text = tokenizer.decode(ort_outs[0][0], skip_special_tokens=True)
    print(f"Translated Text: {translated_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mT5 model to ONNX format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--onnx_output_path", type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument("--opset_version", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--optimize", action='store_true', help="Whether to optimize the ONNX model")
    parser.add_argument("--optimized_model_path", type=str, help="Path to save the optimized ONNX model")
    args = parser.parse_args()
    main(args)
