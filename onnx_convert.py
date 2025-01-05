#!/usr/bin/env python
# convert_mt5_to_onnx.py

import os
import sys
import torch
import onnx
import onnxruntime
import numpy as np
import fire
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Optional: Import optimizer if optimization is desired
try:
    from onnxruntime_tools import optimizer
    from onnxruntime_tools.transformers.onnx_model_mt5 import MT5OptimizationOptions
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

class MT5ONNXConverter:
    def __init__(
        self,
        model_path: str,
        onnx_output_path: str,
        dummy_text: str = "Translate English to German: The house is wonderful.",
        opset_version: int = 13,
        optimize: bool = False,
        optimized_model_path: str = None,
    ):
        """
        Initialize the converter with necessary parameters.

        Args:
            model_path (str): Path or Hugging Face model ID of the fine-tuned mT5-small model.
            onnx_output_path (str): Path where the ONNX model will be saved.
            dummy_text (str, optional): Dummy input text for tracing. Defaults to a translation example.
            opset_version (int, optional): ONNX opset version. Defaults to 13.
            optimize (bool, optional): Whether to optimize the ONNX model. Defaults to False.
            optimized_model_path (str, optional): Path to save the optimized ONNX model. Required if optimize=True.
        """
        self.model_path = model_path
        self.onnx_output_path = onnx_output_path
        self.dummy_text = dummy_text
        self.opset_version = opset_version
        self.optimize = optimize
        self.optimized_model_path = optimized_model_path

        if self.optimize and not OPTIMIZATION_AVAILABLE:
            raise ImportError("onnxruntime-tools is not installed. Install it via 'pip install onnxruntime-tools' to enable optimization.")

        if self.optimize and not self.optimized_model_path:
            raise ValueError("optimized_model_path must be provided if optimize=True.")

    def load_model_and_tokenizer(self):
        """Load the mT5-small model and tokenizer."""
        print(f"Loading model and tokenizer from '{self.model_path}'...")
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def prepare_dummy_input(self):
        """Prepare dummy inputs for ONNX export."""
        print(f"Preparing dummy input: '{self.dummy_text}'")
        self.inputs = self.tokenizer(self.dummy_text, return_tensors="pt")
        self.input_names = ["input_ids", "attention_mask"]
        self.output_names = ["output"]
        self.dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
        print("Dummy input prepared.")

    def export_to_onnx(self):
        """Export the PyTorch model to ONNX format."""
        print(f"Exporting the model to ONNX at '{self.onnx_output_path}'...")
        torch.onnx.export(
            self.model,
            (
                self.inputs["input_ids"],
                self.inputs["attention_mask"],
            ),
            self.onnx_output_path,
            opset_version=self.opset_version,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            do_constant_folding=True,
        )
        print("Model exported to ONNX successfully.")

    def verify_onnx_model(self):
        """Verify the exported ONNX model."""
        print("Verifying the ONNX model...")
        # Load and check the ONNX model
        onnx_model = onnx.load(self.onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")

        # Create an ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(self.onnx_output_path)

        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            "input_ids": self.inputs["input_ids"].numpy(),
            "attention_mask": self.inputs["attention_mask"].numpy(),
        }

        # Run inference with ONNX
        ort_outs = ort_session.run(None, ort_inputs)

        # Run inference with PyTorch
        with torch.no_grad():
            torch_out = self.model.generate(**self.inputs)

        # Decode outputs
        onnx_output = self.tokenizer.decode(ort_outs[0][0], skip_special_tokens=True)
        torch_output = self.tokenizer.decode(torch_out[0], skip_special_tokens=True)

        print(f"ONNX Output: {onnx_output}")
        print(f"PyTorch Output: {torch_output}")

        if onnx_output == torch_output:
            print("Verification successful: ONNX model output matches PyTorch model output.")
        else:
            print("Warning: ONNX model output does not match PyTorch model output.")

    def optimize_onnx_model(self):
        """Optimize the ONNX model for better performance."""
        if not self.optimize:
            print("Optimization not requested. Skipping optimization step.")
            return

        print("Optimizing the ONNX model...")
        # Define optimization options specific to mT5
        opt_options = MT5OptimizationOptions()

        # Optimize the model
        optimized_model = optimizer.optimize_model(
            self.onnx_output_path,
            model_type='mt5',  # Specify model type as mT5
            optimization_options=opt_options
        )

        # Save the optimized model
        optimized_model.save_model_to_file(self.optimized_model_path)
        print(f"Optimized ONNX model saved to '{self.optimized_model_path}'.")

    def run_inference_with_onnx(self):
        """Run inference using the ONNX model to demonstrate deployment."""
        print("Running inference with the ONNX model...")
        # Choose which model to use (optimized or original)
        model_path = self.optimized_model_path if self.optimize else self.onnx_output_path

        ort_session = onnxruntime.InferenceSession(model_path)

        # Prepare inputs
        inputs = self.tokenizer("Translate English to German: The weather is nice today.", return_tensors="np")

        ort_inputs = {k: v for k, v in inputs.items()}

        # Run inference
        ort_outs = ort_session.run(None, ort_inputs)

        # Decode the output
        translated_text = self.tokenizer.decode(ort_outs[0][0], skip_special_tokens=True)
        print(f"Translated Text: {translated_text}")

    def convert(self):
        """Execute the full conversion process."""
        self.load_model_and_tokenizer()
        self.prepare_dummy_input()
        self.export_to_onnx()
        self.verify_onnx_model()
        if self.optimize:
            self.optimize_onnx_model()
        self.run_inference_with_onnx()

def main(**kwargs):
    converter = MT5ONNXConverter(**kwargs)
    converter.convert()

if __name__ == "__main__":
    fire.Fire(main)

