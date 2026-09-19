"""
ONNX Export and Inference Benchmark
Exports DistilBERT to ONNX and benchmarks against PyTorch
"""

import torch
import time
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"
ONNX_PATH = "results/distilbert.onnx"
NUM_RUNS = 100
WARMUP_RUNS = 10

def export_to_onnx(model, tokenizer):
    os.makedirs("results", exist_ok=True)
    sample = tokenizer(
        "Optimizing LLMs for target hardware architectures.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    print("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"  ONNX model saved: {ONNX_PATH}")
    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"  ONNX model size: {size_mb:.2f}MB")
    return size_mb

def benchmark_pytorch(model, inputs, num_runs=NUM_RUNS):
    model.eval()
    device = torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(**inputs)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
    }

def benchmark_onnx(inputs, num_runs=NUM_RUNS):
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)

    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }

    for _ in range(WARMUP_RUNS):
        _ = session.run(None, onnx_inputs)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, onnx_inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
    }

def run_onnx_benchmark():
    print("=" * 60)
    print("ONNX EXPORT AND INFERENCE BENCHMARK")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(
        "Optimizing large language models for efficient inference on target hardware.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    onnx_size = export_to_onnx(model, tokenizer)

    print("\n  Benchmarking PyTorch FP32 (CPU)...")
    pytorch_results = benchmark_pytorch(model, inputs)
    print(f"    Mean: {pytorch_results['mean_ms']}ms | P95: {pytorch_results['p95_ms']}ms")

    print("\n  Benchmarking ONNX Runtime (CPU)...")
    onnx_results = benchmark_onnx(inputs)
    print(f"    Mean: {onnx_results['mean_ms']}ms | P95: {onnx_results['p95_ms']}ms")

    speedup = pytorch_results["mean_ms"] / onnx_results["mean_ms"]

    print("\n" + "=" * 60)
    print("ONNX RESULTS SUMMARY")
    print("=" * 60)
    print(f"  PyTorch FP32:  {pytorch_results['mean_ms']}ms")
    print(f"  ONNX Runtime:  {onnx_results['mean_ms']}ms")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  ONNX Size:     {onnx_size:.2f}MB")

    results = {
        "pytorch_fp32": pytorch_results,
        "onnx_runtime": onnx_results,
        "speedup": round(speedup, 2),
        "onnx_size_mb": round(onnx_size, 2)
    }

    with open("results/onnx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n  Results saved to results/onnx_results.json")
    print("=" * 60)

if __name__ == "__main__":
    run_onnx_benchmark()
