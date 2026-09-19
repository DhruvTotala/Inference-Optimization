"""
LLM Inference Optimization Benchmark
Compares FP32, FP16, INT8 quantization and ONNX runtime
Model: distilbert-base-uncased
"""

import torch
import time
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"
NUM_RUNS = 100
WARMUP_RUNS = 10
BATCH_SIZES = [1, 8, 16]

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

def benchmark_latency(model, inputs, device, num_runs=NUM_RUNS, warmup=WARMUP_RUNS):
    model.eval()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
    }

def run_benchmarks():
    print("=" * 60)
    print("LLM INFERENCE OPTIMIZATION BENCHMARK")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample_text = "Optimizing large language models for target hardware architectures improves inference efficiency."

    results = {}

    for batch_size in BATCH_SIZES:
        print(f"\n[Batch Size: {batch_size}]")
        results[f"batch_{batch_size}"] = {}

        inputs = tokenizer(
            [sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # --- FP32 CPU ---
        print("  Running FP32 (CPU)...")
        model_fp32_cpu = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model_fp32_cpu.eval()
        cpu_device = torch.device("cpu")
        latency = benchmark_latency(model_fp32_cpu, inputs, cpu_device)
        size = get_model_size_mb(model_fp32_cpu)
        results[f"batch_{batch_size}"]["fp32_cpu"] = {"latency": latency, "size_mb": round(size, 2)}
        print(f"    Mean latency: {latency['mean_ms']}ms | Size: {size:.2f}MB")

        # --- FP32 GPU ---
        print("  Running FP32 (GPU)...")
        gpu_device = torch.device("cuda")
        model_fp32_gpu = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(gpu_device)
        model_fp32_gpu.eval()
        latency = benchmark_latency(model_fp32_gpu, inputs, gpu_device)
        results[f"batch_{batch_size}"]["fp32_gpu"] = {"latency": latency, "size_mb": round(size, 2)}
        print(f"    Mean latency: {latency['mean_ms']}ms | Size: {size:.2f}MB")
        del model_fp32_gpu
        torch.cuda.empty_cache()

        # --- FP16 GPU ---
        print("  Running FP16 (GPU)...")
        model_fp16 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).half().to(gpu_device)
        model_fp16.eval()
        inputs_fp16 = {k: v.to(gpu_device) for k, v in inputs.items()}
        latency = benchmark_latency(model_fp16, inputs, gpu_device)
        size_fp16 = get_model_size_mb(model_fp16)
        results[f"batch_{batch_size}"]["fp16_gpu"] = {"latency": latency, "size_mb": round(size_fp16, 2)}
        print(f"    Mean latency: {latency['mean_ms']}ms | Size: {size_fp16:.2f}MB")
        del model_fp16
        torch.cuda.empty_cache()

        # --- INT8 Dynamic Quantization (CPU) ---
        print("  Running INT8 Dynamic Quantization (CPU)...")
        model_int8 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model_int8 = quantize_dynamic(
            model_int8,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        model_int8.eval()
        latency = benchmark_latency(model_int8, inputs, cpu_device)
        size_int8 = get_model_size_mb(model_int8)
        results[f"batch_{batch_size}"]["int8_cpu"] = {"latency": latency, "size_mb": round(size_int8, 2)}
        print(f"    Mean latency: {latency['mean_ms']}ms | Size: {size_int8:.2f}MB")

    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY (Batch Size = 1)")
    print("=" * 60)
    b1 = results["batch_1"]
    fp32_cpu_lat = b1["fp32_cpu"]["latency"]["mean_ms"]
    fp32_gpu_lat = b1["fp32_gpu"]["latency"]["mean_ms"]
    fp16_lat = b1["fp16_gpu"]["latency"]["mean_ms"]
    int8_lat = b1["int8_cpu"]["latency"]["mean_ms"]
    fp32_size = b1["fp32_cpu"]["size_mb"]
    int8_size = b1["int8_cpu"]["size_mb"]

    print(f"  FP32 CPU:  {fp32_cpu_lat}ms | {fp32_size}MB")
    print(f"  FP32 GPU:  {fp32_gpu_lat}ms | {fp32_size}MB  | Speedup: {fp32_cpu_lat/fp32_gpu_lat:.1f}x over CPU")
    print(f"  FP16 GPU:  {fp16_lat}ms    | {b1['fp16_gpu']['size_mb']}MB  | Speedup: {fp32_cpu_lat/fp16_lat:.1f}x over FP32 CPU")
    print(f"  INT8 CPU:  {int8_lat}ms  | {int8_size}MB  | Speedup: {fp32_cpu_lat/int8_lat:.1f}x over FP32 CPU | Size reduction: {fp32_size/int8_size:.1f}x")
    print("\n  Results saved to results/benchmark_results.json")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmarks()
