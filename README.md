# LLM Inference Optimization

Benchmarking and optimizing DistilBERT inference across FP32, FP16, INT8 quantization, and ONNX runtime on CPU and GPU — targeting production-grade latency and memory efficiency for edge and target-architecture deployment.

## Motivation

Deploying LLMs in production requires balancing accuracy, latency, and model size. This project systematically benchmarks optimization techniques across hardware configurations to identify the best latency-memory tradeoff for each deployment scenario.

## Optimization Techniques

| Technique | Hardware | Description |
|---|---|---|
| FP32 baseline | CPU / GPU | Full precision, no optimization |
| FP16 half-precision | GPU | 2x memory reduction, near-zero accuracy loss |
| INT8 dynamic quantization | CPU | Linear layers quantized to 8-bit, 2.8x size reduction |
| ONNX Runtime | CPU | Graph-level optimizations, hardware-agnostic deployment |

## Results

### Latency Benchmark (Batch Size = 1, Sequence Length = 128)

| Configuration | Mean Latency | P95 Latency | Model Size | Speedup vs FP32 CPU |
|---|---|---|---|---|
| FP32 CPU | 40.65ms | 44.49ms | 255.42MB | 1.0x |
| FP32 GPU | 4.62ms | 5.61ms | 255.42MB | 8.8x |
| FP16 GPU | 4.43ms | 5.25ms | 127.71MB | **9.2x** |
| INT8 CPU | 14.98ms | 22.12ms | 91.00MB | 2.7x |
| ONNX Runtime | 49.37ms | 53.80ms | 255.55MB | 1.76x vs PyTorch CPU |

### Batch Size Scaling

| Batch Size | FP32 CPU | FP32 GPU | FP16 GPU | INT8 CPU |
|---|---|---|---|---|
| 1 | 40.65ms | 4.62ms | 4.43ms | 14.98ms |
| 8 | 79.13ms | 6.77ms | 6.08ms | 32.30ms |
| 16 | 143.22ms | 11.00ms | **5.01ms** | 58.13ms |

FP16 GPU throughput scales significantly better than all other configurations at higher batch sizes — at batch=16, FP16 GPU is **28.6x faster** than FP32 CPU (143.22ms vs 5.01ms).

### Key Findings

- FP16 GPU achieves **9.2x** speedup over FP32 CPU at batch=1 with **2x memory reduction** (255MB → 128MB)
- FP16 GPU advantage compounds at scale — **28.6x faster** than FP32 CPU at batch=16
- INT8 dynamic quantization reduces model size by **2.8x** (255MB → 91MB) with CPU-only deployment
- ONNX Runtime achieves **1.76x** faster CPU inference through graph-level optimizations
- FP32 GPU delivers **8.8x** speedup with zero model modification

## Setup

```bash
conda create -n llm-opt python=3.10
conda activate llm-opt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers onnx onnxruntime
```

## Run

```bash
# Full benchmark: FP32 CPU, FP32 GPU, FP16 GPU, INT8 CPU across batch sizes 1, 8, 16
python benchmark.py

# ONNX export and CPU inference benchmark
python export_onnx.py
```

Results saved to `results/benchmark_results.json` and `results/onnx_results.json`.

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3050 |
| CUDA | 12.1 |
| PyTorch | 2.5.1 |
| ONNX Runtime | 1.23.2 |
| Transformers | 5.2.0 |
| Model | distilbert-base-uncased |
| Sequence Length | 128 tokens |
| Benchmark Runs | 100 (10 warmup) |
