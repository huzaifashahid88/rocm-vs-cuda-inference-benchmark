"""
GPU Inference Benchmark: CUDA (NVIDIA) vs ROCm (AMD)

The SAME script runs on both. That's the point: AMD's PyTorch build reuses the
torch.cuda API (under the hood it's HIP, AMD's CUDA-like layer), so code written
for NVIDIA usually runs on AMD with zero changes.

What it measures, for every model x batch size x precision combo:
  - latency    : how long one batch takes (ms). Lower = better.
  - throughput : how many samples per second. Higher = better.
  - peak memory: most GPU memory used during the run (MB).

Run it:  python bench.py
Output:  a CSV file named after the backend and GPU, plus a table printed to screen.
"""

import csv
import time
import sys

import torch

# ---------------- Settings (change these to add more configs) ----------------
MODELS = ["resnet50", "bert-base", "qwen-0.5b"]   # vision, NLP, small LLM
BATCH_SIZES = [1, 4, 16, 32]                       # how many inputs per forward pass
PRECISIONS = ["fp32", "fp16"]                      # full vs half precision numbers
WARMUP = 5      # untimed runs first, so one-time setup cost doesn't skew results
ITERS = 20      # timed runs, averaged
SEQ_LEN = 128   # tokens per text input for BERT / Qwen
# 3 models x 4 batch sizes x 2 precisions = 24 configurations
# ------------------------------------------------------------------------------


def backend_info():
    """Figure out if we're on NVIDIA (CUDA) or AMD (ROCm), and which GPU."""
    if not torch.cuda.is_available():
        sys.exit("No GPU found. In Colab: Runtime > Change runtime type > GPU.")
    # torch.version.hip is only set on the AMD (ROCm) build of PyTorch
    backend = "ROCm" if torch.version.hip else "CUDA"
    version = torch.version.hip or torch.version.cuda
    gpu = torch.cuda.get_device_name(0)
    return backend, version, gpu


def load_model(name):
    """Return (model, function that makes a fake input batch of size bs)."""
    if name == "resnet50":
        # Random weights are fine: we're timing speed, not accuracy.
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        make_input = lambda bs: torch.randn(bs, 3, 224, 224)  # bs images, 224x224 RGB
    elif name == "bert-base":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        make_input = lambda bs: torch.randint(0, 30000, (bs, SEQ_LEN))  # fake token IDs
    elif name == "qwen-0.5b":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        make_input = lambda bs: torch.randint(0, 50000, (bs, SEQ_LEN))
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.eval(), make_input  # eval() = inference mode, not training


def time_model(model, x):
    """Run the model ITERS times and return (latency_ms, peak_memory_mb)."""
    with torch.inference_mode():  # no gradient tracking, faster and less memory
        for _ in range(WARMUP):
            model(x)
        # GPUs run asynchronously. synchronize() waits until the GPU is actually
        # done, otherwise the timer would stop before the work finishes.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(ITERS):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / ITERS * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return latency_ms, peak_mb


def main():
    backend, version, gpu = backend_info()
    print(f"\nBackend: {backend} {version} | GPU: {gpu} | PyTorch {torch.__version__}\n")

    rows = []
    for name in MODELS:
        for prec in PRECISIONS:
            dtype = torch.float16 if prec == "fp16" else torch.float32
            try:
                model, make_input = load_model(name)
                model = model.to("cuda", dtype=dtype)  # "cuda" means AMD GPU on ROCm too
            except Exception as e:
                print(f"[LOAD FAIL] {name} {prec}: {e}")
                rows.append([backend, gpu, name, prec, "-", "-", "-", "-", f"load error: {e}"])
                continue

            for bs in BATCH_SIZES:
                x = make_input(bs).to("cuda")
                if x.is_floating_point():  # images need to match model precision
                    x = x.to(dtype)        # (token IDs stay as integers)
                try:
                    latency, mem = time_model(model, x)
                    throughput = bs / (latency / 1000)
                    status = "ok"
                except RuntimeError as e:  # usually out-of-memory
                    latency = throughput = mem = float("nan")
                    status = f"error: {str(e).splitlines()[0][:80]}"
                    torch.cuda.empty_cache()

                print(f"{name:10} {prec:5} bs={bs:3}  "
                      f"{latency:9.2f} ms  {throughput:10.1f} samples/s  {mem:9.0f} MB  {status}")
                rows.append([backend, gpu, name, prec, bs,
                             round(latency, 3), round(throughput, 1), round(mem, 1), status])

            del model                 # free GPU memory before the next model
            torch.cuda.empty_cache()

    safe_gpu = gpu.replace(" ", "_").replace("/", "-")
    out = f"results_{backend}_{safe_gpu}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "gpu", "model", "precision", "batch_size",
                    "latency_ms", "throughput_samples_per_s", "peak_mem_mb", "status"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out}")


if __name__ == "__main__":
    main()
