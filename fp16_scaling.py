"""
Follow-up experiment: does the fp16 speedup keep growing as batch size grows?

Hypothesis: at small batch sizes the MI300X is not busy enough for precision to
matter (time is dominated by fixed overhead), so fp16 only pays off once the
batch is big enough that the math takes real time.

Same models, same timing method as bench.py, just larger batch sizes.
Run it:  python fp16_scaling.py
Output:  fp16_scaling_<backend>_<gpu>.csv and a speedup table.
"""

import csv
import time
import sys

import torch

BATCH_SIZES = {
    "resnet50": [32, 64, 128, 256, 512],
    "bert-base": [32, 64, 128, 256, 512],
    "qwen-0.5b": [32, 64, 128, 256],   # logits are huge for this model, cap it
}
WARMUP = 5
ITERS = 20
SEQ_LEN = 128


def load_model(name):
    if name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        make_input = lambda bs: torch.randn(bs, 3, 224, 224)
    elif name == "bert-base":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        make_input = lambda bs: torch.randint(0, 30000, (bs, SEQ_LEN))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        make_input = lambda bs: torch.randint(0, 50000, (bs, SEQ_LEN))
    return model.eval(), make_input


def time_model(model, x):
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / ITERS * 1000


def main():
    if not torch.cuda.is_available():
        sys.exit("No GPU found.")
    backend = "ROCm" if torch.version.hip else "CUDA"
    gpu = torch.cuda.get_device_name(0)
    print(f"\nBackend: {backend} | GPU: {gpu} | PyTorch {torch.__version__}\n")

    lat = {}   # (model, precision, bs) -> latency ms
    for name, sizes in BATCH_SIZES.items():
        for prec in ("fp32", "fp16"):
            dtype = torch.float16 if prec == "fp16" else torch.float32
            model, make_input = load_model(name)
            model = model.to("cuda", dtype=dtype)
            for bs in sizes:
                x = make_input(bs).to("cuda")
                if x.is_floating_point():
                    x = x.to(dtype)
                try:
                    lat[(name, prec, bs)] = time_model(model, x)
                except RuntimeError as e:
                    lat[(name, prec, bs)] = float("nan")
                    print(f"[ERROR] {name} {prec} bs={bs}: {str(e).splitlines()[0][:80]}")
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()

    rows = []
    print(f"{'model':10} {'bs':>4} {'fp32 ms':>9} {'fp16 ms':>9} {'speedup':>8}")
    for name, sizes in BATCH_SIZES.items():
        for bs in sizes:
            a, b = lat[(name, "fp32", bs)], lat[(name, "fp16", bs)]
            sp = a / b
            print(f"{name:10} {bs:4} {a:9.2f} {b:9.2f} {sp:7.2f}x")
            rows.append([backend, gpu, name, bs, round(a, 3), round(b, 3), round(sp, 3)])

    safe_gpu = gpu.replace(" ", "_").replace("/", "-")
    out = f"fp16_scaling_{backend}_{safe_gpu}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "gpu", "model", "batch_size", "fp32_ms", "fp16_ms", "fp16_speedup"])
        w.writerows(rows)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
