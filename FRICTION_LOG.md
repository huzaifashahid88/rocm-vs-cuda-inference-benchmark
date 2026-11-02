# Friction Log: Getting Onto an AMD GPU (ROCm) vs NVIDIA (CUDA)

Project: PyTorch inference benchmark (ResNet-50, BERT-base, Qwen2.5-0.5B), 24 configs per platform.
NVIDIA side: Google Colab, Tesla T4. AMD side: Spheron, 1x MI300X (ROCm 7.16, PyTorch 2.14).
Times are approximate, from memory.

## Issues (6 core, 1 minor)

| # | Platform | What I was doing | What went wrong | Layer | Time cost | How I got past it | Who it hurts |
|---|---|---|---|---|---|---|---|
| 1 | ROCm | Creating a GPU droplet on AMD Developer Cloud | Droplet creation was blocked: "request complimentary credit or request a limit increase" before any GPU could be created | Access / onboarding | 5+ days, never resolved | Gave up on it and rented from other providers | Every new developer, especially students with no budget |
| 2 | ROCm | Requesting credits / access | Several overlapping forms (member-site credit form, droplet access form) asked for use case and GPU count; AMD then asked for "reason for requiring additional GPUs" when I only needed 1; no status or timeline visible | Access / onboarding | Days of waiting plus 3+ form submissions | Resubmitted with project details | New developers who can't tell if they're approved or stuck |
| 3 | ROCm | Renting an MI300X on RunPod | "Low availability" then "Out of capacity" for the MI300X; the only workaround ("deploy when available") starts billing whenever it auto-deploys | Capacity | ~5 min (covers issues 3 and 4 together) | Moved to another provider | Anyone trying AMD GPUs on demand |
| 4 | ROCm | Configuring a pod on RunPod | The default template was marked "incompatible" with the MI300X; it's built for NVIDIA GPUs and the error didn't say what to pick instead | Platform template / compatibility | Included in the ~5 min above | Would have switched to an AMD (ROCm) template; ran out of capacity first | Beginners who don't know templates are vendor-specific |
| 5 | ROCm | Paying to rent a GPU | Hot Aisle needed a $20 minimum deposit; Spheron needed $15. Spheron also has a 20-minute minimum runtime and no stop/start, so it bills until destroyed | Cost / billing | ~12 min total (Hot Aisle ~2 min, Spheron ~10 min) | Paid the $15 deposit on Spheron | Students and anyone running a short experiment |
| 6 | ROCm | Running the script on the AMD VM | The Ubuntu (ROCm) VM had ROCm and Docker but no PyTorch on the machine itself (`No module named 'torch'`); it worked once I ran AMD's rocm/pytorch container, which a beginner wouldn't know to do | Framework (PyTorch) layer | ~3 min to download the container image, and only because I was guided | Ran the rocm/pytorch container | Beginners who assume "ROCm image" means ready to use |
| 7 (minor) | ROCm | Running the benchmark | Harmless warnings looked like errors: an unauthenticated Hugging Face warning, a BERT "UNEXPECTED" keys load report, and a pip warning about running as root | Framework / docs | ~0 min, but alarming for a beginner | Ignored them | Beginners |

## What was surprisingly easy
- The exact same script ran on NVIDIA and AMD with zero code changes (AMD's PyTorch build reuses the CUDA-style interface).
- Once on the machine, the AMD GPU was detected immediately (ROCm/HIP 7.16, "AMD Instinct MI300X VF").
- All 24 configs ran on the first try on the AMD side, with no errors.
- The Spheron instance booted to "Running" in about a minute.
- Installing `transformers` and running the full 24-config benchmark took about 20 seconds each (the container image download was the slow part, ~3 min).
- NVIDIA on Colab: nothing broke and nothing confused me. The setup was quick and smooth (free T4 GPU, upload the script, run). That contrast with AMD access is the point. Fair caveat: I was following step-by-step instructions on both sides.

## Minor confusions (a beginner's view, since I'm the target new developer)
- I couldn't find the "Create GPU Droplet" button on AMD Developer Cloud, because it was gated behind the access request, and the page didn't make that obvious.
- An informational box ("PyTorch requires at least an MI300X x1 plan") read like an error, and I thought an option was missing.
- I wasn't sure what an SSH key was or where to find mine, and the instructions assume you already know.
- Minimum deposits ($20 on Hot Aisle, $15 on Spheron) appeared only at checkout, after I'd configured everything.

## Totals
- Money: $15 deposited on Spheron, about $4 actually used for the run.
- Hands-on time on the successful path: about 1.5 hours.

## Key insight
Once I was on the machine, the ROCm software was easy. Getting access to an AMD GPU was the hard part. The bottleneck was onboarding, capacity, and cost, not the software.

## Prioritization: what I'd fix, in order
Ranked by developer impact vs effort. AMD doesn't control third-party clouds, so #3 and #4 are partner-enablement work. These come from one beginner's experience, so they're proposals, not proven problems.

1. **Access gating (issues 1, 2).** New developers couldn't create a GPU at all until they requested credits, then waited with no status, across several overlapping forms. A student will just try NVIDIA instead. Fix: a small, instant GPU allowance on one form with a visible approval status. Medium effort (mostly process and policy, not engineering). Highest impact.
2. **A ready-to-use PyTorch + ROCm setup (issue 6).** The VM had ROCm and Docker but no PyTorch on the machine; it worked once I used AMD's container, which a beginner wouldn't know to do. Fix: make a PyTorch-ready image the default on partner clouds. Low effort, solid impact, because every new user hits it.
3. **Capacity visibility (issue 3).** One provider showed the MI300X as out of capacity, and the only workaround auto-deployed and billed whenever it freed up. Developers need to see availability and wait times up front. High impact, high effort and partner-dependent, so a longer-term roadmap item.
4. **Template compatibility on partner platforms (issue 4).** A partner's default template was flagged incompatible with the AMD GPU without saying which one to use. Fix: a short compatibility note or a pre-selected AMD template. Low effort, needs a partner to make the change.
5. **Explain the harmless warnings in the quickstart (issue 7).** A few warnings looked like errors but were harmless. A short "safe to ignore" note would stop beginners from panicking. Low effort, lower impact.

Not ranked: pricing minimums (issue 5). It's a provider business decision, but it's still a real barrier for students.

## Honest limitations of the benchmark
- T4 (2018, free on Colab) vs MI300X (2023, about $3.59/hr to rent): not a fair speed comparison.
- Random weights and inputs: measures speed, not accuracy.
- One forward pass (not text generation), 20 timed runs averaged, no repeats or error bars.
- Different software versions on each side (PyTorch 2.11.0+cu130 on Colab vs 2.14.0+rocm10.1.0 on AMD).
- These small models use only a few GB of the MI300X's 192 GB, so they don't stress what the chip is built for.
