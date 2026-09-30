"""Check that this machine can run the training and serving stack.

    python scripts/env_check.py --tag 5070ti            # full check incl. QLoRA seq-len sweep
    python scripts/env_check.py --tag 5070ti --quick    # skip the sweep on the large model

Checks, each in its own subprocess so an OOM or CUDA error in one doesn't take
down the rest:

  system       torch/CUDA versions, GPU name, compute capability, memory
  matmul       a bf16 matmul on the GPU (catches "no kernel image" on new GPUs)
  qlora        4-bit NF4 load + LoRA + one forward/backward/optimizer step,
               with peak memory; first on a small model, then a seq-len sweep
               on --qlora-model to see what fits
  vllm         load a small model in vLLM and generate a few tokens
  flash_attn   whether flash-attn imports (optional; SDPA works without it)

On a Mac the CUDA-only checks are reported as "skip". Results are written to
results/env_check/<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_PREFIX = "__RESULT__"
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------- checks
# Each check returns a dict with at least "status" in {ok, warn, fail, skip}.


def check_system(_: dict) -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "packages": {p: _pkg_version(p) for p in ("transformers", "peft", "trl", "bitsandbytes", "vllm", "flash_attn")},
    }
    notes = []
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        info.update(
            gpu=props.name,
            capability=f"{cap[0]}.{cap[1]}",
            gpu_memory_gib=round(props.total_memory / 2**30, 1),
            bf16_supported=torch.cuda.is_bf16_supported(),
            arch_list=torch.cuda.get_arch_list(),
        )
        # Blackwell consumer cards (RTX 50xx) are sm_120 and need CUDA >= 12.8 builds.
        if cap[0] >= 12:
            cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
            if cuda_ver < (12, 8):
                notes.append(f"sm_{cap[0]}{cap[1]} needs a CUDA>=12.8 torch build, found CUDA {torch.version.cuda}")
            if f"sm_{cap[0]}{cap[1]}" not in info["arch_list"]:
                notes.append(f"torch was not compiled for sm_{cap[0]}{cap[1]}: {info['arch_list']}")
    info["notes"] = notes
    info["status"] = "warn" if notes else "ok"
    return info


def check_matmul(_: dict) -> dict:
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        return {"status": "skip", "reason": "no GPU"}
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = a @ a.T
    value = float(b.float().abs().mean().cpu())
    return {"status": "ok" if value == value else "fail", "device": device, "mean_abs": round(value, 3)}


def check_qlora(args: dict) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "no CUDA (bitsandbytes 4-bit training needs a CUDA GPU)"}
    if _pkg_version("bitsandbytes") is None or _pkg_version("peft") is None:
        return {"status": "fail", "reason": "install with: pip install -e '.[train,gpu]'"}

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_name, seq_len = args["model"], args["seq_len"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(100, 10_000, (1, seq_len), device="cuda")
        start = time.time()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    except torch.cuda.OutOfMemoryError:
        return {"status": "fail", "model": model_name, "seq_len": seq_len, "reason": "CUDA out of memory"}

    return {
        "status": "ok" if torch.isfinite(loss).item() else "fail",
        "model": model_name,
        "seq_len": seq_len,
        "loss": round(loss.item(), 3),
        "trainable_params_m": round(trainable / 1e6, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "step_seconds": round(elapsed, 2),
    }


def check_vllm(args: dict) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "no CUDA"}
    if _pkg_version("vllm") is None:
        return {"status": "fail", "reason": "vllm not installed (pip install -e '.[gpu]')"}

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(model=args["model"], max_model_len=2048, gpu_memory_utilization=0.6, enforce_eager=True, seed=0)
    load_seconds = time.time() - start
    out = llm.generate(["The three most important numbers in an earnings report are"], SamplingParams(temperature=0, max_tokens=16))
    text = out[0].outputs[0].text
    return {"status": "ok" if text.strip() else "fail", "model": args["model"], "load_seconds": round(load_seconds, 1), "sample": text}


def check_flash_attn(_: dict) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "no CUDA"}
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:  # ImportError, or an undefined-symbol error from a bad build
        return {"status": "warn", "reason": f"not usable ({type(exc).__name__}); SDPA attention will be used"}
    return {"status": "ok", "version": flash_attn.__version__}


CHECKS = {
    "system": check_system,
    "matmul": check_matmul,
    "qlora": check_qlora,
    "vllm": check_vllm,
    "flash_attn": check_flash_attn,
}


# ---------------------------------------------------------------- runner


def run_isolated(name: str, args: dict, timeout: int) -> dict:
    """Run one check in a fresh Python process and return its result dict."""
    cmd = [sys.executable, __file__, "--run-check", name, "--check-args", json.dumps(args)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "fail", "reason": f"timed out after {timeout}s", **args}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
    return {"status": "fail", "reason": f"exit code {proc.returncode}", "stderr_tail": tail, **args}


def _summary(result: dict) -> str:
    skip = {"status", "packages", "arch_list", "notes", "stderr_tail", "platform"}
    parts = [f"{k}={v}" for k, v in result.items() if k not in skip]
    for note in result.get("notes", []):
        parts.append(f"NOTE: {note}")
    for line in result.get("stderr_tail", []):
        parts.append(f"\n      {line}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="local", help="name for the output file, e.g. 5070ti or hyak-a40")
    parser.add_argument("--qlora-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seq-lens", default="2048,4096,6144")
    parser.add_argument("--quick", action="store_true", help="only the small-model QLoRA check")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per check (includes model download)")
    parser.add_argument("--run-check", help=argparse.SUPPRESS)
    parser.add_argument("--check-args", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_check:
        result = CHECKS[args.run_check](json.loads(args.check_args))
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    plan = [
        ("system", {}),
        ("matmul", {}),
        ("qlora", {"model": SMALL_MODEL, "seq_len": 512}),
    ]
    if not args.quick:
        plan += [("qlora", {"model": args.qlora_model, "seq_len": int(s)}) for s in args.seq_lens.split(",")]
    plan += [("vllm", {"model": SMALL_MODEL}), ("flash_attn", {})]

    results = []
    skip_sweep = False
    for name, check_args in plan:
        if name == "qlora" and skip_sweep:
            continue
        print(f"[{name}] {check_args or ''} ...", flush=True)
        result = run_isolated(name, check_args, args.timeout)
        results.append({"check": name, **result})
        print(f"  {result['status'].upper():4s}  {_summary(result)}", flush=True)
        # The rest of the QLoRA sweep is pointless if the small model already failed.
        if name == "qlora" and check_args.get("model") == SMALL_MODEL and result["status"] in ("fail", "skip"):
            skip_sweep = True

    out = ROOT / "results" / "env_check" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": args.tag, "results": results}, indent=2) + "\n")
    print(f"\nWrote {out}")
    if any(r["status"] == "fail" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
