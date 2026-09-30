"""Generate summaries with a model and score them against ECTSum references.

Mac smoke test (small model, few docs):
    python scripts/run_baseline_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 20

GPU baseline (full test split, vLLM):
    python scripts/run_baseline_eval.py --model Qwen/Qwen2.5-3B-Instruct --backend vllm

Writes predictions.jsonl and metrics.json under results/eval/<run name>/.
Generation settings come from configs/eval.yaml.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from finpost.data import store
from finpost.eval.report import aggregate, score_example
from finpost.prompts import PROMPT_VERSION, build_prompt, postprocess_summary

ROOT = Path(__file__).resolve().parent.parent


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # fp32 on MPS/CPU: slower, but avoids fp16 overflow issues on small models.
    return torch.float32


def generate_hf(prompts, model_name, tokenizer, cfg, device, dtype, batch_size):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=cfg["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {len(outputs)}/{len(prompts)}", flush=True)
    return outputs


def generate_vllm(prompts, model_name, cfg, dtype_name):
    from vllm import LLM, SamplingParams

    # Prompt template adds a few dozen tokens on top of the transcript budget.
    llm = LLM(
        model=model_name,
        dtype=dtype_name,
        max_model_len=cfg["max_input_tokens"] + cfg["max_new_tokens"] + 256,
        seed=0,
    )
    params = SamplingParams(
        temperature=cfg["temperature"],
        max_tokens=cfg["max_new_tokens"],
        repetition_penalty=cfg["repetition_penalty"],
    )
    results = llm.generate(prompts, params)
    return [r.outputs[0].text for r in results]


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "vllm"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _git_state() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eval.yaml")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "ectsum.sqlite")
    parser.add_argument("--limit", type=int, default=None, help="first N docs of the split (by doc_id)")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", default="auto", help="hf backend only: auto, cuda, mps, cpu")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16, float32")
    parser.add_argument("--batch-size", type=int, default=1, help="hf backend only")
    parser.add_argument("--run-name", default=None, help="defaults to <model>_<split>_n<N>_in<budget>")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "eval")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    cfg = yaml.safe_load(args.config.read_text())
    docs = store.load_split(args.db, cfg["split"], limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts, truncated = [], []
    for transcript in docs["transcript"]:
        prompt, was_truncated = build_prompt(transcript, tokenizer, cfg["max_input_tokens"])
        prompts.append(prompt)
        truncated.append(was_truncated)

    device = dtype = None
    start = time.time()
    if args.backend == "hf":
        device = pick_device(args.device)
        dtype = pick_dtype(args.dtype, device)
        print(f"Generating {len(prompts)} summaries with {args.model} on {device} ({dtype}) ...")
        raw_outputs = generate_hf(prompts, args.model, tokenizer, cfg, device, dtype, args.batch_size)
    else:
        print(f"Generating {len(prompts)} summaries with {args.model} via vLLM ...")
        raw_outputs = generate_vllm(prompts, args.model, cfg, args.dtype)
    elapsed = time.time() - start

    rows, records = [], []
    for doc, raw, was_truncated in zip(docs.itertuples(), raw_outputs, truncated):
        prediction = postprocess_summary(raw)
        scores = score_example(prediction, doc.summary, doc.transcript, cfg["allow_rounding"])
        rows.append(scores)
        records.append(
            {
                "doc_id": doc.doc_id,
                "truncated": was_truncated,
                "prediction_raw": raw,
                "prediction": prediction,
                "reference": doc.summary,
                **scores,
            }
        )

    run_name = args.run_name or (
        f"{Path(args.model).name}_{cfg['split']}_n{len(docs)}_in{cfg['max_input_tokens']}"
    )
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    metrics = {
        "run": {
            "model": args.model,
            "backend": args.backend,
            "device": device,
            "dtype": str(dtype) if dtype is not None else args.dtype,
            "n_docs": len(docs),
            "limit": args.limit,
            "prompt_version": PROMPT_VERSION,
            "config": cfg,
            "share_truncated": sum(truncated) / len(truncated),
            "generation_seconds": round(elapsed, 1),
            "versions": _versions(),
            "git": _git_state(),
        },
        "metrics": aggregate(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nWrote {out_dir}")
    for name, m in metrics["metrics"].items():
        if isinstance(m, dict) and "ci95" in m and m["mean"] is not None:
            lo, hi = m["ci95"]
            print(f"  {name:16s} {m['mean']:.3f}  [{lo:.3f}, {hi:.3f}]  n={m['n']}")
    micro = metrics["metrics"]["num_halluc_rate_micro"]
    if micro["mean"] is not None:
        print(f"  {'halluc (micro)':16s} {micro['mean']:.3f}  over {micro['n_numbers']} numbers")


if __name__ == "__main__":
    main()
