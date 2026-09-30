# fin-posttrain

Post-training a small open LLM to summarize earnings-call transcripts:
QLoRA SFT → DPO → quantized vLLM serving. Built on
[ECTSum](https://github.com/rajdeep345/ECTSum) (Mukherjee et al., EMNLP 2022).

**Status: work in progress.** This README only reports numbers produced by
the code in this repo, each linked to the result file it came from.

| Stage | Status |
|---|---|
| Data pipeline (download → clean → SQLite) | done |
| Evaluation (ROUGE, numeric F1, numeric hallucination rate, bootstrap CIs) | done |
| Baseline eval script (HF + vLLM backends) | done; HF path smoke-tested on Mac, vLLM path not yet run |
| GPU environment check | done; Mac path tested, CUDA path not yet run |
| Baseline on the full test split | next |
| QLoRA SFT | planned |
| DPO (preference pairs from SFT samples) | interface + tests written, implementation pending |
| GPTQ/AWQ + vLLM serving benchmark, Docker | planned |

## Data

ECTSum: 2,425 earnings-call transcripts, each paired with a short bullet-point
summary written by Reuters journalists. The official split (1681 / 249 / 495)
is kept unchanged. The data is GPL-3.0 and is not redistributed here;
`scripts/prepare_data.py` downloads a pinned commit.

Facts from [`results/data/ectsum_report.json`](results/data/ectsum_report.json)
that shape the rest of the project:

- **Transcripts are long.** On the test split, the median is 3,504 Qwen2.5
  tokens and p95 is 5,937. The summaries are short: median 34 words.
- **The input budget matters.** Transcripts are cut at line boundaries from the
  start. Share of reference figures (those that appear somewhere in the
  transcript) still present after the cut, test split:

  | budget (tokens) | docs truncated | reference figures kept |
  |---|---|---|
  | 2048 | 91% | 63% |
  | 3072 | 67% | 82% |
  | 4096 | 36% | 94% |
  | 6144 | 5% | 99% |

  4096 is the default in [`configs/eval.yaml`](configs/eval.yaml) until
  QLoRA memory has been measured on the GPU.
- **The references are not purely extractive.** About 13% of the numbers in
  the reference summaries do not appear anywhere in the transcript (test:
  342 / 2618). The journalists also used press releases and consensus
  estimates. Two consequences:
  - Human-level numeric "hallucination" under this metric is about 13%, not 0%.
  - SFT on the raw references may teach the model to state figures that are
    not in its input. This is one reason to filter data or use DPO.

## Metrics

Implemented in [`src/finpost/eval/`](src/finpost/eval):

- **ROUGE-1/2/L/Lsum.** F-measure with stemming. Lsum uses one bullet per line.
- **Numeric P/R/F1.** Set overlap between the summary's numbers and the
  reference's numbers.
- **Numeric hallucination rate.** The share of the summary's numbers that
  cannot be found in the transcript (the complement of ECTSum's Num-Prec).
  Reported both averaged per summary and micro-averaged over all numbers.

Every corpus-level score comes with a 95% bootstrap CI.

Known limitations of the numeric metrics:

- They check whether a number is present, not whether it is attributed
  correctly. "$53.6M, up 16.6% from $64.3M" passes if all three numbers occur
  in the transcript, even though the relation is wrong.
- Units and scale words are ignored.
- Written-out numbers ("two") are not counted.

An LLM judge is planned to cover the attribution gap.

## Results

No baseline yet.

**Pipeline smoke test (not a result to cite).** Qwen2.5-0.5B-Instruct on the
first 20 test transcripts, Mac M4 Max (MPS, fp32), greedy decoding. See
[`results/eval/Qwen2.5-0.5B-Instruct_test_n20_in4096/`](results/eval/Qwen2.5-0.5B-Instruct_test_n20_in4096).

- ROUGE-Lsum 0.121 [0.091, 0.152]
- Numeric F1 0.170 [0.108, 0.232]
- Numeric hallucination (micro) 0.060 over 265 numbers

Three runs gave identical metrics. The low hallucination rate comes from the
small model copying many figures verbatim (13.3 distinct numbers per summary,
versus 4.9 in the references for these 20 transcripts), not from it being
accurate.

## Setup

Mac (data, eval, small-model smoke tests):

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

GPU machine (training and serving). RTX 50xx cards (sm_120) need a PyTorch
build for CUDA 12.8 or newer, so install torch first from the matching index
on pytorch.org, then install the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[train,gpu,dev]'
```

vLLM runs on Linux only. On Windows, use WSL2.

## Usage

```bash
python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-0.5B-Instruct   # data + report
python scripts/env_check.py --tag 5070ti                                # GPU readiness + QLoRA memory sweep
python scripts/run_baseline_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 20      # Mac smoke test
python scripts/run_baseline_eval.py --model Qwen/Qwen2.5-3B-Instruct --backend vllm    # full test split on GPU
pytest
```

## Layout

```
configs/eval.yaml           generation settings shared by every eval
scripts/                    prepare_data, env_check, run_baseline_eval
src/finpost/data/           ECTSum download, cleaning, SQLite store
src/finpost/eval/           ROUGE, numeric metrics, aggregation
src/finpost/prompts.py      prompt template (shared by eval/SFT/DPO), truncation, output cleanup
src/finpost/dpo/            preference-pair construction (reward/select_pair: TODO)
results/                    committed outputs: data report, env checks, eval runs
```
