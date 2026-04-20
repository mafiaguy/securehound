# securehound — MLX-LM LoRA Scaffold Design

**Date:** 2026-04-21
**Status:** Approved
**Author:** Siddhanth Dwivedi (with Claude)

## Purpose

Stand up a clean, open-source-ready repository scaffold for fine-tuning `mistralai/Mistral-7B-Instruct-v0.3` with LoRA on an 18 GB Apple Silicon (M3 Pro) laptop, targeting a security-bug-finder use case (user submits code, model returns vulnerability + fix).

The scaffold must be **runnable end-to-end on the sample data that ships with it**, so that once real datasets are dropped into `data/raw/`, only the loader logic needs filling in.

## Non-Goals (v1)

- No dataset selection is made in this design — only the loader interface that future datasets will plug into.
- No GitHub App, chat UI, API server, or model hosting.
- No 4-bit quantisation (bitsandbytes does not work on Apple Silicon).
- No Unsloth / Axolotl (CUDA-first; unusable here).
- No LLM-as-judge evaluation (deferred; avoids a paid API dependency in v1).

## Fixed Decisions

| Decision | Value | Why |
|---|---|---|
| Training framework | MLX-LM | Native Apple Silicon; fastest on M-series. |
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` | Chat-tuned; `[INST]…[/INST]` template matches inference shape. |
| Adapter type | LoRA | Memory-feasible on 18 GB unified memory. |
| Scope | Full pipeline (data-prep + train + infer + eval + fuse) | User-approved. |
| Training schema | `{"messages": [{"role": "user", …}, {"role": "assistant", …}]}` | Mirrors inference; matches Mistral chat template. |

## Repository Layout

```
securehound/
├── README.md                    # Setup, usage, roadmap
├── CLAUDE.md                    # Project-local guidance (overrides Desktop CLAUDE.md)
├── LICENSE                      # Apache 2.0
├── .gitignore                   # Excludes models/, data/raw/, data/processed/, adapters/, .venv/, wandb/, __pycache__/
├── requirements.txt             # mlx, mlx-lm, datasets, pyyaml, tqdm, rich, click, pydantic
├── pyproject.toml               # Pins Python 3.11+; editable install of src/securehound
│
├── configs/
│   ├── model.yaml               # Base model ID, chat template, max_seq_len
│   ├── lora.yaml                # Rank, alpha, dropout, target modules, num_layers
│   └── train.yaml               # lr, steps, batch, grad-accum, warmup, save/val cadence
│
├── data/
│   ├── sample/                  # ~20 toy examples (committed). Covers SQLi, XSS, SSRF, path
│   │                            #   traversal, insecure deserialisation, etc. in the messages schema.
│   ├── raw/                     # Untracked. User drops CVE / GHSA / report dumps here.
│   └── processed/               # Untracked. data_prep.py writes {train,valid,test}.jsonl here.
│
├── scripts/
│   ├── data_prep.py             # --source cve|ghsa|code_pairs|all → normalised JSONL splits
│   ├── train.py                 # mlx_lm.lora wrapper driven by configs/
│   ├── infer.py                 # Load base + adapter; run on a code snippet or file
│   ├── eval.py                  # Val loss + CWE-match accuracy + ROUGE-L on fix block
│   └── fuse.py                  # mlx_lm.fuse wrapper: merge adapter into base for distribution
│
├── src/securehound/
│   ├── __init__.py
│   ├── prompts.py               # System prompt + user/assistant templates (single source of truth)
│   ├── schema.py                # Pydantic TrainingExample + validators for the messages schema
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── cve.py               # NVD / CVE JSON → List[TrainingExample]
│   │   ├── ghsa.py              # GitHub Security Advisories → List[TrainingExample]
│   │   └── code_pairs.py        # Vulnerable/patched code datasets (Big-Vul, DiverseVul, …)
│   └── eval_metrics.py          # cwe_match_accuracy(), rouge_l_on_fix()
│
├── adapters/                    # Untracked. train.py writes LoRA weights here.
└── docs/
    └── superpowers/specs/       # Design docs (this file lives here)
```

### Why `src/` + `scripts/` split

`src/securehound/` holds pure, importable library code; `scripts/` holds thin CLI entry points that import from the library. This keeps every file small and testable in isolation, and lets future additions (GitHub app, API server) import the same prompts/schema/loaders without duplication.

## Training Example Schema

Every training record conforms to:

```json
{"messages": [
  {"role": "user", "content": "Review this code for security issues:\n\n```<lang>\n<code>\n```"},
  {"role": "assistant", "content": "**Vulnerability:** <type> (CWE-<id>)\n**Severity:** <low|med|high|crit>\n**Explanation:** <what/why>\n**Fix:**\n```<lang>\n<patched_code>\n```"}
]}
```

CVE-only records without an accompanying code snippet map to a Q&A variant (`"Explain CVE-2023-XXXX"` → prose answer) using the same `messages` array, so the schema and training loop stay uniform.

Enforced in code via `src/securehound/schema.py` (Pydantic). Any loader that produces a record failing validation must raise; `data_prep.py` will fail loudly rather than silently dropping rows.

## Configs (tuned for 18 GB M3 Pro)

**`configs/model.yaml`**
```yaml
base_model: mistralai/Mistral-7B-Instruct-v0.3
chat_template: mistral_v3
max_seq_len: 1024
```

**`configs/lora.yaml`**
```yaml
rank: 8
alpha: 16
dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
num_layers: 16   # apply LoRA to the last 16 transformer layers (closest to output); lower layers stay frozen. Mistral-7B has 32 layers total.
```

**`configs/train.yaml`**
```yaml
learning_rate: 1.0e-5
batch_size: 1
grad_accum_steps: 8     # effective batch = 8
max_steps: 1000         # tune once real data arrives
warmup_steps: 50
save_every: 200
val_every: 100
grad_checkpoint: true
seed: 42
```

These defaults are conservative for 18 GB; cranking up rank, seq_len, or batch is a YAML edit, not a code change.

## Data Pipeline

```
data/raw/<source>/*.json
        │
        ▼
scripts/data_prep.py  ──dispatch──► src/securehound/loaders/<source>.py
        │                                        │
        │                                   List[TrainingExample] (validated)
        │                                        │
        ▼                                        ▼
   shuffle + 90/5/5 split ◄──────────────────────┘
        │
        ▼
data/processed/{train,valid,test}.jsonl
```

- `data_prep.py --source all` runs every loader present in `data/raw/` and concatenates the outputs.
- Split filenames (`train.jsonl`, `valid.jsonl`, `test.jsonl`) match what MLX-LM's LoRA trainer expects natively.
- Deterministic shuffle (seeded from `configs/train.yaml`) so splits are reproducible across runs.

## Scripts (CLI Surface)

All scripts use `click`:

```bash
python scripts/data_prep.py --source all --out data/processed/
python scripts/train.py     --config configs/train.yaml --data data/processed/
python scripts/infer.py     --adapter adapters/latest --code-file path/to/file.py
python scripts/eval.py      --adapter adapters/latest --test data/processed/test.jsonl
python scripts/fuse.py      --adapter adapters/latest --out models/securehound-v1/
```

`scripts/train.py` is a thin wrapper around `mlx_lm.lora` that reads the three YAMLs and forwards the right flags. Keeping it thin means MLX-LM upgrades rarely break us.

## Evaluation

Minimal but honest for v1:

1. **Validation loss** during training (MLX-LM native via `--val-batches`).
2. **CWE-match accuracy** — parse the `CWE-<id>` out of the model's first line of output and compare to ground truth. Deterministic, meaningful for a CWE-labelled task.
3. **ROUGE-L on the Fix block** — rough proxy for fix quality. Cheap; no API calls.

An `eval_judge.py` (LLM-as-judge) can be added later when a provider choice is made; leaving it out keeps v1 dependency-free.

## Sample Data

`data/sample/train.jsonl` ships with ~20 hand-written examples covering common CWE classes:

- SQL injection (CWE-89)
- Reflected XSS (CWE-79)
- SSRF (CWE-918)
- Path traversal (CWE-22)
- Insecure deserialisation (CWE-502)
- Hardcoded credentials (CWE-798)
- Weak crypto / RNG (CWE-327 / CWE-338)
- Command injection (CWE-78)
- XXE (CWE-611)
- Open redirect (CWE-601)

Purpose: smoke-test the full train → infer → eval cycle on the laptop before real data is available.

## CLAUDE.md Contents (project-local)

Must cover:

- Project purpose (security-bug-finder LLM via LoRA fine-tune).
- Hardware constraints: M3 Pro, 18 GB unified memory. MLX-only stack. No CUDA, no bitsandbytes, no 4-bit.
- "Don't suggest" list: Unsloth, Axolotl, 4-bit quant, CUDA-only libraries, changing the messages schema without updating `schema.py` + `prompts.py` + loaders together.
- Config locations and what each YAML controls.
- The training-example schema contract (with a pointer to `src/securehound/schema.py` as source of truth).
- Common commands (`data_prep`, `train`, `infer`, `eval`, `fuse`).
- Data licensing caveat: only `data/sample/` is committed; `data/raw/` and `data/processed/` stay out of git because upstream dataset licences vary.

## Out of Scope (explicit, for future specs)

- Dataset selection (CVE-Fixes vs Big-Vul vs DiverseVul vs PrimeVul vs GHSA dump vs …).
- GitHub App and chat UI.
- Hosted inference API.
- LLM-as-judge eval.
- Multi-GPU or CUDA training path.
- RLHF / DPO stages beyond SFT.

Each of these earns its own brainstorm → spec → plan cycle when its time comes.

## Open Questions (deferred, not blocking scaffold)

- Licence choice beyond Apache 2.0 if future datasets have incompatible licences.
- Whether to ship a pre-trained adapter in releases or only source + instructions (depends on dataset licence).
- Concrete target for `max_steps` (depends entirely on dataset size).

None of these block the scaffold; they're flagged so we remember to revisit.
