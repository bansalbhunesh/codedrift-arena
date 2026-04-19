# CodeDrift Arena

RL environment for **PR review under adversarial codebase drift** (rename / file removal / API contract), with a deterministic reward for GRPO-style training.

## Install (pick one)

| Goal | Command |
|------|-----------|
| **Hugging Face Space** (CPU demo) | Default: `pip install -r requirements.txt` |
| **GRPO training** (GPU, Colab) | `pip install -r requirements-train.txt` |
| **OpenEnv HTTP server** | `pip install -r requirements-server.txt` |

## Quick local checks

```bash
python scripts/smoke_env.py
python demo/before_after.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Optional: `export CODEDRIFT_LOG_LEVEL=DEBUG` before running training or env-heavy scripts (see `codedrift/logutil.py`).

## Layout

- `training/train.py` — GRPO training (TRL constructor is version-detected at runtime).
- `server/app.py` — OpenEnv FastAPI app (`uvicorn server.app:app`).
- `app.py` + `hf_space/` — Gradio Space UI; see `hf_space/README.md`.
