# CodeDrift Arena — Hugging Face Space

This folder holds the **Gradio** UI (`space_app.py`). The repository root contains **`app.py`**, which re-exports `demo` so **Hugging Face Spaces** can find it with default settings.

## Create the Space

1. Push this repository to Hugging Face (`git remote add` → `git push`).
2. **New Space** → Gradio → link the repository.
3. Keep **App file** as `app.py` (root) and **requirements** as root `requirements.txt` (CPU-only: `gradio`).

## What judges see

- **New episode**: samples drift + PR diff + codebase snapshot.
- **Score review**: paste any text (e.g. your model’s review); the deterministic `RewardScorer` returns reward and breakdown.

No GPU is required for this demo.

## Full training stack

On Colab or a GPU machine:

```bash
pip install -r requirements-train.txt
python training/train.py --help
```

## OpenEnv HTTP server (Meta OpenEnv track)

```bash
pip install -r requirements-server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
