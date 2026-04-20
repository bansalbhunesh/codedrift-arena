# Contributing

Thanks for contributing to CodeDrift Arena.

## Quick setup

```bash
git clone https://github.com/bansalbhunesh/codedrift-arena.git
cd codedrift-arena
pip install -r requirements.txt
```

## Before opening a PR

Run the local checks:

```bash
python scripts/smoke_env.py
python -m unittest discover -s tests -p "test_*.py" -v
```

If you change OpenEnv code, also verify:

```bash
pip install -r requirements-server.txt
python scripts/openenv_ws_demo.py --help
```

## Code guidelines

- Keep reward logic deterministic in `rewards/scorer.py`.
- Put UI-only behavior in `hf_space/`, not env/reward modules.
- Avoid private attribute coupling across modules; prefer public methods/properties.
- Add or update tests for any scoring or env lifecycle changes.

## Commit style

Use short, imperative commit messages, e.g.:

- `Fix scorer false-positive on contract drift`
- `Add OpenEnv websocket smoke demo`
