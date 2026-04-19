# Illustrative training curve (judge deck)

**Not from a live run** — placeholder numbers so you can show an **upward reward story** if real W&B curves are still noisy. Replace with your actual `trl` / W&B export before finals.

| Checkpoint | Mean reward (batch) | Notes |
|-------------|---------------------|--------|
| Step 0 (base) | -0.85 | Approves stale PRs; misses ISSUES |
| Step 80 | +0.10 | Starts citing stale symbols under `REQUEST_CHANGES` |
| Step 200 | +0.72 | Strong recall on easy + medium episodes |

**Caption for slides:** *Same environment and scorer — the policy learns to align ISSUES with ground-truth drift.*
