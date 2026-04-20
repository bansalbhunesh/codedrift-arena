# Training run summary (update with latest run)

Use this page as the single source of truth for training evidence shown to judges.
If values are illustrative, keep them clearly marked; before submission, replace
with real run exports/logs.

## Current values (illustrative example)

| Checkpoint | Mean reward (batch) | Notes |
|-------------|---------------------|--------|
| Step 0 (base) | -0.85 | Approves stale PRs; misses ISSUES |
| Step 80 | +0.10 | Starts citing stale symbols under `REQUEST_CHANGES` |
| Step 200 | +0.72 | Strong recall on easy + medium episodes |

**Caption for slides:** *Same environment and scorer — the policy learns to align ISSUES with ground-truth drift.*
