---
description: Review code or a claim against the project's methodology principles
mode: ask
---

Review the provided code or claim against the project's methodology. Flag specifically:

## Claim integrity
- Is this a verification claim or a generalization claim? Use the right term.
- Is the scope honest? (e.g., "best chip" is wrong; "best configuration BO proposed" is correct)
- Does this oversell any single result? (CAND-6 dominates *one specific control*, not generally)

## Methodology compliance
- Are pre-registered thresholds preserved, or being moved retroactively?
- Are sanity checks present before scaling steps?
- Are negative results reported with the same prominence as positive?
- Is role separation maintained? (GBDT for RL feedback; GAT for BO surrogate)

## Numerical accuracy
Cross-reference against the canonical numbers:

| Metric | Value |
|---|---|
| Total cost | $8.35 |
| Configs trained on | 23 |
| GAT MAE | 0.153 |
| LR baseline MAE | 0.376 |
| Phase 1 IQM (broken / fixed) | -17.6 / +67.5 |
| C+ result | 7/7 |
| JPEG WNS sign flip | +13.7 → -12.8 ps |
| MC Dropout U/S ratio | 6-9× (over-estimated, favorable) |

If any number in the input doesn't match these, flag it.

## Scope discipline
- Does this introduce a new experimental phase? (Phase 3 multi-knob, transfer learning, new designs are out of scope)
- Does this attempt to recover ORFS v3.0? (closed; build is unrecoverable)
- Does this contradict the role-separation finding?

## Output format

Three sections:

1. **Issues found** (numbered list, ordered by severity — bugs first, scope creep last)
2. **Aligned with project methodology** (numbered list of what's done correctly)
3. **Recommended fixes** (specific, concrete, actionable — not "consider revising")

If nothing is wrong, say so directly. Don't manufacture issues to fill the format.
