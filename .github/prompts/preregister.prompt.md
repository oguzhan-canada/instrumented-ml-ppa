---
description: Pre-register an experiment with success criteria before execution
mode: ask
---

You are about to run an experiment. Pre-register the outcome criteria before executing, following the methodology used throughout this project.

Output a plan.md entry in this exact format:

```markdown
## Experiment: <name>

**Date pre-registered:** <today>
**Hypothesis:** <one sentence — what you expect to happen>

**Pre-registered outcomes:**

| Outcome | Threshold | Interpretation |
|---|---|---|
| <Outcome A> | <specific numerical/boolean threshold> | <what it means if this happens> |
| <Outcome B> | <threshold> | <interpretation> |
| <Outcome C> | <threshold> | <interpretation> |

**Stopping rule:** <when to abort early — typically 3/3 negative early signals>

**What I will NOT change after seeing results:**
- The thresholds above
- The interpretation of each outcome

**What I MAY change after diagnosis:**
- The architecture / method (with explicit re-registration)
- The implementation (bug fixes don't require re-registration)
```

Critical rules:

1. **Thresholds must be specific.** "Improvement" is not a threshold; "MAE < 0.2" is.
2. **Each outcome must be falsifiable.** If every result you can imagine satisfies the criterion, it's not a threshold.
3. **Cover the failure modes.** Don't only define what "success" looks like — also define what "definitively wrong" looks like.
4. **Don't move goalposts.** If the result misses the threshold, that's a NULL or FAIL — not a reason to relax the threshold.

Reference past pre-registrations from this project:

- Phase 1 (RL): "IQM > 0 after 500K steps" → FAIL with monolithic GAT, threshold unchanged, architecture re-specified to GBDT → PASS
- Phase 2a (GAT): "LODO-CV MAE < 0.2" → PASS at 0.153
- Phase 2a (Criticality ablation): "|ΔMAE| > 0.01 → criticality helps; else redundant" → NULL at -0.003
- Phase 2a (BO): "Pareto front grows by ≥1 candidate" → PASS (+4 AES, +3 JPEG)
- C+ probe: "≥4/7 Pareto-dominant → generalizes" → PASS (7/7)

Save the pre-registration to `plan.md` and commit it before running the experiment.
