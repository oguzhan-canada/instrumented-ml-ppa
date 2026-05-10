---
description: Generate a pre-flight sanity check before scaling an experiment
mode: ask
---

You are about to scale an experiment. Before consuming compute, write a pre-flight sanity check that would catch failure modes early.

The sanity check should:

1. **Be cheap.** Use a small subset (1-3 samples), not the full dataset.
2. **Test a specific failure mode.** Not "does this run?" but "would this catch the kind of bug we worry about?"
3. **Have a pass/fail threshold.** No subjective interpretation.
4. **Run before any expensive operation.**

Reference the four real bug catches in this project for failure-mode patterns:

- **RL entropy ablation** caught broken predictor (random-graph inputs producing noise-dominated rewards)
- **Data provenance spot-check** caught v1/v2 confusion (byte-identical DEFs, only 4 unique designs)
- **EDA calibration probe** caught ORFS version mismatch (>10% PPA divergence on identical inputs)
- **Implausible-value check** caught SDC time-unit mismatch (22× inflated power)

Pattern: each bug would have produced a *plausible-looking but wrong* result if not caught. Your sanity check should target the same class of failure for the experiment at hand.

Output:

1. The failure mode you're targeting (one sentence)
2. The check (executable code or pseudocode)
3. The pass/fail threshold (specific numerical or boolean condition)
4. What action to take if it fails (don't just print — stop the pipeline)

Pre-register the check by saving it to `plan.md` with the threshold before running it.
