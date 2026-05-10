# GitHub Copilot Instructions — Instrumented ML-PPA Optimization

This file gives Copilot persistent context about the ML-PPA project. Copilot Chat reads this automatically when working in this repository.

---

## Project context

This repository contains an instrumented framework for ML-driven Power, Performance, and Area (PPA) optimization in ASIC chip design. It is independent research by Oguzhan Tekin, submitted to MLCAD Workshop 2026. The full experimental arc — from surrogate training through Bayesian optimization to EDA verification — cost $8.35 in cloud compute.

Key facts to remember:
- **Author:** Oguzhan Tekin, Independent Researcher, Toronto
- **Total compute spend:** $8.35 across the entire project
- **Designs used:** AES, Ibex, JPEG encoder, RISC-V 32I (4 ASAP7 RTL designs)
- **EDA tool:** OpenROAD Flow Scripts (ORFS) — versions v3.0 (training) and 26Q1 (verification)
- **PDK:** ASAP7 7nm predictive process design kit
- **Submission target:** MLCAD Workshop 2026

---

## Methodology principles (the project's spine)

Every coding decision in this repo follows these principles. Apply them to suggestions you make.

### 1. Pre-registration before execution

Before any experiment runs, success criteria must be committed to a plan document. Outcomes are pass/fail against pre-specified thresholds. Don't change thresholds after seeing results — re-register architecture only, with the same threshold.

### 2. Sanity checks before scaling

Every experimental phase begins with a pre-flight diagnostic that must pass before scaling. Four sanity checks have caught four real bugs in this project:
- Random-graph predictor (RL entropy ablation caught it)
- V1/V2 data confusion (provenance spot-check caught it)
- ORFS version mismatch (calibration probe caught it)
- SDC time-unit mismatch (implausible-value check caught it)

When suggesting code, prefer designs that include verification steps before consuming compute.

### 3. Negative results report with the same rigor as positive

The criticality null and ORFS version sensitivity are first-class findings, not buried limitations. When discussing trade-offs in code or documentation, don't soften unfavorable findings.

### 4. Honest scoping of claims

The C+ probe is a *cross-version generalization* claim, not a verification claim, because the v3.0 build is unrecoverable. Don't conflate these terms in suggestions.

### 5. Stopping rules

When investigating something:
- 3/3 rejections → stop
- 1/3 → continue
- 2/3 → judgment call

Apply this when running iterative analyses, not just full sessions.

---

## Architecture: surrogate role separation

The project uses two surrogate models with different roles. Don't suggest using one for the other's job.

| Model | Role | Why |
|---|---|---|
| **GBDT** (gradient-boosted trees, XGBoost-based) | Reinforcement learning reward model | Fast (~0.1ms inference), tabular input matches RL's clock-target action space |
| **GAT** (graph attention network, PyTorch Geometric) | Bayesian optimization surrogate | Captures netlist topology, MC Dropout provides uncertainty estimates |

**Bug to remember:** Using the GAT for RL produced IQM = -17.6 (random-graph inputs). Replacing with GBDT produced IQM = +67.5. This is Finding 1 of the paper.

---

## Coding conventions

### Python environment

- Python 3.11 or 3.12 (NOT 3.14 — Manim and PyTorch Geometric lag)
- Conda environment named `ppa-framework-real`
- Key dependencies: PyTorch, PyTorch Geometric, BoTorch, Stable-Baselines3, XGBoost, gymnasium

### Naming patterns

- Phase scripts: `run_<phase>_<version>.py` (e.g., `run_bo_v2a.py`, `run_rl.py`)
- Manifest files: `manifest_real_<version>.csv` (e.g., `manifest_real_v2a.csv`)
- S3 bucket: `s3://ppa-framework-30ee10a0/ppa/`
- Trained models in `models/real/`, results in `results/<phase>/`

### Acquisition function

Use `qLogNEHVI` from BoTorch (numerically stable variant of qNEHVI). Cite Ament et al. 2023 NeurIPS for qLogNEHVI; cite Daulton et al. 2021 NeurIPS for the underlying qNEHVI concept. Don't use plain `qEHVI` — it was the older approach replaced in Phase 0.

### MC Dropout

50 forward passes per candidate. The U/S ratio of 6-9× per objective indicates *over*-estimated uncertainty, which is the favorable failure mode (drives conservative BO exploration). Don't suggest "fixing" this — it makes the Pareto-expansion claims more robust.

### Cross-validation

Leave-One-Design-Out (LODO-CV) only. With just 4 design families, random splits would leak design identity. Don't suggest k-fold or stratified splits.

---

## Things to NOT do in this codebase

- **Don't add scope.** The project's experimental arc is closed. No Phase 3 multi-knob, no transfer learning, no new designs unless explicitly requested.
- **Don't try to recover ORFS v3.0.** The build is unrecoverable on Ubuntu 22.04+ due to absl dependency rot. This was investigated and closed.
- **Don't conflate the C+ result with verification.** It's cross-version generalization. Use the right term.
- **Don't oversell CAND-6.** It strictly dominates one specific control (CTRL-2), not "the baseline" generically.
- **Don't suggest GPUs.** The project's accessibility claim depends on it being CPU-only.
- **Don't change pre-registered thresholds retroactively.** Re-specifying architecture is fine; moving goalposts isn't.

---

## Common request patterns

When asked to:

**"Reproduce a result"** → Point to PROGRESS.md and the S3 artifacts, not just the code. The experimental trail is part of the deliverable.

**"Improve the model"** → First check whether the improvement aligns with the current submission. The paper is final; new experiments belong in a follow-up project.

**"Add a new design"** → Verify it can be synthesized through ORFS 26Q1 successfully (sanity check first). New designs need full LODO-CV before claims can include them.

**"Optimize the BO loop"** → Numerical stability matters more than aggressive exploration. The conservative U/S ratio is favorable, not a bug to fix.

**"Add another surrogate"** → Justify why role-separation requires it. The current GBDT/GAT split is load-bearing for the project's main finding.

---

## Reference: key numbers in the paper

These should never be misquoted in code, comments, or docs:

| Metric | Value |
|---|---|
| Total project cost | $8.35 |
| Configurations trained on | 23 |
| Designs | 4 (AES, Ibex, JPEG, RISC-V 32I) |
| BO trials per family | 50 |
| GAT MAE (LODO-CV mean) | 0.153 |
| Size+clock LR baseline MAE | 0.376 |
| GAT advantage | 2.5× over LR |
| Criticality null ΔMAE | -0.003 (within ±0.01 indifference zone) |
| Phase 1 IQM (broken) | -17.6 |
| Phase 1 IQM (fixed) | +67.5 |
| Phase 1 improvement | +85.1 |
| Pareto candidates: AES | 4 (was 5, grew to 9) |
| Pareto candidates: JPEG | 3 (was 6, grew to 9) |
| C+ generalization | 7/7 confirmed |
| CAND-6 power | 0.0683 W |
| CAND-6 area | 6386 µm² |
| CAND-6 fmax | 1262.75 MHz |
| JPEG WNS sign flip | +13.7 ps (v3.0) → -12.8 ps (26Q1) |
| MC Dropout U/S ratio | 6-9× (over-estimated, favorable mode) |
| RL training steps | 500K (PPO, 5 seeds) |

---

## Tone preferences

- **Concise.** No filler explanations.
- **Honest.** Don't soften limitations.
- **Specific.** Numbers > adjectives.
- **No hedging without reason.** Direct claims are fine when supported.
- **No emoji in code or comments.** Tables and structured text are fine.

When in doubt about a tradeoff, default to the option that aligns with the project's methodology principles above.
