# Instrumented ML-Driven PPA Optimization

**Pre-Registered Experiments, Negative Results, and Cross-Version EDA Generalization**

Oguzhan Tekin · Independent Researcher · Toronto, Canada

📄 [Paper (PDF)](paper/paper.pdf) · 📊 [Experimental Trail (PROGRESS.md)](PROGRESS.md)

---

## Overview

An end-to-end ML framework for Power, Performance, and Area (PPA) optimization of ASIC designs using OpenROAD Flow Scripts (ORFS) and the ASAP7 PDK. The project is distinguished by its methodological instrumentation: pre-registered experimental outcomes, systematic sanity checks, and honest reporting of negative results.

**Key results:**
- **6 pre-registered findings** across positive results, a mechanistically-explained null, and a reproducibility hazard
- **7/7 BO candidates** confirmed on the Pareto front via cross-version EDA generalization (C+ probe)
- **$8.35 total compute cost** on AWS commodity instances — no GPUs, no institutional infrastructure

## Repository Structure

```
├── paper/                  # MLCAD workshop paper
│   ├── paper.tex           # LaTeX source (IEEE conference format)
│   ├── paper.pdf           # Compiled PDF (6 pages)
│   └── figures/            # Figure generation scripts and outputs
│       ├── figures.py      # Generates all 4 publication figures
│       ├── fig1_pareto.*   # Pareto fronts: AES (4/4) + JPEG (3/3)
│       ├── fig2_tradeoff.* # Trade-off direction consistency
│       ├── fig3_version.*  # ORFS version sensitivity (v3.0 vs 26Q1)
│       └── fig4_cost.*     # Per-phase cost breakdown
│
├── framework/              # ML pipeline code
│   ├── configs/            # YAML configuration files
│   ├── data/               # Data manifests and data card
│   ├── eval/               # Model evaluation scripts
│   ├── features/           # Graph extraction and feature engineering
│   │   ├── dataset.py      # PyTorch Geometric dataset construction
│   │   └── criticality.py  # Depth-based criticality edge weights
│   ├── models/             # Trained model checkpoints
│   │   ├── gat_ppa.py      # GAT surrogate architecture
│   │   └── gbdt_ppa.py     # GBDT reward model
│   ├── optimize/           # Optimization engines
│   │   ├── bo_engine.py    # qLogNEHVI Bayesian optimization
│   │   ├── rl_env.py       # PPO RL environment
│   │   └── run_bo.py       # BO execution script
│   ├── scripts/            # Data processing and training scripts
│   ├── results/            # Experimental outputs
│   └── tests/              # Unit tests
│
├── aws/                    # Infrastructure-as-Code
│   ├── terraform/          # EC2 spot instance provisioning
│   │   ├── main.tf         # Resource definitions
│   │   ├── variables.tf    # Configuration variables
│   │   └── terraform.tfvars.example  # Template (no secrets)
│   ├── docker/             # Container definitions
│   └── scripts/            # Bootstrap and sync scripts
│
├── PROGRESS.md             # Complete experimental trail
└── .gitignore
```

## Experimental Phases

| Phase | Description | Key Finding | Status |
|-------|-------------|-------------|--------|
| **Phase 0** | Bootstrap, qLogNEHVI migration | Framework operational | ✅ |
| **Phase 1** | RL training (PPO, 500K steps, 5 seeds) | Role separation: GBDT for RL, GAT for BO (+85.1 IQM) | ✅ |
| **Phase 2a** | GAT training, criticality ablation | Criticality null (ΔMAE = −0.003); LODO-CV MAE = 0.153 | ✅ |
| **Phase 2a** | Bayesian optimization (50 qLogNEHVI trials) | +4 AES, +3 JPEG Pareto candidates | ✅ |
| **Phase 2a** | EDA calibration probe | ORFS v3.0→26Q1 divergence >10%; WNS sign flip | ✅ |
| **Phase 2a-C+** | Cross-version generalization | 7/7 candidates on Pareto front; CAND-6 strict domination | ✅ |

## Six Findings

1. **Role Separation** — Replacing GAT with GBDT as RL reward model: IQM −17.6 → +67.5
2. **GAT Surrogate Quality** — LODO-CV MAE = 0.153, outperforming linear regression 2.5×
3. **Criticality Null** — Depth-based criticality edges redundant (ΔMAE = −0.003) due to perfect node-count/clock correlation
4. **Surrogate-Only Pareto Expansion** — BO found +4 AES, +3 JPEG Pareto candidates
5. **ORFS Version Sensitivity** — v3.0→26Q1 produces >10% PPA divergence; JPEG WNS flips sign
6. **Cross-Version Generalization** — 7/7 BO candidates confirmed on Pareto front against 26Q1 baselines

## Sanity Checks

Four pre-flight checks caught four distinct bugs before they could propagate:

1. **RL entropy ablation** → broken GAT predictor (random-graph inputs)
2. **Data provenance spot-check** → v1/v2 data confusion (byte-identical DEFs)
3. **EDA calibration probe** → ORFS version mismatch (v3.0 vs 26Q1)
4. **C+ implausible-value check** → SDC time-unit mismatch (ps vs ns)

## Artifacts

All experimental artifacts are organized by phase:

- **Phase 0:** Bootstrap scripts, qLogNEHVI migration code
- **Phase 1:** RL training checkpoints (5 seeds), entropy ablation logs, IQM curves, reward landscape plots
- **Phase 2a:** GAT weights (with/without criticality, 3 seeds), LODO-CV splits, BO trial histories, criticality ablation results
- **C+ Probe:** Full ORFS timing/power/area reports for all 10 candidate and control configurations

Pre-registered experimental plans with success/failure thresholds are committed before each phase's execution.

## Reproducibility

```bash
# Clone and setup
git clone https://github.com/oguzhan-canada/instrumented-ml-ppa.git
cd instrumented-ml-ppa/framework
pip install -r requirements.txt

# Run tests
pytest tests/

# Generate paper figures
cd ../paper/figures
python figures.py
```

The complete experimental trail is documented in [PROGRESS.md](PROGRESS.md), recording every decision, bug fix, and sanity-check result in chronological order.

## Compute Cost

| Phase | Instance | Duration | Cost |
|-------|----------|----------|------|
| Bootstrap + migration | t3.medium spot | ~2h | $0.85 |
| RL training (5 seeds) | t3.medium spot | ~3h | $1.20 |
| GAT + BO + ablation | c6i.xlarge spot | ~6h | $2.95 |
| EDA calibration | c6i.4xlarge on-demand | ~1h | $0.65 |
| C+ generalization | c6i.4xlarge on-demand | ~4h | $2.70 |
| **Total** | | | **$8.35** |

No GPUs were used. All computation ran on commodity AWS instances.

## Citation

```bibtex
@inproceedings{tekin2026instrumented,
  title={Instrumented {ML}-Driven {PPA} Optimization: Pre-Registered Experiments,
         Negative Results, and Cross-Version {EDA} Generalization},
  author={Tekin, Oguzhan},
  booktitle={Proc. ACM/IEEE Workshop on Machine Learning for CAD (MLCAD)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
