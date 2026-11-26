<div align="center">

# CFR-Pro

### Proximity Matters: Local Proximity Enhanced Balancing for Treatment Effect Estimation

**Hao Wang · Zhichao Chen · Zhaoran Liu · Xu Chen · Haoxuan Li · Zhouchen Lin**

**ACM SIGKDD 2025** · Proceedings, Volume 2 · pp. 2927–2937

**Official source code for Proximity-enhanced CounterFactual Regression.**

[Overview](#overview) · [Repository](#repository-structure) · [Quick start](#installation-and-quick-start) · [Reproduce Table 1](#reproducing-table-1) · [Citation](#citation)

</div>

---

> **TL;DR:** Global distribution alignment alone overlooks local structure. CFR-Pro brings local proximity into representation balancing through an optimal-transport-based pair-wise regularizer and an informative subspace projector for heterogeneous treatment effect estimation.

## Overview

Estimating heterogeneous treatment effects (HTEs) from observational data is difficult because treatment assignment is not random. Representation-balancing methods typically mitigate treatment selection bias by aligning the global distributions of treated and control units. Yet global alignment does not fully exploit a useful local principle: **similar units tend to have similar outcomes**.

CFR-Pro makes this local structure part of the balancing objective:

- **Proximity-enhanced balancing.** An optimal-transport-based pair-wise proximity regularizer incorporates local relationships into discrepancy calculation, rather than relying only on global alignment.
- **Informative subspace projection.** A projector trades off minimal distance precision for improved sample complexity, addressing the difficulty of estimating proximity and discrepancies in high-dimensional, limited-data settings.
- **Treatment effect estimation.** The resulting representations support more accurate matching across treatment groups and mitigate treatment selection bias. The paper reports extensive experiments demonstrating improved estimation performance over competing methods.

In this release, CFR-Pro is implemented in **[`src/estimator_gromov.py`](src/estimator_gromov.py)**. The `fgwpca` option combines PCA-based subspace projection with fused Gromov–Wasserstein balancing.

<details>
<summary><strong>Paper abstract</strong></summary>

Heterogeneous treatment effect (HTE) estimation from observational data poses significant challenges due to treatment selection bias. Existing methods address this bias by minimizing distribution discrepancies between treatment groups in latent space, focusing on global alignment. However, the fruitful aspect of local proximity, where similar units exhibit similar outcomes, is often overlooked. In this study, we propose Proximity-enhanced CounterFactual Regression (CFR-Pro) to exploit proximity for enhancing representation balancing within the HTE estimation context. Specifically, we introduce a pair-wise proximity regularizer based on optimal transport to incorporate the local proximity in discrepancy calculation. However, the curse of dimensionality renders the proximity measure and discrepancy estimation ineffective—exacerbated by limited data availability for HTE estimation. To handle this problem, we further develop an informative subspace projector, which trades off minimal distance precision for improved sample complexity. Extensive experiments demonstrate that CFR-Pro accurately matches units across different treatment groups, effectively mitigates treatment selection bias, and significantly outperforms competitors.

</details>

## Repository structure

```text
CFR-Pro/
├── src/                             # CFR-Pro, baselines, and shared utilities
│   ├── estimator_gromov.py          # CFR-Pro (fgwpca) and transport variants
│   ├── estimator_cfr.py             # CFR-MMD and CFR-Wass
│   ├── estimator_escfr.py           # ESCFR
│   ├── estimator_descn.py           # DESCN
│   ├── estimator_site.py            # SITE
│   ├── estimator_metalearner.py     # Neural S-, T-, and Y-learners
│   ├── estimator_ml.py              # Classical and forest baselines
│   ├── base_estimator.py            # Shared neural training and evaluation
│   ├── models.py                   # Neural model definitions
│   ├── data_processor.py           # Benchmark preprocessing
│   └── ...                         # Metrics, optimization, and other helpers
├── scripts/                         # Experiment recipes and hyperparameters
│   └── exp_overall.sh               # Overall-performance / Table 1 experiments
├── assets_sigkdd/                    # Saved results and experimental analyses
│   ├── baseline_final/              # ACIC and IHDP results across ten seeds
│   ├── OverallPerformance.ipynb     # Aggregate benchmark performance
│   ├── HyperparameterSensitivity.ipynb
│   └── TransportVisualization.ipynb
└── Datasets/                        # Raw and processed benchmark datasets
    ├── ACIC/
    └── IHDP/
```

| Directory | Purpose |
| :--- | :--- |
| **`src/`** | Contains the implementation of CFR-Pro and all baselines included in this release, together with shared models, preprocessing, and evaluation utilities. Estimator entry points can be run directly and tuned independently; helper modules support those entry points. **Our method is `estimator_gromov.py`.** |
| **`scripts/`** | Contains the scripts for running the methods in `src/`, with concrete dataset-specific hyperparameters and random seeds for the experiments underlying **Table 1**. |
| **`assets_sigkdd/`** | Contains saved experiment results and analysis notebooks for overall performance, hyperparameter sensitivity, and transport visualization. |
| **`Datasets/`** | Contains the ACIC and IHDP benchmark data, including raw inputs and the processed splits consumed by the estimators. |

## Installation and quick start

### 1. Set up the repository and dependencies

```bash
git clone https://github.com/haowangzju/CFR-Pro.git
cd CFR-Pro

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch tensorboard numpy pandas scipy scikit-learn POT PyYAML tqdm matplotlib seaborn econml jupyterlab
```

The package list is derived from the released code's imports; this release does not include a pinned environment or lockfile. Install a PyTorch build appropriate for your hardware if using CUDA. `POT` provides the Python module `ot`. Some optional transport-visualization cells also require `fugw`.

**Run estimator commands from the repository root.** Dataset paths are relative to the current working directory, and `Datasets/` is case-sensitive on many systems. The processed benchmark data are already included; neural training does not require regenerating them first.

### 2. Run CFR-Pro on ACIC

```bash
python src/estimator_gromov.py \
  --data ACIC \
  --seed 1 \
  --model fgwpca \
  --ot fgwpca \
  --root reproduction_results
```

### 3. Run CFR-Pro on IHDP

The following settings match the CFR-Pro configuration in `scripts/exp_overall.sh`:

```bash
python src/estimator_gromov.py \
  --data IHDP \
  --seed 1 \
  --model fgwpca \
  --ot fgwpca \
  --dim_backbone 16,16 \
  --dim_task 32,32 \
  --fgw_alpha 0.2 \
  --lambda 2 \
  --ot_scale 1.0 \
  --pca_ratio 0.5 \
  --root reproduction_results
```

`--ot fgwpca` selects the CFR-Pro balancing implementation. `--model fgwpca` supplies the experiment label used in result directories; it should not be confused with the transport selector.

For these examples, outputs are written to `reproduction_results/ACIC/fgwpca_1/` and `reproduction_results/IHDP/fgwpca_1/`. Neural runs save:

- `run.txt`: training, validation, and test metrics;
- `config.yaml`: the run's hyperparameters;
- `model.pt`: the selected model checkpoint;
- `events.out.tfevents.*`: TensorBoard logs.

**If `model.pt` already exists at the destination, the neural entry point skips training.** Use a fresh `--root` for a new experiment. Checkpoints serialize whole PyTorch model objects: load only trusted checkpoints and account for PyTorch-version compatibility.

## Running and tuning individual methods

Each estimator entry point can be executed independently of the overall experiment script. Neural entry points expose command-line hyperparameters; classical estimators can be tuned through their constructors in `estimator_ml.py`.

| Method | Entry point | Selector |
| :--- | :--- | :--- |
| **CFR-Pro** | `src/estimator_gromov.py` | `--model fgwpca --ot fgwpca` |
| CFR-MMD / CFR-Wass | `src/estimator_cfr.py` | `--model cfr-mmd` / `--model cfr-wass` |
| ESCFR | `src/estimator_escfr.py` | Dataset-specific `--gamma` |
| DESCN | `src/estimator_descn.py` | Dataset-specific loss weights |
| SITE | `src/estimator_site.py` | Dataset-specific dimensions and penalties |
| Neural S- / T- / Y-learner | `src/estimator_metalearner.py` | `--model slearner` / `tlearner` / `ylearner` |
| OLS-S, OLS-T, KNN, RF-S, PSM, OForest, CForest | `src/estimator_ml.py` | Runs the classical baseline suite for `--data` |

For example:

```bash
python src/estimator_cfr.py --data ACIC --seed 1 --model cfr-mmd --root reproduction_results
python src/estimator_metalearner.py --data IHDP --seed 1 --model tlearner --root reproduction_results
python src/estimator_gromov.py --help
```

### Key CFR-Pro hyperparameters

| Argument | Meaning | ACIC recipe | IHDP recipe |
| :--- | :--- | :---: | :---: |
| `--fgw_alpha` | Trade-off between feature matching and pair-wise proximity in fused GW | `0.2` | `0.2` |
| `--lambda` | Weight of the representation-balancing loss | `0.5` | `2.0` |
| `--pca_ratio` | Retained subspace dimension as a fraction of representation width | `0.2` | `0.5` |
| `--ot_scale` | Scale of the transport cost matrices | `0.1` | `1.0` |
| `--dim_backbone` | Representation-network hidden dimensions | `16,16` | `16,16` |
| `--dim_task` | Outcome-network hidden dimensions | `32,32` | `32,32` |
| `--batchSize` | Training mini-batch size | `32` | `32` |
| `--lr` | Adam learning rate | `0.001` | `0.001` |
| `--l2_reg` | Weight decay | `0.0001` | `0.0001` |

Use `--tune 1` when sweeping CFR-Pro's balancing parameters: this enables a parameter-encoded output path instead of only a method/seed label. For other sweeps, assign distinct output roots to avoid checkpoint collisions. The projected dimension is `int(pca_ratio * representation_width)`; keep it positive and within the dimensions supported by the mini-batch.

## Reproducing Table 1

### Experiment recipe

[`scripts/exp_overall.sh`](scripts/exp_overall.sh) provides the concrete settings for the overall-performance experiments in **Table 1**, covering ACIC and IHDP with seeds **1–10**:

```bash
# Run from the repository root.
bash scripts/exp_overall.sh
```

The default output root is `assets_sigkdd/baseline_final`, where the supplied results are already stored. To train anew without replacing the released artifacts, set `ROOT` in the script to a fresh directory such as `assets_sigkdd/reproduction_table1`, then point the performance notebook to that directory.

**Execution notes for the released script:**

- It sets `MAX_JOBS=16` and explicitly uses `CUDA_VISIBLE_DEVICES=-1` for neural jobs, so those jobs run on **CPU**. Adjust concurrency to your machine. For individual CUDA runs, select a visible GPU with `CUDA_VISIBLE_DEVICES`; the current estimators choose CUDA based on availability, rather than the parsed `--device` argument.
- Neural jobs are launched in the background. The classical baseline commands appear before the final `wait`; if re-running the full suite, wait for neural jobs to finish before launching classical preprocessing to avoid concurrent reads and writes of the dataset splits.
- `estimator_ml.py` regenerates processed data while looping over seeds. Run it only in a writable working copy if you need to preserve the supplied splits. Neural ACIC runs use the unsuffixed splits; neural IHDP runs use seed-specific replications. Preserve this convention when comparing results.
- A fresh output root avoids skipped neural runs and replacement of saved result files. Hardware, dependency versions, and numerical randomness may affect the reproduced values.

### Aggregate the supplied results

The provided results can be analyzed without retraining:

```bash
cd assets_sigkdd
jupyter lab OverallPerformance.ipynb
```

Run the notebook from `assets_sigkdd/`, so its relative `./baseline_final/ACIC` and `./baseline_final/IHDP` paths resolve correctly. The active cells aggregate **test-set mean ± standard deviation** over seeds. They explicitly filter out `CForest`; in-sample aggregation cells are commented out. The saved method label **`fgwpca` corresponds to CFR-Pro**.

The results include treatment-effect and outcome-prediction metrics, including PEHE, ATE/ATT errors, factual/counterfactual prediction metrics, and uplift metrics. Consult the paper for the exact Table 1 metric selection and reporting conventions.

## Additional experimental analyses

| Notebook | Purpose | Inputs |
| :--- | :--- | :--- |
| [`OverallPerformance.ipynb`](assets_sigkdd/OverallPerformance.ipynb) | Aggregate ACIC and IHDP overall performance | Included `baseline_final/` results |
| [`HyperparameterSensitivity.ipynb`](assets_sigkdd/HyperparameterSensitivity.ipynb) | Analyze sensitivity to balancing hyperparameters | Expects `exp_hyperparam/` sweep results, **not included in this release** |
| [`TransportVisualization.ipynb`](assets_sigkdd/TransportVisualization.ipynb) | Visualize transport behavior on illustrative data | Notebook-generated toy data; optional cells require additional packages such as `fugw` |

## Citation

If you use CFR-Pro or this codebase in your research, please cite:

```bibtex
@inproceedings{wang2025proximity,
  title={Proximity matters: Local proximity enhanced balancing for treatment effect estimation},
  author={Wang, Hao and Chen, Zhichao and Liu, Zhaoran and Chen, Xu and Li, Haoxuan and Lin, Zhouchen},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2927--2937},
  year={2025}
}
```

## Acknowledgements

We thank the authors of the baseline methods and the maintainers of the scientific Python ecosystem. This implementation uses PyTorch, Python Optimal Transport (POT), scikit-learn, and EconML. The uplift-evaluation utilities in `src/auuc.py` acknowledge code adapted from causalML. Please credit the original benchmark providers and respect the applicable terms for third-party code and data.

For questions about the implementation or reproduction protocol, please [open an issue](https://github.com/haowangzju/CFR-Pro/issues).
