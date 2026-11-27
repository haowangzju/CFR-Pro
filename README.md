<div align="center">

<img src="assets_sigkdd/cfr-pro-logo.svg" alt="CFR-Pro — Proximity-enhanced CounterFactual Regression" width="550">

## Local Proximity Enhanced Balancing for Treatment Effect Estimation

**ACM SIGKDD 2025** · Proceedings, Volume 2 · pp. 2927–2937

🤗[Overview](#overview) ｜ 🤗[Repository](#repository-structure) ｜ 🤗[Quick start](#installation-and-quick-start) ｜ 🤗[Reproduce Table 1](#reproducing-table-1) ｜ 🤗[Citation](#citation)

</div>

---
## Overview

<details>
<summary><strong>Paper abstract</strong></summary>

Heterogeneous treatment effect (HTE) estimation from observational data poses significant challenges due to treatment selection bias. Existing methods address this bias by minimizing distribution discrepancies between treatment groups in latent space, focusing on global alignment. However, the fruitful aspect of local proximity, where similar units exhibit similar outcomes, is often overlooked. In this study, we propose Proximity-enhanced CounterFactual Regression (CFR-Pro) to exploit proximity for enhancing representation balancing within the HTE estimation context. Specifically, we introduce a pair-wise proximity regularizer based on optimal transport to incorporate the local proximity in discrepancy calculation. However, the curse of dimensionality renders the proximity measure and discrepancy estimation ineffective—exacerbated by limited data availability for HTE estimation. To handle this problem, we further develop an informative subspace projector, which trades off minimal distance precision for improved sample complexity. Extensive experiments demonstrate that CFR-Pro accurately matches units across different treatment groups, effectively mitigates treatment selection bias, and significantly outperforms competitors.
</details>

In this release, CFR-Pro is implemented in **[`src/estimator_gromov.py`](src/estimator_gromov.py)**. The `fgwpca` option combines PCA-based subspace projection with fused Gromov–Wasserstein balancing.


## Repository structure

```text
CFR-Pro/
├── src/                             
│   ├── estimator_gromov.py          # CFR-Pro
│   ├── estimator_cfr.py             # CFR-MMD and CFR-Wass
│   ├── estimator_escfr.py           # ESCFR
│   ├── estimator_descn.py           # DESCN
│   ├── estimator_site.py            # SITE
│   ├── estimator_metalearner.py     # Neural S-, T-, and Y-learners, where Y-learner is TARNet
│   ├── estimator_ml.py              # Classical and forest baselines
│   ├── base_estimator.py            # Shared neural training and evaluation
│   ├── models.py                   # Neural model definitions
│   ├── data_processor.py           # Benchmark preprocessing
│   └── ...                         # Metrics, optimization, and other helpers
├── scripts/                         
│   └── exp_overall.sh               # Shell scripts to reproduce Table 1 experiments
├── assets_sigkdd/                    
│   ├── baseline_final/              # Reproduced Table 1 results
│   ├── OverallPerformance.ipynb     # Notebook to analyze baseline_final/
│   ├── HyperparameterSensitivity.ipynb # Notebook to plot Figure 4
│   └── TransportVisualization.ipynb # Notebook to draw Figure 2
└── Datasets/                        
    ├── ACIC/
    └── IHDP/
```

| Directory | Purpose |
| :--- | :--- |
| **`src/`** | Contains the implementation of CFR-Pro and all baselines. **Each script can be run directly and tuned independently**. **CFR-Pro is `estimator_gromov.py`.** |
| **`scripts/`** | Contains the scripts for running the methods in `src/`, with concrete dataset-specific hyperparameters for the experiments underlying **Table 1**. |
| **`assets_sigkdd/`** | Contains saved experiment results and analysis notebooks. Notably, the results are reproduced in 20261126, with 2% fluctuation. |
| **`Datasets/`** | Contains the ACIC and IHDP benchmark data. |

## Installation and quick start

### 1. Set up the repository and dependencies

```bash
git clone https://github.com/haowangzju/CFR-Pro.git
cd CFR-Pro
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch tensorboard numpy pandas scipy scikit-learn POT PyYAML tqdm matplotlib seaborn econml jupyterlab
```



### 2. Run CFR-Pro on ACIC and IHDP

**Run estimator commands from the repository root.**
The following settings are consistent to `scripts/exp_overall.sh`. The default value is tuned for ACIC.

```bash
python src/estimator_gromov.py \
  --data ACIC \
  --seed 1 \
  --model fgwpca \
  --ot fgwpca \
  --root reproduction_results
```

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

For these examples, outputs are written to `root=reproduction_results/ACIC/fgwpca_1/` and `root=reproduction_results/IHDP/fgwpca_1/`. Neural runs save:

- `run.txt`: training, validation, and test metrics;
- `config.yaml`: the run's hyperparameters;
- `model.pt`: the selected model checkpoint;
- `events.out.tfevents.*`: TensorBoard logs.

**If `model.pt` already exists at the destination, the estimator_*.py code skips running.** So, use a fresh `--root` for a new experiment. 

### Running and tuning baseline methods

Each estimator_*.py can be executed and tuned with command-lines. Classical estimators can be tuned through `estimator_ml.py`.

For example:

```bash
python src/estimator_cfr.py --data ACIC --seed 1 --model cfr-mmd --root reproduction_results
python src/estimator_metalearner.py --data IHDP --seed 1 --model tlearner --root reproduction_results
```

## Reproducing Table 1

### Experiment recipe

[`scripts/exp_overall.sh`](scripts/exp_overall.sh) provides the concrete settings for the overall-performance experiments in **Table 1**:

```bash
# Run from the repository root.
bash scripts/exp_overall.sh
```

The bash file sets MAX_JOBS=16` and `CUDA_VISIBLE_DEVICES=-1`, so those jobs run on **CPU**. For individual CUDA runs, select a visible GPU with `CUDA_VISIBLE_DEVICES`.

The default output root is `assets_sigkdd/baseline_final`, where the supplied results are already stored. 
Notably, these results are reproduced in 20261127, with small (2%) variation for certain methods.


### Experiment result analyses

| Notebook | Purpose |
| :--- | :--- |
| [`OverallPerformance.ipynb`](assets_sigkdd/OverallPerformance.ipynb) | Aggregate ACIC and IHDP overall performance for Table 1 |
| [`HyperparameterSensitivity.ipynb`](assets_sigkdd/HyperparameterSensitivity.ipynb) | Analyze sensitivity to hyperparameters for Figure 4 | 
| [`TransportVisualization.ipynb`](assets_sigkdd/TransportVisualization.ipynb) | Visualize transport behavior on toy data for Figure 2 | 

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
