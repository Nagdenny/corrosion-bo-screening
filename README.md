# Bayesian Optimization for Efficient Screening of Alloys with Enhanced Corrosion Resistance

Code accompanying the article:

> A. K. Nageri, F. Chen, B. Zhao, H. Zhang. *Bayesian Optimization for Efficient Screening
> of Alloys with Enhanced Corrosion Resistance*. (Manuscript under review.)

The framework combines supervised machine learning with closed-pool Bayesian optimization
(BO) and active classification to predict the corrosion behavior of stainless steels
(Severe / Poor / Good / Resistant) and to prioritize corrosion-resistant candidates.

![Graphical abstract](figures/Graphical_abstract.png)

## Repository structure

```
corrosion-bo-screening/
├── data/                       # place the two input datasets here (see data/README.md)
├── scripts/
│   ├── Machine Learning Model for Dataset No. 1.ipynb   # DT, RF, Bagging, XGBoost – Dataset No. 1
│   ├── Machine Learning Model for Dataset No. 2.ipynb   # same pipeline – Dataset No. 2
│   ├── bo_multiseed.py                  # closed-pool BO: RF/GP × EI/PI/UCB, one seed per run
│   ├── analyze_multiseed.py             # aggregates the five BO seeds
│   ├── active_classification.py         # active Bayesian classification: RF classifier × EI/PI/UCB/Random
│   ├── analyze_active_classification.py # aggregates the active-classification runs
│   ├── bo_fixed_environment.py          # fixed-environment alloy selection
│   ├── analyze_fixed_environment.py     # aggregates the fixed-environment runs
│   ├── corrosion_figstyle.py            # shared figure style (fonts, colours, sizes)
│   └── Plot figures.ipynb               # redraws the publication figures
├── figures/                    # publication figures (PNG, PDF, SVG)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Nagdenny/corrosion-bo-screening.git
cd corrosion-bo-screening
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then add the datasets to `data/` as described in [`data/README.md`](data/README.md).

## Reproducing the results

All commands are run from the repository root.

| Manuscript item | How to reproduce |
|---|---|
| Tables 3–4, Figures 3–6 (classification, feature importance, reduced-descriptor models) | Run the two `Machine Learning Model for Dataset No. *.ipynb` notebooks |
| Table 5, Figure 7 (closed-pool BO, Section 3.3) | `for s in 0 1 2 3 4; do BO_SEED=$s python scripts/bo_multiseed.py; done` then `python scripts/analyze_multiseed.py` |
| Table 6, Figures 8–9 (active Bayesian classification, Section 3.4) | `for s in 0 1 2 3 4; do BO_SEED=$s python scripts/active_classification.py; done` then `python scripts/analyze_active_classification.py` |
| Table 7, Figure 10 (fixed-environment selection, Section 3.5) | `FE_MODE=transfer python scripts/bo_fixed_environment.py`, `FE_MODE=cold python scripts/bo_fixed_environment.py`, then `python scripts/analyze_fixed_environment.py` |
| Publication-style figures | Run `Plot figures.ipynb` (after the steps above; reads from `results/`, writes to `results/figures_publication/`) |

Outputs are written to `results/`, which is not tracked by Git. The notebooks can be opened
from `scripts/` or from the repository root; they locate `data/` and `results/` automatically.

### Options

`bo_multiseed.py` reads the random seed from the `BO_SEED` environment variable (default `0`).
`active_classification.py` also reads the seed from `BO_SEED` (default `0`).
`bo_fixed_environment.py` accepts `FE_DATA`, `FE_SEEDS`, `FE_MODE` (`transfer` or `cold`),
`FE_KMAX`, `FE_OUT`, `FE_N_JOBS` and `FE_MAX_ENVS`; see the header of the script.

## Data

The datasets are from Hakimian et al. (2023), *Computational Materials Science* 228, 112352,
https://doi.org/10.1016/j.commatsci.2023.112352. They are not redistributed here.

## Citation

If you use this code, please cite the article (see `CITATION.cff`).

## Acknowledgements

Computing time was provided on the high-performance computer Lichtenberg at the NHR Centers
NHR4CES at TU Darmstadt. B. Zhao and H. Zhang acknowledge support from the Deutsche
Forschungsgemeinschaft (DFG) – CRC 1487 "Iron, upgraded!" – project number 443703006.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
