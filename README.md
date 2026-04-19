# Age-Dependent Heterogeneity in the Association Between Physical Activity and Mental Distress

A causal machine learning analysis of 1.9 million U.S. adults from the Behavioral Risk Factor Surveillance System (BRFSS), 2015–2024.

## Key Findings

1. **Physical activity (PA) is associated with 37% lower odds of frequent mental distress (FMD)** overall (adjusted OR = 0.633, 95% CI: 0.620–0.647), consistent with prior large-scale evidence.

2. **The protective effect is profoundly age-dependent**: OR ranges from 0.87 (18–24, weak) to 0.52 (65+, strong), forming a monotonic gradient that has not been previously documented in the literature.

3. **The young-adult PA effect has been eroding over time**: the 18–24 PA OR declined from 0.76 (2019) to 0.92 (2023) to 1.01 (2024, null), paralleling the deepening youth mental health crisis.

4. **Causal Forest analysis confirms age as the dominant driver** of treatment effect heterogeneity (feature importance = 0.39, 2.5x the next feature), validating the finding through nonparametric causal machine learning.

## Analytical Pipeline

![Analytical Pipeline](figures/pipeline.png)

## Repository Structure

```
├── report.qmd              # Paper (Quarto → PDF)
├── report.pdf              # Rendered manuscript
├── references.bib          # Bibliography (25 citations)
├── apa.csl                 # APA 7th citation style
├── requirements.txt        # Python dependencies
│
├── src/                    # Analysis pipeline (run in order)
│   ├── 01_data_harmonize.py    # Load & harmonize 6 BRFSS years
│   ├── 02_imputation.py        # Multiple imputation for missing income
│   ├── 03_survey_logistic.py   # Survey-weighted logistic regression
│   ├── 04_temporal_validation.py  # Year-by-year age-stratified ORs
│   ├── 05_causal_forest.py     # CausalForestDML (HTE discovery)
│   ├── 06_robustness.py        # E-values, propensity overlap, placebo
│   └── 07_figures.py           # Publication-quality figures
│
├── data/                   # Raw & processed data (see data/README.md)
├── tables/                 # Analysis outputs (CSV, JSON)
├── figures/                # Publication figures (PNG)
└── dcc/                    # SLURM job script for Duke Compute Cluster
```

## Reproducing the Analysis

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data

Download the 6 BRFSS XPT files as described in [`data/README.md`](data/README.md).

### Pipeline

Run scripts sequentially from the project root:

```bash
python src/01_data_harmonize.py     # ~3 min (loads 6 × 1GB files)
python src/02_imputation.py         # ~5 sec
python src/03_survey_logistic.py    # ~1 min
python src/04_temporal_validation.py # ~30 sec
python src/05_causal_forest.py      # ~60 min (or use dcc/dcc_job.sh on cluster)
python src/06_robustness.py         # ~1 min
python src/07_figures.py            # ~5 sec
```

Step 5 (Causal Forest) is computationally intensive. A SLURM job script for HPC clusters is provided in `dcc/dcc_job.sh`.

### Render Paper

```bash
quarto render report.qmd --to pdf
```

## Methods

- **Survey-weighted logistic regression** with cluster-robust standard errors on 1.92M complete-case observations
- **Causal Forest via Double Machine Learning** (EconML `CausalForestDML`) with HistGradientBoosting nuisance models
- **Temporal validation** across 6 independent BRFSS waves (2015–2024)
- **Sensitivity analyses**: E-values for unmeasured confounding, propensity score overlap, placebo/falsification test, complete-case vs. imputation comparison

## Data

All data are publicly available from the CDC BRFSS:
https://www.cdc.gov/brfss/annual_data/annual_data.htm

## Author

Yuan Shan

## License

This project is for academic research purposes. The BRFSS data are public domain.
