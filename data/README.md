# Data Sources

This study uses the **Behavioral Risk Factor Surveillance System (BRFSS)**, a CDC-administered annual telephone survey of U.S. adults.

## Required Raw Data

Download the SAS Transport (XPT) files from the CDC and place them in this directory:

| File | Year | Download URL |
|------|------|-------------|
| `LLCP2015.XPT` | 2015 | https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip |
| `LLCP2017.XPT` | 2017 | https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip |
| `LLCP2019.XPT` | 2019 | https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip |
| `LLCP2021.XPT` | 2021 | https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip |
| `LLCP2023.XPT` | 2023 | https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip |
| `LLCP2024.XPT` | 2024 | https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip |

Each file is approximately 1 GB. After unzipping, place the `.XPT` files directly in this `data/` directory.

## Processed Data

Running `src/01_data_harmonize.py` produces the following intermediate files (git-ignored):

- `pooled_raw.parquet` — All 6 years harmonized, before complete-case filtering
- `pooled_cc.parquet` — Complete-case analytic sample (1.92M observations)
- `pooled_imputed.parquet` — With imputed income values (from `src/02_imputation.py`)
- `cate_individual.parquet` — Individual-level CATE estimates (from `src/05_causal_forest.py`)
- `propensity_scores.parquet` — Propensity scores by age group (from `src/06_robustness.py`)

## Documentation

- `Overview_2024-508.pdf` — BRFSS 2024 survey overview
- `Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf` — Sampling design documentation
