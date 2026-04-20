# Data Sources

This study uses the **Behavioral Risk Factor Surveillance System (BRFSS)**, a CDC-administered annual telephone survey of U.S. adults.

## Required Raw Data

Download all 10 annual SAS Transport (XPT) files from the CDC and place them in this directory:

| File | Year | Download URL |
|------|------|-------------|
| `LLCP2015.XPT` | 2015 | https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip |
| `LLCP2016.XPT` | 2016 | https://www.cdc.gov/brfss/annual_data/2016/files/LLCP2016XPT.zip |
| `LLCP2017.XPT` | 2017 | https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip |
| `LLCP2018.XPT` | 2018 | https://www.cdc.gov/brfss/annual_data/2018/files/LLCP2018XPT.zip |
| `LLCP2019.XPT` | 2019 | https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip |
| `LLCP2020.XPT` | 2020 | https://www.cdc.gov/brfss/annual_data/2020/files/LLCP2020XPT.zip |
| `LLCP2021.XPT` | 2021 | https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip |
| `LLCP2022.XPT` | 2022 | https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip |
| `LLCP2023.XPT` | 2023 | https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip |
| `LLCP2024.XPT` | 2024 | https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip |

Each file is approximately 1 GB. After unzipping, place the `.XPT` files directly in this directory.

## Processed Data

Running `src/01_data_harmonize.py` produces the following intermediate files (git-ignored due to size):

| File | Description | Rows |
|------|-------------|------|
| `pooled_raw.parquet` | All 10 years harmonized, before complete-case filtering | 3,803,597 |
| `pooled_cc.parquet` | Complete-case analytic sample | 3,242,218 |
| `pooled_imputed.parquet` | With imputed income values (from `02_imputation.py`) | 3,803,597 |
| `cate_individual.parquet` | Individual-level CATE estimates (from `05_causal_forest.py`) | 49,998 |
| `propensity_scores.parquet` | Propensity scores by age group (from `06_robustness.py`) | 3,242,218 |

## Variable Harmonization Notes

- **Sex**: Variable name changed across years (`SEX` -> `SEX1` -> `SEXVAR`); all harmonized to `SEXVAR`.
- **Race/ethnicity**: 2022 renamed `_RACEGR3` to `_RACEGR4` with identical coding; harmonized back to `_RACEGR3`.
- **Income**: Pre-2021 uses `_INCOMG` (5 levels); 2021+ uses `_INCOMG1` (7 levels). Harmonized to a common 5-level scale by collapsing upper categories.

## Documentation

- `Overview_2024-508.pdf` -- BRFSS 2024 survey overview
- `Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf` -- Sampling design documentation
