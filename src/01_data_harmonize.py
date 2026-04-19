"""
01_data_harmonize.py
Load 6 years of BRFSS (2015, 2017, 2019, 2021, 2023, 2024), harmonize
variable names/codings, and produce a pooled analytic dataset.

Outputs:
  data/pooled_raw.parquet        – all observations, pre-complete-case
  data/pooled_cc.parquet         – complete-case analytic sample
  tables/harmonize_summary.json  – sample sizes by year
"""

import pandas as pd, numpy as np, json, os, time, warnings

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)

YEARS = [2015, 2017, 2019, 2021, 2023, 2024]

# Variable name mappings across years.
# Sex variable changed from SEX -> SEXVAR around 2022-2024.
# Income grouping changed from _INCOMG (5 lvl) -> _INCOMG1 (7 lvl) around 2021.
SEX_CANDIDATES = ["SEXVAR", "SEX1", "SEX", "_SEX"]
INC_NEW = "_INCOMG1"       # 7-level, available 2021+
INC_OLD = "_INCOMG"        # 5-level, available 2015-2020
COMMON_COLS = ["MENTHLTH", "_TOTINDA", "_AGE_G", "_RACEGR3",
               "_EDUCAG", "_BMI5CAT", "_LLCPWT", "_STSTR", "_PSU"]

# _AGE80 provides continuous age top-coded at 80
AGE_CONT = "_AGE80"


def _find_col(df, candidates):
    """Return first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_year(year):
    path = f"data/LLCP{year}.XPT"
    print(f"  Loading {path} …", end=" ", flush=True)
    t0 = time.time()

    want = set(COMMON_COLS + SEX_CANDIDATES + [INC_NEW, INC_OLD, AGE_CONT])
    raw = pd.read_sas(path, format="xport", encoding="utf-8")
    avail = [c for c in want if c in raw.columns]
    df = raw[avail].copy()
    del raw

    # --- Sex harmonisation ---
    sex_col = _find_col(df, SEX_CANDIDATES)
    if sex_col and sex_col != "SEXVAR":
        df.rename(columns={sex_col: "SEXVAR"}, inplace=True)

    # --- Income harmonisation to 5-level common scale ---
    if INC_NEW in df.columns:
        # Collapse 5,6,7 -> 5  ($50K+)
        df["INCOME5"] = df[INC_NEW].copy()
        df.loc[df["INCOME5"].isin([6, 7]), "INCOME5"] = 5
    elif INC_OLD in df.columns:
        df["INCOME5"] = df[INC_OLD].copy()
    else:
        df["INCOME5"] = np.nan

    # Keep the 7-level variable for single-year analyses
    if INC_NEW in df.columns:
        df.rename(columns={INC_NEW: "INCOME7"}, inplace=True)
    else:
        df["INCOME7"] = np.nan

    # Continuous age
    if AGE_CONT not in df.columns:
        df[AGE_CONT] = np.nan

    df["YEAR"] = year
    keep = ["YEAR", "MENTHLTH", "_TOTINDA", "SEXVAR", "_AGE_G",
            "_RACEGR3", "_EDUCAG", "INCOME5", "INCOME7",
            "_BMI5CAT", "_LLCPWT", "_STSTR", "_PSU", AGE_CONT]
    df = df[[c for c in keep if c in df.columns]]

    elapsed = time.time() - t0
    print(f"{len(df):,} rows  ({elapsed:.1f}s)")
    return df


def recode(df):
    """Apply standard recoding to the pooled frame."""
    df["MENTHLTH"] = df["MENTHLTH"].replace({88: 0})
    df = df[df["MENTHLTH"].notna() & ~df["MENTHLTH"].isin([77, 99])].copy()
    df["FMD"] = (df["MENTHLTH"] >= 14).astype(np.int8)

    df = df[df["_TOTINDA"].isin([1, 2])].copy()
    df["PA"] = (df["_TOTINDA"] == 1).astype(np.int8)

    df = df[df["SEXVAR"].isin([1, 2])].copy()
    df["Female"] = (df["SEXVAR"] == 2).astype(np.int8)

    df = df[df["_AGE_G"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df[df["_RACEGR3"].isin([1, 2, 3, 4, 5])].copy()
    df = df[df["_EDUCAG"].isin([1, 2, 3, 4])].copy()
    df = df[df["_BMI5CAT"].isin([1, 2, 3, 4])].copy()
    df = df[df["_LLCPWT"].notna()].copy()

    return df


def main():
    print("=" * 60)
    print("BRFSS Multi-Year Harmonisation")
    print("=" * 60)

    frames = []
    for yr in YEARS:
        frames.append(load_year(yr))
    pooled = pd.concat(frames, ignore_index=True)
    print(f"\nRaw pooled: {len(pooled):,} rows across {len(YEARS)} years")

    # Recode
    pooled = recode(pooled)
    print(f"After basic recode (excl. income filter): {len(pooled):,}")

    # Save raw (before income complete-case) for imputation
    pooled.to_parquet("data/pooled_raw.parquet", index=False)
    print("Saved data/pooled_raw.parquet")

    # Complete-case: require valid income
    cc = pooled[pooled["INCOME5"].isin([1, 2, 3, 4, 5])].copy()

    # Adjust weight for pooling: divide by number of years
    cc["WEIGHT"] = cc["_LLCPWT"] / len(YEARS)

    print(f"Complete-case sample: {len(cc):,}  "
          f"({len(cc)/len(pooled)*100:.1f}% retention)")

    cc.to_parquet("data/pooled_cc.parquet", index=False)
    print("Saved data/pooled_cc.parquet")

    # Summary by year
    summary = {"n_years": len(YEARS), "years": YEARS}
    summary["n_raw_pooled"] = int(len(pooled))
    summary["n_cc_pooled"] = int(len(cc))
    year_info = {}
    for yr in YEARS:
        yr_raw = pooled[pooled["YEAR"] == yr]
        yr_cc = cc[cc["YEAR"] == yr]
        fmd_rate = (yr_cc.loc[yr_cc["FMD"] == 1, "WEIGHT"].sum() /
                    yr_cc["WEIGHT"].sum() * 100) if len(yr_cc) > 0 else 0
        year_info[str(yr)] = {
            "n_raw": int(len(yr_raw)),
            "n_cc": int(len(yr_cc)),
            "fmd_pct": round(fmd_rate, 1),
        }
    summary["by_year"] = year_info

    with open("tables/harmonize_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved tables/harmonize_summary.json")

    # Quick descriptive table
    print("\n── Year-level summary ──")
    for yr, info in year_info.items():
        print(f"  {yr}: n_raw={info['n_raw']:,}  n_cc={info['n_cc']:,}  "
              f"FMD={info['fmd_pct']}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
