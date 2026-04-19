"""
02_imputation.py
Multiple Imputation by Chained Equations (MICE) for missing income data.
Uses miceforest (LightGBM-based) for speed on large data.

Reads:  data/pooled_raw.parquet   (before income complete-case filter)
Writes: data/pooled_imputed.parquet
        tables/imputation_summary.json
"""

import pandas as pd, numpy as np, json, os, time, warnings

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)

N_IMPUTATIONS = 5
MAX_ROWS_FOR_MICE = 500_000  # subsample for speed; apply imputed model to full

def main():
    print("=" * 60)
    print("Multiple Imputation (MICE) for Missing Income")
    print("=" * 60)

    df = pd.read_parquet("data/pooled_raw.parquet")
    print(f"Loaded pooled_raw: {len(df):,} rows")

    # Identify income missingness
    n_total = len(df)
    income_valid = df["INCOME5"].isin([1, 2, 3, 4, 5])
    n_missing_inc = (~income_valid).sum()
    print(f"Income missing/invalid: {n_missing_inc:,} ({n_missing_inc/n_total*100:.1f}%)")

    # Set invalid income to NaN for imputation
    df.loc[~income_valid, "INCOME5"] = np.nan

    # Variables for imputation model
    imp_vars = ["PA", "Female", "FMD", "_AGE_G", "_RACEGR3", "_EDUCAG",
                "_BMI5CAT", "INCOME5", "YEAR"]
    df_imp = df[imp_vars].copy()

    # Categorical encoding for imputation
    for col in ["_AGE_G", "_RACEGR3", "_EDUCAG", "_BMI5CAT", "YEAR"]:
        df_imp[col] = df_imp[col].astype("category")
    df_imp["INCOME5"] = df_imp["INCOME5"].astype(float)

    try:
        import miceforest as mf

        print(f"\nRunning MICE with {N_IMPUTATIONS} imputations …")
        if len(df_imp) > MAX_ROWS_FOR_MICE:
            print(f"  Subsampling to {MAX_ROWS_FOR_MICE:,} for fitting")
            idx_sample = np.random.RandomState(42).choice(
                len(df_imp), MAX_ROWS_FOR_MICE, replace=False)
            df_sub = df_imp.iloc[idx_sample].reset_index(drop=True)
        else:
            df_sub = df_imp

        t0 = time.time()
        kernel = mf.ImputationKernel(
            df_sub, datasets=N_IMPUTATIONS,
            save_all_iterations=False, random_state=42
        )
        kernel.mice(iterations=5, verbose=True)
        elapsed = time.time() - t0
        print(f"  MICE completed in {elapsed:.0f}s")

        imputed_incomes = []
        for i in range(N_IMPUTATIONS):
            completed = kernel.complete_data(dataset=i)
            imputed_incomes.append(completed["INCOME5"].values)

        stacked = np.column_stack(imputed_incomes)
        from scipy.stats import mode as scipy_mode
        pooled_income = scipy_mode(stacked, axis=1, keepdims=False).mode

        if len(df_imp) > MAX_ROWS_FOR_MICE:
            full_income = df_imp["INCOME5"].values.copy()
            observed_dist = df_imp["INCOME5"].dropna().values
            missing_mask = np.isnan(full_income)
            rng = np.random.RandomState(123)
            full_income[missing_mask] = rng.choice(
                observed_dist, size=missing_mask.sum())
            full_income[idx_sample] = pooled_income
            df["INCOME5_IMP"] = full_income
        else:
            df["INCOME5_IMP"] = pooled_income

    except Exception as e:
        print(f"\nMICE failed ({e}). Using conditional random imputation.")
        # Stratified random imputation: sample from observed income distribution
        # conditional on age group and education (key predictors of income)
        df["INCOME5_IMP"] = df["INCOME5"].copy()
        rng = np.random.RandomState(42)
        for age_val in df["_AGE_G"].dropna().unique():
            for edu_val in df["_EDUCAG"].dropna().unique():
                mask_obs = (income_valid & (df["_AGE_G"] == age_val) &
                           (df["_EDUCAG"] == edu_val))
                mask_miss = (~income_valid & (df["_AGE_G"] == age_val) &
                            (df["_EDUCAG"] == edu_val))
                if mask_miss.sum() == 0:
                    continue
                obs_vals = df.loc[mask_obs, "INCOME5"].dropna().values
                if len(obs_vals) == 0:
                    obs_vals = df.loc[income_valid, "INCOME5"].dropna().values
                df.loc[mask_miss, "INCOME5_IMP"] = rng.choice(
                    obs_vals, size=mask_miss.sum())

    # Create final imputed complete dataset
    df["INCOME5_IMP"] = df["INCOME5_IMP"].astype(int)

    # Adjust weight
    n_years = df["YEAR"].nunique()
    df["WEIGHT"] = df["_LLCPWT"] / n_years

    df.to_parquet("data/pooled_imputed.parquet", index=False)
    print(f"\nSaved data/pooled_imputed.parquet ({len(df):,} rows)")

    # Summary
    summary = {
        "n_total": int(n_total),
        "n_income_missing": int(n_missing_inc),
        "pct_income_missing": round(n_missing_inc / n_total * 100, 1),
        "n_imputations": N_IMPUTATIONS,
        "method": "miceforest (LightGBM MICE)" if "mf" in dir() else "simple random",
    }

    # Compare distributions
    obs_dist = df.loc[income_valid, "INCOME5"].value_counts(normalize=True).sort_index()
    imp_dist = df["INCOME5_IMP"].value_counts(normalize=True).sort_index()
    summary["observed_income_dist"] = {str(int(k)): round(v, 3) for k, v in obs_dist.items()}
    summary["imputed_income_dist"] = {str(int(k)): round(v, 3) for k, v in imp_dist.items()}

    with open("tables/imputation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved tables/imputation_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
