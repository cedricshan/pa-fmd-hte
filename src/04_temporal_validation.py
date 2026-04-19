"""
04_temporal_validation.py
Year-by-year age-stratified PA odds ratios to validate the 18-24 null
finding across 2015-2024.  Also tests three-way YEAR × PA × Age interaction.

Reads:  data/pooled_cc.parquet
Writes: tables/temporal_age_or.csv
        tables/temporal_trend_test.json
"""

import pandas as pd, numpy as np, json, os, warnings
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)

AGE_MAP = {1: "18-24", 2: "25-34", 3: "35-44",
           4: "45-54", 5: "55-64", 6: "65+"}


def build_X(df):
    """Minimal design matrix for stratified models within one age-year cell."""
    race_d = pd.get_dummies(df["_RACEGR3"], prefix="R", drop_first=True, dtype=int)
    edu_d = pd.get_dummies(df["_EDUCAG"], prefix="E", drop_first=True, dtype=int)
    inc_d = pd.get_dummies(df["INCOME5"], prefix="I", drop_first=True, dtype=int)
    bmi_d = pd.get_dummies(df["_BMI5CAT"], prefix="B", drop_first=True, dtype=int)
    X = pd.concat([df[["PA", "Female"]].reset_index(drop=True),
                    race_d.reset_index(drop=True), edu_d.reset_index(drop=True),
                    inc_d.reset_index(drop=True), bmi_d.reset_index(drop=True)], axis=1)
    return X


def fit_pa_or(subset):
    """Return PA OR and 95% CI from a weighted logistic regression."""
    df = subset.reset_index(drop=True)
    X = build_X(df)
    Xc = sm.add_constant(X)
    y = df["FMD"].values
    w = df["_LLCPWT"].values
    try:
        model = GLM(y, Xc, family=families.Binomial(), freq_weights=w).fit(
            maxiter=100, disp=False)
        ci = model.conf_int().loc["PA"]
        return (np.exp(model.params["PA"]),
                np.exp(ci[0]), np.exp(ci[1]),
                float(model.pvalues["PA"]))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


def main():
    print("=" * 60)
    print("Temporal Validation: Year-by-Year Age-Stratified PA ORs")
    print("=" * 60)

    df = pd.read_parquet("data/pooled_cc.parquet")
    years = sorted(df["YEAR"].unique())
    print(f"Loaded {len(df):,} rows, years: {years}")

    # ── Year × Age stratified ORs ──
    rows = []
    for yr in years:
        print(f"\n── {yr} ──")
        df_yr = df[df["YEAR"] == yr]
        for age_val, age_lab in AGE_MAP.items():
            subset = df_yr[df_yr["_AGE_G"] == age_val]
            n = len(subset)
            if n < 100:
                print(f"  Age {age_lab}: n={n} (too small, skipping)")
                continue
            or_val, ci_l, ci_u, p = fit_pa_or(subset)
            rows.append({
                "Year": yr, "Age_Group": age_lab,
                "n": n, "OR": or_val, "CI_L": ci_l, "CI_U": ci_u, "p": p
            })
            print(f"  Age {age_lab}: n={n:,}  OR={or_val:.3f} "
                  f"({ci_l:.3f}–{ci_u:.3f})")

    temporal_df = pd.DataFrame(rows)
    temporal_df.to_csv("tables/temporal_age_or.csv", index=False)
    print("\nSaved tables/temporal_age_or.csv")

    # ── Summary: is the 18-24 null finding consistent? ──
    young = temporal_df[temporal_df["Age_Group"] == "18-24"]
    print("\n── 18-24 PA OR across years ──")
    for _, r in young.iterrows():
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else
              ("*" if r["p"] < 0.05 else "ns"))
        print(f"  {int(r['Year'])}: OR={r['OR']:.3f}  {sig}")

    # ── Three-way interaction test: PA × Age × Year (linear trend) ──
    print("\n── Three-way Interaction Test ──")
    df_test = df.copy()
    df_test["YEAR_C"] = (df_test["YEAR"] - df_test["YEAR"].median()) / 5
    df_test["AGE_C"] = (df_test["_AGE_G"] - 3.5) / 2.5

    # Main model: PA + controls + PA×AGE_C
    X_base = build_X(df_test)
    X_base["PA_x_AGE"] = X_base["PA"] * df_test["AGE_C"].values

    # Extended model: + PA×AGE_C×YEAR_C
    X_ext = X_base.copy()
    X_ext["PA_x_AGE_x_YEAR"] = (X_base["PA"] *
                                  df_test["AGE_C"].values *
                                  df_test["YEAR_C"].values)

    y = df_test["FMD"].reset_index(drop=True).values
    w = df_test["_LLCPWT"].values

    Xb = sm.add_constant(X_base)
    Xe = sm.add_constant(X_ext)

    m_base = GLM(y, Xb, family=families.Binomial(), freq_weights=w).fit(
        maxiter=100, disp=False)
    m_ext = GLM(y, Xe, family=families.Binomial(), freq_weights=w).fit(
        maxiter=100, disp=False)

    lr = m_base.deviance - m_ext.deviance
    lr_p = 1 - stats.chi2.cdf(lr, df=1)
    three_way_or = np.exp(m_ext.params["PA_x_AGE_x_YEAR"])

    trend_result = {
        "lr_chi2": round(float(lr), 1),
        "lr_p": f"{lr_p:.4e}",
        "three_way_or": round(float(three_way_or), 4),
        "interpretation": ("Age-dependent PA effect is "
                          + ("stable" if lr_p > 0.01 else "changing")
                          + " over 2015-2024"),
    }
    with open("tables/temporal_trend_test.json", "w") as f:
        json.dump(trend_result, f, indent=2)
    print(f"  3-way LR chi2 = {lr:.1f}, p = {lr_p:.4e}")
    print(f"  3-way OR = {three_way_or:.4f}")
    print(f"  → {trend_result['interpretation']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
