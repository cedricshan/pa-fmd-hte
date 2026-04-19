"""
06_robustness.py
Robustness and sensitivity analyses:
  (a) E-values for unmeasured confounding
  (b) Propensity score overlap by age group
  (c) Placebo/falsification test
  (d) Comparison: complete-case vs imputed

Reads:  data/pooled_cc.parquet, data/pooled_imputed.parquet (if exists)
Writes: tables/evalues.csv
        tables/propensity_overlap.csv
        tables/placebo_test.csv
        tables/cc_vs_imputed.csv
"""

import pandas as pd, numpy as np, json, os, warnings
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)

AGE_MAP = {1: "18-24", 2: "25-34", 3: "35-44",
           4: "45-54", 5: "55-64", 6: "65+"}


def compute_evalue(or_val, ci_lower=None):
    """
    Compute the E-value (VanderWeele & Ding, 2017).
    For OR: E = OR + sqrt(OR × (OR - 1))  [when OR >= 1]
    For protective OR < 1: convert to 1/OR first.
    """
    if np.isnan(or_val) or or_val <= 0:
        return np.nan, np.nan
    if or_val < 1:
        or_val = 1 / or_val
    e_point = or_val + np.sqrt(or_val * (or_val - 1))
    e_ci = np.nan
    if ci_lower is not None and not np.isnan(ci_lower):
        ci_or = ci_lower if ci_lower >= 1 else 1 / ci_lower
        if ci_or > 1:
            e_ci = ci_or + np.sqrt(ci_or * (ci_or - 1))
        else:
            e_ci = 1.0
    return round(e_point, 2), round(e_ci, 2)


def build_X(df):
    race_d = pd.get_dummies(df["_RACEGR3"], prefix="R", drop_first=True, dtype=int)
    edu_d = pd.get_dummies(df["_EDUCAG"], prefix="E", drop_first=True, dtype=int)
    inc_d = pd.get_dummies(df["INCOME5"], prefix="I", drop_first=True, dtype=int)
    bmi_d = pd.get_dummies(df["_BMI5CAT"], prefix="B", drop_first=True, dtype=int)
    age_d = pd.get_dummies(df["_AGE_G"], prefix="A", drop_first=True, dtype=int)
    X = pd.concat([df[["PA", "Female"]].reset_index(drop=True),
                    age_d.reset_index(drop=True), race_d.reset_index(drop=True),
                    edu_d.reset_index(drop=True), inc_d.reset_index(drop=True),
                    bmi_d.reset_index(drop=True)], axis=1)
    return X


def fit_pa_or(df_sub):
    """Fit weighted logistic and return PA OR + CI."""
    df_sub = df_sub.reset_index(drop=True)
    X = build_X(df_sub)
    Xc = sm.add_constant(X)
    y = df_sub["FMD"].values
    w = df_sub["WEIGHT"].values if "WEIGHT" in df_sub.columns else df_sub["_LLCPWT"].values
    try:
        m = GLM(y, Xc, family=families.Binomial(), freq_weights=w).fit(
            maxiter=100, disp=False)
        ci = m.conf_int().loc["PA"]
        return np.exp(m.params["PA"]), np.exp(ci[0]), np.exp(ci[1])
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    print("=" * 60)
    print("Robustness & Sensitivity Analyses")
    print("=" * 60)

    df = pd.read_parquet("data/pooled_cc.parquet")
    print(f"Loaded pooled_cc: {len(df):,}")

    # ════════════════════════════════════════════
    # (a) E-values
    # ════════════════════════════════════════════
    print("\n── (a) E-values for Unmeasured Confounding ──")
    evalue_rows = []

    # Overall
    or_all, ci_l_all, ci_u_all = fit_pa_or(df)
    e_pt, e_ci = compute_evalue(or_all, ci_u_all)
    evalue_rows.append({
        "Subgroup": "Overall", "OR": round(or_all, 3),
        "CI_Lower": round(ci_l_all, 3), "CI_Upper": round(ci_u_all, 3),
        "E_value_point": e_pt, "E_value_CI": e_ci,
    })

    # By age group
    for age_val, lab in AGE_MAP.items():
        sub = df[df["_AGE_G"] == age_val]
        o, cl, cu = fit_pa_or(sub)
        # For protective effects (OR<1), E-value uses 1/OR
        # For null effect (OR~1), E-value is close to 1 (robust)
        ep, ec = compute_evalue(o, cu if o < 1 else cl)
        evalue_rows.append({
            "Subgroup": f"Age: {lab}", "OR": round(o, 3),
            "CI_Lower": round(cl, 3), "CI_Upper": round(cu, 3),
            "E_value_point": ep, "E_value_CI": ec,
        })

    ev_df = pd.DataFrame(evalue_rows)
    ev_df.to_csv("tables/evalues.csv", index=False)
    print(ev_df.to_string(index=False))
    print("Saved tables/evalues.csv")

    # ════════════════════════════════════════════
    # (b) Propensity Score Overlap
    # ════════════════════════════════════════════
    print("\n── (b) Propensity Score Overlap ──")

    # Build features for propensity model
    feat_cols = []
    for c in ["Female", "_AGE_G", "_RACEGR3", "_EDUCAG", "INCOME5", "_BMI5CAT"]:
        feat_cols.append(c)
    X_prop = df[feat_cols].values.astype(float)
    T_prop = df["PA"].values
    W_prop = df["WEIGHT"].values

    # Subsample for speed
    rng = np.random.RandomState(42)
    n_prop = min(200_000, len(df))
    idx = rng.choice(len(df), n_prop, replace=False)

    print(f"  Fitting propensity model on {n_prop:,} observations …")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.1, subsample=0.8,
                                     random_state=42)
    gb.fit(X_prop[idx], T_prop[idx], sample_weight=W_prop[idx])
    ps_all = gb.predict_proba(X_prop)[:, 1]

    overlap_rows = []
    for age_val, lab in AGE_MAP.items():
        mask = df["_AGE_G"].values == age_val
        ps_ag = ps_all[mask]
        t_ag = T_prop[mask]

        ps_treated = ps_ag[t_ag == 1]
        ps_control = ps_ag[t_ag == 0]
        overlap_rows.append({
            "Age_Group": lab,
            "n": int(mask.sum()),
            "PS_mean_treated": round(float(ps_treated.mean()), 3),
            "PS_mean_control": round(float(ps_control.mean()), 3),
            "PS_overlap_pct": round(float(
                ((ps_treated > np.percentile(ps_control, 5)) &
                 (ps_treated < np.percentile(ps_control, 95))).mean() * 100), 1),
            "PS_min_treated": round(float(ps_treated.min()), 3),
            "PS_max_control": round(float(ps_control.max()), 3),
        })

    ps_df = pd.DataFrame(overlap_rows)
    ps_df.to_csv("tables/propensity_overlap.csv", index=False)
    print(ps_df.to_string(index=False))
    print("Saved tables/propensity_overlap.csv")

    # Save propensity scores for figures
    ps_out = pd.DataFrame({
        "age_group": df["_AGE_G"].values,
        "pa": T_prop,
        "ps": ps_all,
    })
    ps_out.to_parquet("data/propensity_scores.parquet", index=False)

    # ════════════════════════════════════════════
    # (c) Placebo / Falsification Test
    # ════════════════════════════════════════════
    print("\n── (c) Placebo Test ──")
    # Use general health (GENHLTH) recoded as binary "poor/fair health"
    # as a positive control, and check if age heterogeneity pattern differs
    # Since GENHLTH is not in our pooled data, we use the 2024 data directly

    # We'll test: does the age heterogeneity pattern hold for a different
    # binary outcome derived from the SAME data?
    # Placebo outcome: "being female" (should show NO PA effect heterogeneity by age)
    # This is a true placebo since PA cannot cause sex.

    print("  Placebo outcome: Female (PA should have zero effect)")
    placebo_rows = []
    for age_val, lab in AGE_MAP.items():
        sub = df[df["_AGE_G"] == age_val].reset_index(drop=True)
        race_d = pd.get_dummies(sub["_RACEGR3"], prefix="R", drop_first=True, dtype=int)
        edu_d = pd.get_dummies(sub["_EDUCAG"], prefix="E", drop_first=True, dtype=int)
        inc_d = pd.get_dummies(sub["INCOME5"], prefix="I", drop_first=True, dtype=int)
        bmi_d = pd.get_dummies(sub["_BMI5CAT"], prefix="B", drop_first=True, dtype=int)
        X_p = pd.concat([sub[["PA"]].reset_index(drop=True),
                          race_d.reset_index(drop=True), edu_d.reset_index(drop=True),
                          inc_d.reset_index(drop=True), bmi_d.reset_index(drop=True)],
                         axis=1)
        Xc = sm.add_constant(X_p)
        y_placebo = sub["Female"].values
        w_p = sub["WEIGHT"].values
        try:
            m = GLM(y_placebo, Xc, family=families.Binomial(), freq_weights=w_p).fit(
                maxiter=100, disp=False)
            ci = m.conf_int().loc["PA"]
            placebo_rows.append({
                "Age_Group": lab,
                "Outcome": "Female (placebo)",
                "OR": round(float(np.exp(m.params["PA"])), 3),
                "CI_L": round(float(np.exp(ci[0])), 3),
                "CI_U": round(float(np.exp(ci[1])), 3),
            })
        except Exception:
            placebo_rows.append({
                "Age_Group": lab, "Outcome": "Female (placebo)",
                "OR": np.nan, "CI_L": np.nan, "CI_U": np.nan,
            })

    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv("tables/placebo_test.csv", index=False)
    print(placebo_df.to_string(index=False))
    print("Saved tables/placebo_test.csv")

    # ════════════════════════════════════════════
    # (d) Complete-case vs Imputed comparison
    # ════════════════════════════════════════════
    print("\n── (d) Complete-Case vs Imputed ──")
    imp_path = "data/pooled_imputed.parquet"
    if os.path.exists(imp_path):
        df_imp = pd.read_parquet(imp_path)
        # Use imputed income
        df_imp_valid = df_imp.copy()
        df_imp_valid["INCOME5"] = df_imp_valid["INCOME5_IMP"]
        df_imp_valid = df_imp_valid[df_imp_valid["INCOME5"].isin([1,2,3,4,5])]

        comp_rows = []
        for label, data in [("Complete-case", df), ("Imputed", df_imp_valid)]:
            for age_val, alab in AGE_MAP.items():
                sub = data[data["_AGE_G"] == age_val]
                o, cl, cu = fit_pa_or(sub)
                comp_rows.append({
                    "Method": label, "Age_Group": alab,
                    "n": len(sub), "OR": round(o, 3),
                    "CI_L": round(cl, 3), "CI_U": round(cu, 3),
                })

        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv("tables/cc_vs_imputed.csv", index=False)
        print(comp_df.to_string(index=False))
        print("Saved tables/cc_vs_imputed.csv")
    else:
        print("  Imputed data not found; skipping comparison.")

    print("\nDone.")


if __name__ == "__main__":
    main()
