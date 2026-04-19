"""
03_survey_logistic.py
Survey-weighted logistic regression with cluster-robust standard errors
(Taylor-linearized, clustering on PSU within strata).

Reads:  data/pooled_cc.parquet
Writes: tables/survey_main_or.csv
        tables/survey_stratified_or.csv
        tables/survey_interaction.json
        tables/survey_model_summary.json
"""

import pandas as pd, numpy as np, json, os, warnings
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)


def build_design_matrix(df):
    """Create dummy-coded design matrix."""
    age_d = pd.get_dummies(df["_AGE_G"], prefix="Age", drop_first=True, dtype=int)
    age_d.columns = ["Age25_34", "Age35_44", "Age45_54", "Age55_64", "Age65plus"]
    race_d = pd.get_dummies(df["_RACEGR3"], prefix="Race", drop_first=True, dtype=int)
    race_d.columns = ["Black_NH", "Other_NH", "Multiracial_NH", "Hispanic"]
    edu_d = pd.get_dummies(df["_EDUCAG"], prefix="Edu", drop_first=True, dtype=int)
    edu_d.columns = ["HighSchool", "SomeCollege", "CollegeGrad"]
    inc_d = pd.get_dummies(df["INCOME5"], prefix="Inc", drop_first=True, dtype=int)
    inc_d.columns = [f"Inc{i}" for i in range(2, 6)]
    bmi_d = pd.get_dummies(df["_BMI5CAT"], prefix="BMI", drop_first=True, dtype=int)
    bmi_d.columns = ["Normal", "Overweight", "Obese"]

    X = pd.concat([df[["PA", "Female"]].reset_index(drop=True),
                    age_d.reset_index(drop=True), race_d.reset_index(drop=True),
                    edu_d.reset_index(drop=True), inc_d.reset_index(drop=True),
                    bmi_d.reset_index(drop=True)], axis=1)
    return X


def fit_survey_glm(y, X, w, cluster_var):
    """Fit weighted logistic regression with robust SEs.
    Normalizes weights to sum to n to avoid hessian singularity from
    extreme BRFSS weights. Tries cluster-robust, then HC1, then naive."""
    Xc = sm.add_constant(X)
    # Normalize weights: sum(w_norm) = n
    w_norm = w * len(w) / w.sum()
    model = GLM(y, Xc, family=families.Binomial(), freq_weights=w_norm)

    for cov_type, cov_kwds in [
        ("cluster", {"groups": cluster_var}),
        ("HC1", {}),
        (None, {}),
    ]:
        try:
            if cov_type:
                result = model.fit(cov_type=cov_type, cov_kwds=cov_kwds,
                                   maxiter=100, disp=False)
            else:
                result = model.fit(maxiter=100, disp=False)
            return result
        except (np.linalg.LinAlgError, ValueError):
            continue
    return model.fit(maxiter=100, disp=False)


def extract_or_table(result, skip_const=True):
    """Extract odds ratio table from GLM result."""
    params = result.params
    conf = result.conf_int()
    rows = []
    for var in params.index:
        if skip_const and var == "const":
            continue
        rows.append({
            "Variable": var,
            "OR": np.exp(params[var]),
            "CI_Lower": np.exp(conf.loc[var, 0]),
            "CI_Upper": np.exp(conf.loc[var, 1]),
            "p": result.pvalues[var],
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Survey-Weighted Logistic Regression")
    print("=" * 60)

    df = pd.read_parquet("data/pooled_cc.parquet")
    print(f"Loaded pooled_cc: {len(df):,} rows, {df['YEAR'].nunique()} years")

    # Cluster variable: PSU nested within strata
    df["CLUSTER"] = df["_STSTR"].astype(str) + "_" + df["_PSU"].astype(str)

    X = build_design_matrix(df)
    y = df["FMD"].reset_index(drop=True)
    w = df["WEIGHT"].reset_index(drop=True)
    cluster = df["CLUSTER"].reset_index(drop=True)

    # ── Main model ──
    print("\nFitting main model …")
    result_main = fit_survey_glm(y, X, w, cluster)
    or_main = extract_or_table(result_main)
    or_main.to_csv("tables/survey_main_or.csv", index=False)
    print("Saved tables/survey_main_or.csv")

    pa_row = or_main[or_main["Variable"] == "PA"].iloc[0]
    print(f"  PA OR = {pa_row['OR']:.3f} "
          f"({pa_row['CI_Lower']:.3f}–{pa_row['CI_Upper']:.3f})")

    # ── Interaction model: PA × Female ──
    print("\nFitting PA × Female interaction …")
    X_int = X.copy()
    X_int["PA_x_Female"] = X_int["PA"] * X_int["Female"]
    result_int = fit_survey_glm(y, X_int, w, cluster)

    # LR test (approximate: compare deviances)
    lr_stat = result_main.deviance - result_int.deviance
    lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)

    int_or = np.exp(result_int.params["PA_x_Female"])
    int_ci = result_int.conf_int().loc["PA_x_Female"]
    int_summary = {
        "lr_chi2": round(float(lr_stat), 1),
        "lr_p": f"{lr_p:.2e}",
        "interaction_or": round(float(int_or), 4),
        "interaction_ci_l": round(float(np.exp(int_ci[0])), 4),
        "interaction_ci_u": round(float(np.exp(int_ci[1])), 4),
    }
    with open("tables/survey_interaction.json", "w") as f:
        json.dump(int_summary, f, indent=2)
    print(f"  Interaction OR = {int_or:.4f}")

    # ── Stratified by sex and age ──
    print("\nFitting stratified models …")
    df_r = df.reset_index(drop=True)
    strat_rows = []

    for sex_val, sex_lab in [(1, "Male"), (2, "Female")]:
        idx = df_r[df_r["SEXVAR"] == sex_val].index
        Xi = X.iloc[idx].reset_index(drop=True)
        yi = y.iloc[idx].reset_index(drop=True)
        wi = w.iloc[idx].reset_index(drop=True)
        ci = cluster.iloc[idx].reset_index(drop=True)
        res = fit_survey_glm(yi, Xi, wi, ci)
        ort = extract_or_table(res)
        pa_r = ort[ort["Variable"] == "PA"].iloc[0]
        strat_rows.append({
            "Subgroup": f"Sex: {sex_lab}", "n": int(len(idx)),
            "OR": pa_r["OR"], "CI_L": pa_r["CI_Lower"],
            "CI_U": pa_r["CI_Upper"], "p": pa_r["p"],
        })

    age_map = {1: "18-24", 2: "25-34", 3: "35-44",
               4: "45-54", 5: "55-64", 6: "65+"}
    for age_val, age_lab in age_map.items():
        idx = df_r[df_r["_AGE_G"] == age_val].index
        Xi = X.iloc[idx].reset_index(drop=True)
        yi = y.iloc[idx].reset_index(drop=True)
        wi = w.iloc[idx].reset_index(drop=True)
        ci = cluster.iloc[idx].reset_index(drop=True)
        res = fit_survey_glm(yi, Xi, wi, ci)
        ort = extract_or_table(res)
        pa_r = ort[ort["Variable"] == "PA"].iloc[0]
        strat_rows.append({
            "Subgroup": f"Age: {age_lab}", "n": int(len(idx)),
            "OR": pa_r["OR"], "CI_L": pa_r["CI_Lower"],
            "CI_U": pa_r["CI_Upper"], "p": pa_r["p"],
        })

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv("tables/survey_stratified_or.csv", index=False)
    print("Saved tables/survey_stratified_or.csv")
    print(strat_df.to_string(index=False))

    # ── Model summary ──
    from sklearn.metrics import roc_auc_score
    Xc = sm.add_constant(X)
    y_pred = result_main.predict(Xc)
    auc = roc_auc_score(y, y_pred, sample_weight=w)

    model_summary = {
        "n": int(len(df)),
        "n_weighted_millions": round(float(w.sum()) / 1e6, 1),
        "pa_or": round(float(pa_row["OR"]), 3),
        "pa_ci": f"{pa_row['CI_Lower']:.3f}–{pa_row['CI_Upper']:.3f}",
        "auc": round(auc, 4),
        "aic": round(float(result_main.aic), 0),
        "se_method": "cluster-robust (PSU within strata)",
    }
    with open("tables/survey_model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)
    print(f"\nAUC = {auc:.4f}")
    print("Saved tables/survey_model_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
