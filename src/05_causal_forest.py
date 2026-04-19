"""
05_causal_forest.py
Causal Forest via Double Machine Learning (CausalForestDML) to estimate
heterogeneous treatment effects of PA on FMD.

Optimized for DCC cluster: HistGradientBoosting + multi-core parallelism.

Reads:  data/pooled_cc.parquet
Writes: tables/cate_by_age.csv
        tables/cate_summary.json
        tables/shap_importance.csv
        tables/cart_subgroups.txt
        data/cate_individual.parquet
"""

import pandas as pd, numpy as np, json, os, time, warnings

warnings.filterwarnings("ignore")
os.makedirs("tables", exist_ok=True)
os.makedirs("data", exist_ok=True)

SUBSAMPLE_N = 50_000
RANDOM_STATE = 42


def prepare_features(df):
    """Build numeric feature matrix for causal forest."""
    feats = pd.DataFrame(index=df.index)
    feats["age_ordinal"] = df["_AGE_G"].values
    feats["female"] = df["Female"].values
    feats["race"] = df["_RACEGR3"].values
    feats["education"] = df["_EDUCAG"].values
    feats["income"] = df["INCOME5"].values
    feats["bmi"] = df["_BMI5CAT"].values
    feats["year"] = (df["YEAR"].values - 2019) / 5

    for r in [2, 3, 4, 5]:
        feats[f"race_{r}"] = (df["_RACEGR3"].values == r).astype(int)

    if "_AGE80" in df.columns and df["_AGE80"].notna().sum() > len(df) * 0.5:
        feats["age_cont"] = df["_AGE80"].values
        feats["age_cont"] = feats["age_cont"].fillna(feats["age_cont"].median())
    else:
        age_midpoints = {1: 21, 2: 29.5, 3: 39.5, 4: 49.5, 5: 59.5, 6: 72}
        feats["age_cont"] = df["_AGE_G"].map(age_midpoints).values

    return feats


def main():
    print("=" * 60)
    print("Causal Forest (CausalForestDML) — DCC Optimized")
    print("=" * 60)

    df = pd.read_parquet("data/pooled_cc.parquet")
    print(f"Loaded: {len(df):,} rows")

    if len(df) > SUBSAMPLE_N:
        print(f"Stratified subsampling to {SUBSAMPLE_N:,} …")
        frames = []
        n_per_age = SUBSAMPLE_N // 6
        rng = np.random.RandomState(RANDOM_STATE)
        for age_val in range(1, 7):
            sub = df[df["_AGE_G"] == age_val]
            n_take = min(n_per_age, len(sub))
            idx = rng.choice(len(sub), n_take, replace=False)
            frames.append(sub.iloc[idx])
        df_sub = pd.concat(frames, ignore_index=True)
        print(f"  Subsample: {len(df_sub):,}")
    else:
        df_sub = df.reset_index(drop=True)

    feats = prepare_features(df_sub)
    Y = df_sub["FMD"].values.astype(float)
    T = df_sub["PA"].values.astype(float)
    W = df_sub["WEIGHT"].values

    X_cols = ["age_cont", "female", "education", "income", "bmi", "year"]
    X = feats[X_cols].values

    W_conf_cols = [c for c in feats.columns if c.startswith("race_")]
    W_conf = feats[W_conf_cols].values

    print(f"\nY shape: {Y.shape}, T shape: {T.shape}")
    print(f"X shape (heterogeneity): {X.shape}")
    print(f"W shape (confounders): {W_conf.shape}")

    # ── Fit CausalForestDML ──
    # HistGradientBoosting is 10-100x faster than GradientBoosting
    # and natively supports multi-threading.
    from econml.dml import CausalForestDML
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    print("\nFitting CausalForestDML …")
    print(f"  Nuisance: HistGradientBoosting (150 iter, depth 5)")
    print(f"  Forest:   500 trees, min_leaf=50, cv=2")
    t0 = time.time()

    est = CausalForestDML(
        model_y=HistGradientBoostingRegressor(
            max_iter=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE),
        model_t=HistGradientBoostingClassifier(
            max_iter=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE),
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=50,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        cv=2,
    )

    est.fit(Y=Y, T=T, X=X, W=W_conf, sample_weight=W)
    elapsed = time.time() - t0
    print(f"  Fitted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Extract CATEs ──
    print("\nExtracting CATEs …")
    cate = est.effect(X).flatten()
    cate_lower, cate_upper = est.effect_interval(X, alpha=0.05)
    cate_lower = cate_lower.flatten()
    cate_upper = cate_upper.flatten()

    print(f"CATE summary: mean={cate.mean():.4f}, "
          f"median={np.median(cate):.4f}, std={cate.std():.4f}")

    # ── CATE by age group ──
    age_groups = df_sub["_AGE_G"].values
    age_labels = {1: "18-24", 2: "25-34", 3: "35-44",
                  4: "45-54", 5: "55-64", 6: "65+"}
    cate_by_age = []
    for ag, lab in age_labels.items():
        mask = age_groups == ag
        cate_by_age.append({
            "Age_Group": lab,
            "n": int(mask.sum()),
            "CATE_mean": round(float(cate[mask].mean()), 4),
            "CATE_median": round(float(np.median(cate[mask])), 4),
            "CATE_std": round(float(cate[mask].std()), 4),
            "CI_Lower": round(float(cate_lower[mask].mean()), 4),
            "CI_Upper": round(float(cate_upper[mask].mean()), 4),
        })
        print(f"  {lab}: CATE = {cate[mask].mean():.4f} "
              f"[{cate_lower[mask].mean():.4f}, {cate_upper[mask].mean():.4f}]")

    cate_age_df = pd.DataFrame(cate_by_age)
    cate_age_df.to_csv("tables/cate_by_age.csv", index=False)
    print("Saved tables/cate_by_age.csv")

    # ── ATE (overall) ──
    ate = est.ate(X=X, T0=0, T1=1)
    ate_lower, ate_upper = est.ate_interval(X=X, T0=0, T1=1, alpha=0.05)
    print(f"\nATE = {ate:.4f} [{ate_lower:.4f}, {ate_upper:.4f}]")

    # ── Feature importance ──
    print("\nComputing feature importance …")
    try:
        shap_values = est.shap_values(X)
        shap_importance = np.abs(shap_values["Y0"]).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": X_cols,
            "SHAP_Importance": shap_importance,
        }).sort_values("SHAP_Importance", ascending=False)
        shap_df.to_csv("tables/shap_importance.csv", index=False)
        print(shap_df.to_string(index=False))
    except Exception as e:
        print(f"  SHAP failed ({e}), using forest feature_importances_")
        try:
            fi = est.feature_importances_
            shap_df = pd.DataFrame({
                "Feature": X_cols,
                "Importance": fi,
            }).sort_values("Importance", ascending=False)
            shap_df.to_csv("tables/shap_importance.csv", index=False)
            print(shap_df.to_string(index=False))
        except Exception:
            print("  Feature importance also failed; skipping.")

    # ── CART subgroup discovery ──
    print("\nCART subgroup discovery …")
    from sklearn.tree import DecisionTreeRegressor, export_text

    X_interp = pd.DataFrame({
        "Age_Group": df_sub["_AGE_G"].values,
        "Female": df_sub["Female"].values,
        "Income": df_sub["INCOME5"].values,
        "Education": df_sub["_EDUCAG"].values,
    })

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5000,
                                  random_state=RANDOM_STATE)
    tree.fit(X_interp, cate, sample_weight=W)
    tree_text = export_text(tree, feature_names=list(X_interp.columns),
                            max_depth=3)
    with open("tables/cart_subgroups.txt", "w") as f:
        f.write("CART Subgroup Discovery on Estimated CATEs\n")
        f.write("=" * 50 + "\n")
        f.write(tree_text)
    print(tree_text)

    # ── Save individual CATEs ──
    cate_out = pd.DataFrame({
        "age_cont": feats["age_cont"].values,
        "age_group": df_sub["_AGE_G"].values,
        "female": df_sub["Female"].values,
        "cate": cate,
        "cate_lower": cate_lower,
        "cate_upper": cate_upper,
        "weight": W,
    })
    cate_out.to_parquet("data/cate_individual.parquet", index=False)
    print("Saved data/cate_individual.parquet")

    # ── Summary ──
    summary = {
        "n_subsample": int(len(df_sub)),
        "n_estimators": 500,
        "ate": round(float(ate), 4),
        "ate_ci": f"[{ate_lower:.4f}, {ate_upper:.4f}]",
        "cate_mean": round(float(cate.mean()), 4),
        "cate_std": round(float(cate.std()), 4),
        "elapsed_seconds": round(elapsed, 0),
    }
    with open("tables/cate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
