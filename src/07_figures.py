"""
07_figures.py
Publication-quality figures for the extended PA-FMD analysis.

Reads various tables/ and data/ outputs from prior scripts.
Writes to figures/ directory.
"""

import pandas as pd, numpy as np, os, json, warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

os.makedirs("figures", exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})
COLORS = {"active": "#2196F3", "inactive": "#F44336",
           "main": "#1565C0", "accent": "#E65100",
           "neutral": "#616161", "ci": "#90CAF9"}


def fig_temporal_validation():
    """Year-by-year age-stratified PA ORs (key validation figure)."""
    path = "tables/temporal_age_or.csv"
    if not os.path.exists(path):
        print("  Skipping temporal validation figure (data not found)")
        return

    df = pd.read_csv(path)
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    years = sorted(df["Year"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(years)))

    for i, yr in enumerate(years):
        sub = df[df["Year"] == yr].set_index("Age_Group").reindex(age_groups)
        ax.plot(range(len(age_groups)), sub["OR"], "o-", color=cmap[i],
                markersize=4, linewidth=1.2, label=str(int(yr)))
        ax.fill_between(range(len(age_groups)), sub["CI_L"], sub["CI_U"],
                         alpha=0.06, color=cmap[i])

    ax.axhline(y=1, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xticks(range(len(age_groups)))
    ax.set_xticklabels(age_groups)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("PA Odds Ratio for FMD")
    ax.set_title("Age-Stratified PA Effect Across 2015–2024")
    ax.legend(title="Year", ncol=3, loc="upper right", fontsize=7)
    ax.set_ylim(0.2, 1.4)

    plt.tight_layout()
    plt.savefig("figures/fig_temporal_validation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_temporal_validation.png")


def fig_cate_by_age():
    """CATE curve by continuous age — the visual centerpiece."""
    path = "data/cate_individual.parquet"
    if not os.path.exists(path):
        print("  Skipping CATE figure (data not found)")
        return

    df = pd.read_parquet(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                              gridspec_kw={"width_ratios": [2, 1]})

    # Panel (a): CATE vs continuous age (smoothed)
    ax = axes[0]
    age_bins = np.arange(18, 82, 2)
    bin_means, bin_lower, bin_upper, bin_centers = [], [], [], []
    for i in range(len(age_bins) - 1):
        mask = (df["age_cont"] >= age_bins[i]) & (df["age_cont"] < age_bins[i+1])
        if mask.sum() < 50:
            continue
        w = df.loc[mask, "weight"].values
        c = df.loc[mask, "cate"].values
        cl = df.loc[mask, "cate_lower"].values
        cu = df.loc[mask, "cate_upper"].values
        wm = np.average(c, weights=w)
        bin_means.append(wm)
        bin_lower.append(np.average(cl, weights=w))
        bin_upper.append(np.average(cu, weights=w))
        bin_centers.append((age_bins[i] + age_bins[i+1]) / 2)

    ax.plot(bin_centers, bin_means, color=COLORS["main"], linewidth=2)
    ax.fill_between(bin_centers, bin_lower, bin_upper,
                     alpha=0.2, color=COLORS["ci"])
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("CATE (PA effect on P(FMD))")
    ax.set_title("(a) Estimated Treatment Effect by Age")

    # Annotate the zero-crossing
    for i in range(1, len(bin_means)):
        if bin_means[i-1] >= 0 and bin_means[i] < 0:
            cross_age = bin_centers[i]
            ax.annotate(f"Zero crossing ≈ {cross_age:.0f}",
                        xy=(cross_age, 0), xytext=(cross_age + 5, 0.02),
                        arrowprops=dict(arrowstyle="->", color=COLORS["accent"]),
                        fontsize=8, color=COLORS["accent"])
            break

    # Panel (b): CATE by age group (bar chart with CIs)
    ax2 = axes[1]
    cate_path = "tables/cate_by_age.csv"
    if os.path.exists(cate_path):
        cate_ag = pd.read_csv(cate_path)
        age_labels = cate_ag["Age_Group"].values
        means = cate_ag["CATE_mean"].values
        ci_l = cate_ag["CI_Lower"].values
        ci_u = cate_ag["CI_Upper"].values

        colors = [COLORS["inactive"] if m >= 0 else COLORS["active"] for m in means]
        bars = ax2.barh(range(len(age_labels)), means, color=colors,
                         edgecolor="black", linewidth=0.3, height=0.6)
        ax2.errorbar(means, range(len(age_labels)),
                      xerr=[means - ci_l, ci_u - means],
                      fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
        ax2.set_yticks(range(len(age_labels)))
        ax2.set_yticklabels(age_labels)
        ax2.set_xlabel("CATE (mean)")
        ax2.set_title("(b) CATE by Age Group")
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("figures/fig_cate_by_age.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_cate_by_age.png")


def fig_shap_importance():
    """SHAP feature importance for treatment effect heterogeneity."""
    path = "tables/shap_importance.csv"
    if not os.path.exists(path):
        print("  Skipping SHAP figure (data not found)")
        return

    df = pd.read_csv(path)
    imp_col = "SHAP_Importance" if "SHAP_Importance" in df.columns else "Importance"
    df = df.sort_values(imp_col, ascending=True)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(range(len(df)), df[imp_col].values,
            color=COLORS["main"], edgecolor="black", linewidth=0.3, height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"].values)
    ax.set_xlabel("Mean |SHAP value| for Treatment Effect")
    ax.set_title("Feature Importance for PA Effect Heterogeneity")

    plt.tight_layout()
    plt.savefig("figures/fig_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_shap_importance.png")


def fig_propensity_overlap():
    """Propensity score distributions by age group and PA status."""
    path = "data/propensity_scores.parquet"
    if not os.path.exists(path):
        print("  Skipping propensity overlap figure (data not found)")
        return

    df = pd.read_parquet(path)
    age_labels = {1: "18-24", 2: "25-34", 3: "35-44",
                  4: "45-54", 5: "55-64", 6: "65+"}

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)

    for i, (ag, lab) in enumerate(age_labels.items()):
        ax = axes[i // 3, i % 3]
        sub = df[df["age_group"] == ag]
        ps_active = sub.loc[sub["pa"] == 1, "ps"]
        ps_inactive = sub.loc[sub["pa"] == 0, "ps"]

        ax.hist(ps_active, bins=50, alpha=0.6, color=COLORS["active"],
                label="Active", density=True)
        ax.hist(ps_inactive, bins=50, alpha=0.6, color=COLORS["inactive"],
                label="Inactive", density=True)
        ax.set_title(f"Age {lab}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
        if i >= 3:
            ax.set_xlabel("P(PA=Active | X)")
        if i % 3 == 0:
            ax.set_ylabel("Density")

    plt.suptitle("Propensity Score Overlap by Age Group", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/fig_propensity_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_propensity_overlap.png")


def fig_evalues():
    """E-value visualization."""
    path = "tables/evalues.csv"
    if not os.path.exists(path):
        print("  Skipping E-value figure (data not found)")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel (a): ORs by subgroup
    ax = axes[0]
    df_age = df[df["Subgroup"].str.startswith("Age")].copy()
    df_age["label"] = df_age["Subgroup"].str.replace("Age: ", "")
    ypos = range(len(df_age))
    ax.errorbar(df_age["OR"], ypos,
                xerr=[df_age["OR"] - df_age["CI_Lower"],
                      df_age["CI_Upper"] - df_age["OR"]],
                fmt="s", color=COLORS["accent"], ecolor="#999",
                capsize=3, markersize=5)
    ax.axvline(x=1, color="red", linestyle="--", linewidth=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df_age["label"])
    ax.set_xlabel("PA Odds Ratio")
    ax.set_title("(a) PA Effect by Age (Pooled 2015–2024)")
    ax.invert_yaxis()

    # Panel (b): E-values
    ax2 = axes[1]
    ax2.barh(ypos, df_age["E_value_point"].values,
             color=COLORS["main"], edgecolor="black", linewidth=0.3, height=0.5,
             label="E-value (point)")
    ax2.barh(ypos, df_age["E_value_CI"].values,
             color=COLORS["ci"], edgecolor="black", linewidth=0.3, height=0.3,
             label="E-value (CI bound)")
    ax2.axvline(x=1, color="red", linestyle="--", linewidth=0.7)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(df_age["label"])
    ax2.set_xlabel("E-value")
    ax2.set_title("(b) E-values for Unmeasured Confounding")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("figures/fig_evalues.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_evalues.png")


def fig_stratified_pooled():
    """Updated stratified OR figure with pooled multi-year data."""
    path = "tables/survey_stratified_or.csv"
    if not os.path.exists(path):
        print("  Skipping pooled stratified figure (data not found)")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ypos = range(len(df))
    ax.errorbar(df["OR"], ypos,
                xerr=[df["OR"] - df["CI_L"], df["CI_U"] - df["OR"]],
                fmt="s", color=COLORS["accent"], ecolor="#999",
                capsize=3, markersize=5)
    ax.axvline(x=1, color="red", linestyle="--", linewidth=0.7, alpha=0.5)

    # Overall OR line
    main_path = "tables/survey_main_or.csv"
    if os.path.exists(main_path):
        main_or = pd.read_csv(main_path)
        pa_or = main_or.loc[main_or["Variable"] == "PA", "OR"].values
        if len(pa_or):
            ax.axvline(x=pa_or[0], color=COLORS["main"], linestyle=":",
                       linewidth=1, alpha=0.7,
                       label=f"Overall OR={pa_or[0]:.3f}")

    ax.set_yticks(ypos)
    ax.set_yticklabels(df["Subgroup"], fontsize=7.5)
    ax.set_xlabel("PA Odds Ratio (95% CI)")
    ax.set_title("Stratified PA Effect on FMD (Pooled 2015–2024)")
    ax.legend(fontsize=7, loc="upper right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("figures/fig_stratified_pooled.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved figures/fig_stratified_pooled.png")


def main():
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)

    fig_temporal_validation()
    fig_cate_by_age()
    fig_shap_importance()
    fig_propensity_overlap()
    fig_evalues()
    fig_stratified_pooled()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
