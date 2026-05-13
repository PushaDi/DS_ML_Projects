"""
Meta-analysis: consolidate uplift estimates from 4 methods and select the best.

Methods compared:
  SCM   - Synthetic Control (Abadie-style, ridge-augmented)
  GSC   - Generalized Synthetic Control + Interactive Fixed Effects (Xu 2017)
  MC    - Matrix Completion (Athey et al. 2021)
  HBM   - Hierarchical Bayesian Model on log(sales), pooled by tool x category

Outputs:
  results/unified_per_case.csv      - one row per (placement_id, tool_name) with all 4 estimates
  results/unified_tool_summary.csv  - tool-level rollup with consensus uplift
  results/method_agreement.csv      - pairwise agreement diagnostics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_per_case() -> pd.DataFrame:
    full = pd.read_csv(DATA / "per_case_full_results_v2.csv")
    gsc = pd.read_csv(DATA / "gsc_per_case.csv")[
        ["placement_id", "barcode", "tool_name", "relative_uplift", "uplift_units", "single_tool_case"]
    ].rename(columns={
        "relative_uplift": "gsc_relative_uplift",
        "uplift_units": "gsc_uplift_units",
        "single_tool_case": "gsc_single_tool",
    })
    mc = pd.read_csv(DATA / "mc_per_case.csv")[
        ["placement_id", "barcode", "tool_name", "relative_uplift", "uplift_units"]
    ].rename(columns={
        "relative_uplift": "mc_relative_uplift",
        "uplift_units": "mc_uplift_units",
    })
    df = full.merge(gsc, on=["placement_id", "barcode", "tool_name"], how="left")
    df = df.merge(mc, on=["placement_id", "barcode", "tool_name"], how="left")
    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only good-quality SCM cases."""
    return df[df["quality_flag"] == "Хорошее"].copy()


def per_case_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Combine SCM/GSC/MC point estimates with HBM posterior.

    Strategy:
      1. Use HBM posterior median as the main estimate (it is pooled & uncertainty-aware).
      2. Sanity-check vs SCM and GSC: flag disagreement when sign differs.
      3. Confidence = HBM prob_positive (>0.85 high, 0.65-0.85 med, else low).
    """
    df = df.copy()
    df["scm_rel"] = df["raw_relative_uplift"]
    df["hbm_rel"] = df["hbm_v2_posterior_median"]
    df["gsc_rel"] = df["gsc_relative_uplift"]
    df["mc_rel"] = df["mc_relative_uplift"]

    estimates = df[["scm_rel", "hbm_rel", "gsc_rel", "mc_rel"]]
    df["n_methods_available"] = estimates.notna().sum(axis=1)
    df["median_across_methods"] = estimates.median(axis=1)
    df["mean_across_methods"] = estimates.mean(axis=1)
    df["std_across_methods"] = estimates.std(axis=1)

    df["sign_agreement_count"] = (
        (estimates > 0).sum(axis=1) - (estimates < 0).sum(axis=1)
    ).abs()  # 4 = full agreement, 0 = 2v2 split

    def _confidence(row):
        p = row.get("hbm_v2_prob_positive")
        if pd.isna(p):
            return "no_hbm"
        if p > 0.90 or p < 0.10:
            return "high"
        if p > 0.75 or p < 0.25:
            return "medium"
        return "low"

    df["consensus_confidence"] = df.apply(_confidence, axis=1)

    cols = [
        "placement_id", "barcode", "tool_name", "sku_category", "route", "quality_flag",
        "scm_rel", "hbm_rel", "gsc_rel", "mc_rel",
        "median_across_methods", "mean_across_methods", "std_across_methods",
        "sign_agreement_count", "n_methods_available",
        "hbm_v2_prob_positive", "hbm_v2_hdi90_low", "hbm_v2_hdi90_high",
        "consensus_confidence",
        "attributed_cum_uplift", "attributed_relative_uplift", "tool_share",
        "n_concurrent_tools", "sole_tool_in_window", "placebo_pvalue",
    ]
    return df[cols]


def tool_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per case to tool level."""
    rows = []
    for tool, g in per_case.groupby("tool_name"):
        rows.append({
            "tool_name": tool,
            "n_cases": len(g),
            "n_sole_tool": int((g["sole_tool_in_window"] == True).sum()),
            "scm_median": g["scm_rel"].median(),
            "hbm_median": g["hbm_rel"].median(),
            "gsc_median": g["gsc_rel"].median(),
            "mc_median": g["mc_rel"].median(),
            "consensus_median": g[["scm_rel", "hbm_rel"]].median(axis=1).median(),  # SCM+HBM only (most reliable)
            "sign_agreement_share": (g["sign_agreement_count"] >= 3).mean(),
            "high_conf_share": (g["consensus_confidence"] == "high").mean(),
            "sum_attributed_units": g["attributed_cum_uplift"].sum(),
        })
    out = pd.DataFrame(rows).sort_values("consensus_median", ascending=False)
    return out


def method_agreement(per_case: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation / sign-agreement between methods."""
    methods = ["scm_rel", "hbm_rel", "gsc_rel", "mc_rel"]
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            sub = per_case[[m1, m2]].dropna()
            if len(sub) < 5:
                continue
            corr = sub[m1].corr(sub[m2])
            sign_agree = ((sub[m1] > 0) == (sub[m2] > 0)).mean()
            rows.append({
                "method_1": m1, "method_2": m2, "n": len(sub),
                "pearson_corr": corr, "sign_agreement": sign_agree,
            })
    return pd.DataFrame(rows)


def main():
    print("Loading per-case data from 4 methods...")
    raw = load_per_case()
    good = filter_quality(raw)
    print(f"Total cases: {len(raw)} | Good quality (passes SCM filter): {len(good)}")
    print(f"  with GSC estimate:  {good['gsc_relative_uplift'].notna().sum()}")
    print(f"  with MC estimate:   {good['mc_relative_uplift'].notna().sum()}")
    print(f"  with HBM posterior: {good['hbm_v2_posterior_median'].notna().sum()}")

    per_case = per_case_consensus(good)
    per_case.to_csv(OUT / "unified_per_case.csv", index=False)

    tool_sum = tool_summary(per_case)
    tool_sum.to_csv(OUT / "unified_tool_summary.csv", index=False)

    agree = method_agreement(per_case)
    agree.to_csv(OUT / "method_agreement.csv", index=False)

    print("\n=== Tool-level consensus (SCM+HBM median) ===")
    print(tool_sum.to_string(index=False))
    print("\n=== Method pairwise agreement ===")
    print(agree.to_string(index=False))


if __name__ == "__main__":
    main()
