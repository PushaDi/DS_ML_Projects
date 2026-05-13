"""
Lavka-Twin DiD: a simple, business-explainable uplift estimator.

Idea
----
For each (SKU, tool, activation period), use the SAME SKU in Lavka as a
counterfactual (no promotion runs in Lavka in this dataset). A multiplicative
diff-in-diff between Samokat and Lavka on a short window before vs during
activation gives the lift.

Formula
-------
S_pre, L_pre = mean weekly Samokat/Lavka sales in pre-window
S_test, L_test = mean weekly Samokat/Lavka sales in activation window
S_expected = S_pre * (L_test / L_pre)          # counterfactual
uplift_pct = (S_test - S_expected) / S_expected
uplift_units = (S_test - S_expected) * n_weeks_test

Outputs
-------
results/lavka_did_per_case.csv
results/lavka_did_validation.csv   (placebo + sole-tool agreement)
results/lavka_did_tool_summary.csv
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/Users/dmitry/DS_ML_Projects/PepsiCo/Synthetic_AB/Test 2.1/INITIAL_DATA")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

PRE_WEEKS = 8        # pre-period length
MIN_PRE_WEEKS = 4    # need at least this many non-null pre-weeks
MIN_SALES_FLOOR = 1.0  # avoid division by tiny numbers


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_weekly_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Samokat & Lavka daily/city data to (sku, iso_week) weekly panels."""
    sam = pd.read_parquet(RAW / "samokat_sales.parquet").rename(columns={"gtin": "barcode"})
    lav = pd.read_parquet(RAW / "lavka_sales.parquet")

    # to weekly (Monday-anchored)
    sam["week"] = pd.to_datetime(sam["date"]).dt.to_period("W-SUN").dt.start_time
    lav["week"] = pd.to_datetime(lav["date"]).dt.to_period("W-SUN").dt.start_time

    sam_w = (sam.groupby(["barcode", "week"], as_index=False)["sales_quantity"]
                 .sum().rename(columns={"sales_quantity": "samokat_units"}))
    lav_w = (lav.groupby(["barcode", "week"], as_index=False)["sales_quantity"]
                 .sum().rename(columns={"sales_quantity": "lavka_units"}))
    return sam_w, lav_w


def load_activations(only_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (test_activations, ALL_activations).
    test = the 144 placements we want to estimate; ALL = full overlap context for placebo filtering."""
    a = pd.read_parquet(RAW / "activations.parquet")
    a["start_date"] = pd.to_datetime(a["start_date"])
    a["end_date"] = pd.to_datetime(a["end_date"])
    a["barcode"] = a["barcode"].astype(str)
    test_ids = pd.read_csv(RAW / "test_placement_ids.csv", header=None)[0].tolist()[1:]  # skip header "0"
    test_a = a[a["placement_id"].isin(test_ids)].copy()
    return test_a, a


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def lavka_twin_did(
    barcode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sam_w: pd.DataFrame,
    lav_w: pd.DataFrame,
    pre_weeks: int = PRE_WEEKS,
) -> dict:
    """Single (SKU, period) DiD estimate."""
    s = sam_w[sam_w["barcode"].astype(str) == str(barcode)].set_index("week")["samokat_units"]
    l = lav_w[lav_w["barcode"].astype(str) == str(barcode)].set_index("week")["lavka_units"]

    pre_start = start - pd.Timedelta(weeks=pre_weeks)
    pre_mask = lambda idx: (idx >= pre_start) & (idx < start)
    test_mask = lambda idx: (idx >= start) & (idx <= end)

    s_pre = s[pre_mask(s.index)]
    s_test = s[test_mask(s.index)]
    l_pre = l[pre_mask(l.index)]
    l_test = l[test_mask(l.index)]

    out = {
        "n_pre_weeks": len(s_pre),
        "n_test_weeks": len(s_test),
        "s_pre_mean": s_pre.mean() if len(s_pre) else np.nan,
        "l_pre_mean": l_pre.mean() if len(l_pre) else np.nan,
        "s_test_mean": s_test.mean() if len(s_test) else np.nan,
        "l_test_mean": l_test.mean() if len(l_test) else np.nan,
    }
    if (len(s_pre) < MIN_PRE_WEEKS or len(s_test) < 1
            or pd.isna(out["l_pre_mean"]) or out["l_pre_mean"] < MIN_SALES_FLOOR
            or out["s_pre_mean"] < MIN_SALES_FLOOR):
        out.update({"s_expected": np.nan, "uplift_units": np.nan,
                    "uplift_pct": np.nan, "valid": False})
        return out

    ratio_shift = out["l_test_mean"] / out["l_pre_mean"]
    s_expected = out["s_pre_mean"] * ratio_shift
    uplift_per_week = out["s_test_mean"] - s_expected
    out.update({
        "ratio_shift": ratio_shift,
        "s_expected": s_expected,
        "uplift_units": uplift_per_week * out["n_test_weeks"],
        "uplift_pct": uplift_per_week / s_expected if s_expected > 0 else np.nan,
        "valid": True,
    })
    return out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all(act: pd.DataFrame, sam_w: pd.DataFrame, lav_w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in act.iterrows():
        est = lavka_twin_did(r["barcode"], r["start_date"], r["end_date"], sam_w, lav_w)
        est.update({
            "placement_id": r["placement_id"],
            "barcode": r["barcode"],
            "tool_name": r["tool_name"],
            "sku_category": r["sku_category"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
        })
        rows.append(est)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation 1: placebo on pre-period
# ---------------------------------------------------------------------------

def placebo_test(test_act: pd.DataFrame, all_act: pd.DataFrame,
                 sam_w: pd.DataFrame, lav_w: pd.DataFrame,
                 max_offset_weeks: int = 30) -> pd.DataFrame:
    """For each test placement, find the EARLIEST safe pre-window of equal length
    where the same SKU had NO activations at all, and run DiD there.
    A non-zero uplift on such a clean window = false positive."""
    # SKU -> list of (start, end) of any activation
    sku_act = (all_act.groupby("barcode")
                      .apply(lambda g: list(zip(g["start_date"], g["end_date"])))
                      .to_dict())
    placebo_rows = []
    for _, r in test_act.iterrows():
        bc = str(r["barcode"])
        n_weeks = max(1, int((r["end_date"] - r["start_date"]).days / 7) + 1)
        # Try shifts 8, 12, 16, 20, 24, 30 weeks back
        for off in range(8, max_offset_weeks + 1, 4):
            cand_start = r["start_date"] - pd.Timedelta(weeks=off)
            cand_end = cand_start + pd.Timedelta(weeks=n_weeks)
            pre_start = cand_start - pd.Timedelta(weeks=PRE_WEEKS)
            # Overlap with any real activation for this SKU?
            other_acts = sku_act.get(bc, [])
            overlap = any(not (a_end < pre_start or a_start > cand_end)
                          for (a_start, a_end) in other_acts)
            if not overlap:
                rr = r.copy()
                rr["start_date"] = cand_start
                rr["end_date"] = cand_end
                placebo_rows.append(rr)
                break
    if not placebo_rows:
        return pd.DataFrame()
    placebo_df = pd.DataFrame(placebo_rows)
    return run_all(placebo_df, sam_w, lav_w)


# ---------------------------------------------------------------------------
# Validation 2: agreement with SCM+HBM consensus
# ---------------------------------------------------------------------------

def compare_with_consensus(did: pd.DataFrame) -> pd.DataFrame:
    consensus = pd.read_csv(ROOT / "results" / "unified_per_case.csv")
    did = did.copy(); did["barcode"] = did["barcode"].astype(str)
    consensus["barcode"] = consensus["barcode"].astype(str)
    m = did[["placement_id", "barcode", "tool_name", "uplift_pct", "valid"]].merge(
        consensus[["placement_id", "barcode", "tool_name", "scm_rel", "hbm_rel", "median_across_methods"]],
        on=["placement_id", "barcode", "tool_name"], how="inner",
    )
    m = m[m["valid"]].copy()
    return m


# ---------------------------------------------------------------------------
# Attribution: simple 2-step EM
# ---------------------------------------------------------------------------

def attribute_uplift(did: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """Attribute total uplift across overlapping tools using sole-tool typical effects."""
    # Step 1: build sku-week activation map
    rows = []
    for _, r in act.iterrows():
        weeks = pd.date_range(r["start_date"], r["end_date"], freq="W-MON")
        for w in weeks:
            rows.append({"barcode": r["barcode"], "week": w, "tool_name": r["tool_name"],
                         "placement_id": r["placement_id"]})
    sku_week = pd.DataFrame(rows)

    # Mark sole-tool placements
    counts = sku_week.groupby(["barcode", "week"])["tool_name"].nunique().reset_index(name="n_tools")
    sku_week = sku_week.merge(counts, on=["barcode", "week"])
    placement_min_overlap = sku_week.groupby("placement_id")["n_tools"].min().reset_index(name="min_overlap")

    did2 = did.merge(placement_min_overlap, on="placement_id", how="left")
    did2["sole_tool"] = did2["min_overlap"] == 1

    # Step 2: base effects from sole-tool cases
    sole = did2[did2["valid"] & did2["sole_tool"]]
    base_effect = sole.groupby("tool_name")["uplift_pct"].median().to_dict()

    # Step 3: for multi-tool cases, get concurrent tools per placement and split proportionally
    placement_tools = sku_week.groupby("placement_id")["tool_name"].unique().to_dict()

    def _attribute(row):
        if not row["valid"]:
            return np.nan
        if row["sole_tool"]:
            return row["uplift_pct"]
        concurrent = placement_tools.get(row["placement_id"], [row["tool_name"]])
        weights = np.array([max(base_effect.get(t, 0.0), 0.0) for t in concurrent])
        if weights.sum() == 0:
            share = 1.0 / len(concurrent)
        else:
            idx = list(concurrent).index(row["tool_name"])
            share = weights[idx] / weights.sum()
        return row["uplift_pct"] * share

    did2["attributed_pct"] = did2.apply(_attribute, axis=1)
    did2["attributed_units"] = did2["attributed_pct"] * did2["s_expected"] * did2["n_test_weeks"]
    return did2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/5] Loading & weekly aggregation...")
    sam_w, lav_w = load_weekly_panels()
    test_act, all_act = load_activations()
    print(f"   Samokat weekly rows: {len(sam_w)} | Lavka weekly rows: {len(lav_w)}")
    print(f"   TEST activations: {len(test_act)} | total context activations: {len(all_act)}")

    print("[2/5] Running Lavka-Twin DiD on TEST activations...")
    did = run_all(test_act, sam_w, lav_w)
    did.to_csv(OUT / "lavka_did_per_case.csv", index=False)
    valid = did[did["valid"]]
    print(f"   Valid cases: {len(valid)}/{len(did)} ({100*len(valid)/len(did):.1f}%)")
    print(f"   Median uplift_pct (all valid): {valid['uplift_pct'].median():.4f}")

    print("[3/5] Placebo on CLEAN pre-period (no activations in window)...")
    placebo = placebo_test(test_act, all_act, sam_w, lav_w)
    pl_valid = placebo[placebo["valid"]]
    fpr = (pl_valid["uplift_pct"].abs() > 0.10).mean()
    placebo_median = pl_valid["uplift_pct"].median()
    print(f"   Placebo valid: {len(pl_valid)}")
    print(f"   Placebo median uplift (should be ≈ 0): {placebo_median:.4f}")
    print(f"   Placebo false-positive rate (|u|>10%): {fpr:.2%}")

    print("[4/5] Agreement with SCM+HBM consensus...")
    try:
        comp = compare_with_consensus(did)
        corr_scm = comp[["uplift_pct", "scm_rel"]].corr().iloc[0, 1]
        corr_hbm = comp[["uplift_pct", "hbm_rel"]].corr().iloc[0, 1]
        sign_scm = ((comp["uplift_pct"] > 0) == (comp["scm_rel"] > 0)).mean()
        sign_hbm = ((comp["uplift_pct"] > 0) == (comp["hbm_rel"] > 0)).mean()
        print(f"   n compared: {len(comp)}")
        print(f"   corr(DiD, SCM):  {corr_scm:.3f}   sign agree: {sign_scm:.2%}")
        print(f"   corr(DiD, HBM):  {corr_hbm:.3f}   sign agree: {sign_hbm:.2%}")
        comp.to_csv(OUT / "lavka_did_vs_consensus.csv", index=False)
    except Exception as e:
        print("   skipped:", e)

    print("[5/5] Attribution across overlapping tools...")
    attributed = attribute_uplift(did, all_act)
    attributed.to_csv(OUT / "lavka_did_attributed.csv", index=False)
    summary = (attributed[attributed["valid"]]
               .groupby("tool_name")
               .agg(n_cases=("uplift_pct", "size"),
                    n_sole=("sole_tool", "sum"),
                    median_raw_pct=("uplift_pct", "median"),
                    median_attr_pct=("attributed_pct", "median"),
                    sum_attr_units=("attributed_units", "sum"))
               .sort_values("median_attr_pct", ascending=False))
    summary.to_csv(OUT / "lavka_did_tool_summary.csv")
    print("\n=== Tool-level summary (Lavka-Twin DiD) ===")
    print(summary.to_string())

    # Save validation block
    val_rows = [
        {"check": "n_valid_cases", "value": len(valid)},
        {"check": "median_real_uplift_all", "value": float(valid["uplift_pct"].median())},
        {"check": "placebo_median_uplift", "value": float(placebo_median)},
        {"check": "placebo_fpr_at_10pct", "value": float(fpr)},
    ]
    pd.DataFrame(val_rows).to_csv(OUT / "lavka_did_validation.csv", index=False)


if __name__ == "__main__":
    main()
