"""
Lavka-Twin DiD v2 — normalized for distribution and price.

Changes vs v1:
  1. Units are normalized to per-active-city ("same-store" rate) before DiD.
     This removes the distribution-expansion bias (12% of cases grew listing by >10%).
  2. Price drop during activation is flagged (17% of cases have >5% price cut)
     and uplift is decomposed into "shelf effect" vs "price effect" using a
     simple log-log elasticity estimated on the pre-period.
  3. Both raw and normalized estimates are kept for diagnostics.

Why per-active-city, not per-all-cities:
  When a SKU appears in NEW cities during activation, that listing expansion
  is not "promo uplift" — it is a different lever. We want uplift WITHIN
  the stores that already carried the SKU. This mirrors the "same-store
  sales" concept in classical retail analytics.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/Users/dmitry/DS_ML_Projects/PepsiCo/Synthetic_AB/Test 2.1/INITIAL_DATA")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

PRE_WEEKS = 8
MIN_PRE_WEEKS = 4
MIN_SALES_FLOOR = 1.0
PRICE_DROP_THRESHOLD = 0.05      # >5% drop = flag "price_promo_overlap"
FALLBACK_ELASTICITY = -1.2        # FMCG beverages default if pre-period too short


# ---------------------------------------------------------------------------
# Data prep with distribution & price
# ---------------------------------------------------------------------------

def load_weekly_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    sam = pd.read_parquet(RAW / "samokat_sales.parquet").rename(columns={"gtin": "barcode"})
    lav = pd.read_parquet(RAW / "lavka_sales.parquet")
    sam = sam[sam["sales_quantity"] > 0]              # drop returns/corrections
    sam["week"] = pd.to_datetime(sam["date"]).dt.to_period("W-SUN").dt.start_time
    lav["week"] = pd.to_datetime(lav["date"]).dt.to_period("W-SUN").dt.start_time
    sam_w = (sam.groupby(["barcode", "week"], as_index=False)
                 .agg(samokat_units=("sales_quantity", "sum"),
                      samokat_n_cities=("city", "nunique"),
                      samokat_price=("price", "mean")))
    lav_w = (lav.groupby(["barcode", "week"], as_index=False)
                 .agg(lavka_units=("sales_quantity", "sum"),
                      lavka_n_outlets=("lavka_id", "nunique")))
    return sam_w, lav_w


def load_activations() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_parquet(RAW / "activations.parquet")
    a["start_date"] = pd.to_datetime(a["start_date"])
    a["end_date"] = pd.to_datetime(a["end_date"])
    a["barcode"] = a["barcode"].astype(str)
    test_ids = pd.read_csv(RAW / "test_placement_ids.csv", header=None)[0].tolist()[1:]
    return a[a["placement_id"].isin(test_ids)].copy(), a


# ---------------------------------------------------------------------------
# Price elasticity (per-SKU, fallback to category default)
# ---------------------------------------------------------------------------

def estimate_elasticity_pre(s_pre: pd.DataFrame) -> float:
    """Simple OLS on log(units_per_city) ~ log(price) on pre-period."""
    df = s_pre[(s_pre["samokat_units"] > 0) & (s_pre["samokat_price"] > 0)
               & (s_pre["samokat_n_cities"] > 0)].copy()
    if len(df) < 4:
        return FALLBACK_ELASTICITY
    df["log_q"] = np.log(df["samokat_units"] / df["samokat_n_cities"])
    df["log_p"] = np.log(df["samokat_price"])
    if df["log_p"].std() < 0.01:           # almost no price variation -> use default
        return FALLBACK_ELASTICITY
    x = df["log_p"].values - df["log_p"].mean()
    y = df["log_q"].values - df["log_q"].mean()
    beta = (x * y).sum() / (x * x).sum()
    return float(np.clip(beta, -3.0, 0.0))   # clip pathological estimates


# ---------------------------------------------------------------------------
# Core estimator v2
# ---------------------------------------------------------------------------

def lavka_twin_did_v2(barcode: str, start: pd.Timestamp, end: pd.Timestamp,
                       sam_w: pd.DataFrame, lav_w: pd.DataFrame,
                       pre_weeks: int = PRE_WEEKS) -> dict:
    s = sam_w[sam_w["barcode"].astype(str) == str(barcode)].set_index("week")
    l = lav_w[lav_w["barcode"].astype(str) == str(barcode)].set_index("week")

    pre_start = start - pd.Timedelta(weeks=pre_weeks)
    s_pre = s.loc[(s.index >= pre_start) & (s.index < start)]
    s_test = s.loc[(s.index >= start) & (s.index <= end)]
    l_pre = l.loc[(l.index >= pre_start) & (l.index < start)]
    l_test = l.loc[(l.index >= start) & (l.index <= end)]

    out: dict = {"n_pre_weeks": len(s_pre), "n_test_weeks": len(s_test)}

    if (len(s_pre) < MIN_PRE_WEEKS or len(s_test) < 1
            or len(l_pre) == 0 or len(l_test) == 0):
        out.update({"valid": False, "reason": "insufficient_history"})
        return out

    # Per-active-city rate ("same-store sales")
    s_pre_pcty = (s_pre["samokat_units"] / s_pre["samokat_n_cities"]).mean()
    s_test_pcty = (s_test["samokat_units"] / s_test["samokat_n_cities"]).mean()
    l_pre_punit = (l_pre["lavka_units"] / l_pre["lavka_n_outlets"]).mean()
    l_test_punit = (l_test["lavka_units"] / l_test["lavka_n_outlets"]).mean()

    if s_pre_pcty < 0.01 or l_pre_punit < 0.001:
        out.update({"valid": False, "reason": "tiny_baseline"})
        return out

    # === DiD on per-city rates ===
    ratio_shift = l_test_punit / l_pre_punit
    s_expected_pcty = s_pre_pcty * ratio_shift
    uplift_per_city_per_week = s_test_pcty - s_expected_pcty
    uplift_pct_norm = uplift_per_city_per_week / s_expected_pcty

    # === Raw DiD (for comparison) ===
    s_pre_tot = s_pre["samokat_units"].mean()
    s_test_tot = s_test["samokat_units"].mean()
    l_pre_tot = l_pre["lavka_units"].mean()
    l_test_tot = l_test["lavka_units"].mean()
    s_expected_tot = s_pre_tot * (l_test_tot / max(l_pre_tot, 1e-6))
    uplift_pct_raw = (s_test_tot - s_expected_tot) / s_expected_tot if s_expected_tot > 0 else np.nan

    # === Distribution & price diagnostics ===
    cities_growth = (s_test["samokat_n_cities"].mean()
                     / max(s_pre["samokat_n_cities"].mean(), 1)) - 1
    price_pre = s_pre["samokat_price"].mean()
    price_test = s_test["samokat_price"].mean()
    price_change = (price_test / price_pre) - 1 if price_pre > 0 else np.nan
    price_promo = bool(price_change < -PRICE_DROP_THRESHOLD)

    # === Price-elasticity decomposition (if discount) ===
    elasticity = estimate_elasticity_pre(s_pre)
    # Expected % change in units due to price alone
    if pd.notna(price_change):
        uplift_from_price = elasticity * price_change
    else:
        uplift_from_price = 0.0
    uplift_pct_shelf_only = uplift_pct_norm - uplift_from_price   # subtract price effect

    # Final uplift in absolute units (using per-city scale × number of active cities in test)
    avg_test_cities = s_test["samokat_n_cities"].mean()
    uplift_units_norm = uplift_per_city_per_week * avg_test_cities * len(s_test)
    uplift_units_shelf = uplift_pct_shelf_only * s_expected_pcty * avg_test_cities * len(s_test)

    out.update({
        "valid": True, "reason": "ok",
        # raw vs normalized
        "uplift_pct_raw": uplift_pct_raw,
        "uplift_pct_norm": uplift_pct_norm,                # distribution-adjusted
        "uplift_pct_shelf_only": uplift_pct_shelf_only,    # also price-adjusted
        # absolute units
        "uplift_units_norm": uplift_units_norm,
        "uplift_units_shelf_only": uplift_units_shelf,
        # diagnostics
        "cities_growth": cities_growth,
        "price_change": price_change,
        "price_promo_flag": price_promo,
        "elasticity": elasticity,
        "uplift_from_price": uplift_from_price,
        # internals
        "s_pre_pcty": s_pre_pcty, "s_test_pcty": s_test_pcty,
        "l_pre_punit": l_pre_punit, "l_test_punit": l_test_punit,
        "ratio_shift": ratio_shift,
    })
    return out


# ---------------------------------------------------------------------------
# Batch & validation
# ---------------------------------------------------------------------------

def run_all(act, sam_w, lav_w):
    rows = []
    for _, r in act.iterrows():
        est = lavka_twin_did_v2(r["barcode"], r["start_date"], r["end_date"], sam_w, lav_w)
        est.update({"placement_id": r["placement_id"], "barcode": r["barcode"],
                    "tool_name": r["tool_name"], "sku_category": r["sku_category"],
                    "start_date": r["start_date"], "end_date": r["end_date"]})
        rows.append(est)
    return pd.DataFrame(rows)


def compare_with_consensus(did):
    consensus = pd.read_csv(ROOT / "results" / "unified_per_case.csv")
    did = did.copy(); did["barcode"] = did["barcode"].astype(str)
    consensus["barcode"] = consensus["barcode"].astype(str)
    m = did[did["valid"]].merge(
        consensus[["placement_id", "barcode", "tool_name", "scm_rel", "hbm_rel"]],
        on=["placement_id", "barcode", "tool_name"], how="inner")
    return m


def main():
    print("[1/4] Loading panels with distribution & price...")
    sam_w, lav_w = load_weekly_panels()
    test_act, _ = load_activations()
    print(f"   Samokat: {len(sam_w)} sku-weeks | Lavka: {len(lav_w)} sku-weeks")
    print(f"   Test activations: {len(test_act)}")

    print("[2/4] Running Lavka-Twin DiD v2 (distribution-normalized + price-controlled)...")
    did = run_all(test_act, sam_w, lav_w)
    did.to_csv(OUT / "lavka_did_v2_per_case.csv", index=False)
    valid = did[did["valid"]]
    print(f"   Valid: {len(valid)}/{len(did)} ({100*len(valid)/len(did):.1f}%)")
    print(f"   Median uplift RAW       : {valid['uplift_pct_raw'].median():.4f}")
    print(f"   Median uplift NORM (dist): {valid['uplift_pct_norm'].median():.4f}")
    print(f"   Median uplift SHELF (dist+price): {valid['uplift_pct_shelf_only'].median():.4f}")
    print(f"   Cases with price_promo flag: {valid['price_promo_flag'].sum()} ({valid['price_promo_flag'].mean():.1%})")
    print(f"   Cases with cities_growth >10%: {(valid['cities_growth']>0.10).sum()}")
    print(f"   Median elasticity: {valid['elasticity'].median():.3f}")

    print("\n[3/4] Agreement with SCM+HBM consensus...")
    comp = compare_with_consensus(did)
    for col in ["uplift_pct_raw", "uplift_pct_norm", "uplift_pct_shelf_only"]:
        c_scm = comp[[col, "scm_rel"]].corr().iloc[0, 1]
        c_hbm = comp[[col, "hbm_rel"]].corr().iloc[0, 1]
        sa_scm = ((comp[col] > 0) == (comp["scm_rel"] > 0)).mean()
        print(f"   {col:30s}  corr_SCM={c_scm:+.3f}  corr_HBM={c_hbm:+.3f}  sign_agree_SCM={sa_scm:.1%}")
    comp.to_csv(OUT / "lavka_did_v2_vs_consensus.csv", index=False)

    print("\n[4/4] Tool-level summary (3 versions side by side)...")
    summary = (valid.groupby("tool_name")
               .agg(n=("uplift_pct_raw", "size"),
                    raw_med=("uplift_pct_raw", "median"),
                    norm_med=("uplift_pct_norm", "median"),
                    shelf_med=("uplift_pct_shelf_only", "median"),
                    cities_growth_med=("cities_growth", "median"),
                    price_promo_share=("price_promo_flag", "mean"))
               .sort_values("shelf_med", ascending=False))
    summary.to_csv(OUT / "lavka_did_v2_tool_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
