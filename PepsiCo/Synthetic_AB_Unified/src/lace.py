"""
LACE — Lavka-Anchored Causal Estimator.

Algorithm: see reports/ALGORITHM.md.

Inputs (from INITIAL_DATA/):
  activations.parquet     — placement_id, barcode, tool_name, start_date, end_date, sku_category
  samokat_sales.parquet   — city, gtin, date, sales_quantity
  lavka_sales.parquet     — lavka_id, barcode, date, sales_quantity, category

Outputs (results/):
  lace_per_case.csv       — one row per activation case with uplift + gates + confidence
  lace_tool_summary.csv   — tool×category rollup with bootstrap 90% CI
  lace_vs_scm_hbm.csv     — cross-validation against SCM and HBM (external evidence)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/Users/dmitry/DS_ML_Projects/PepsiCo/Synthetic_AB/Test 2.1/INITIAL_DATA")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Hyperparameters
PRE_WEEKS = 8
PLACEBO_WEEKS = 8           # in-time placebo window length
PLACEBO_GAP = 1             # gap weeks between placebo and pre
N_PERMUTATIONS = 50         # in-space placebo SKUs
PARALLEL_CORR_MIN = 0.3
PLACEBO_TOL = 0.15          # |placebo rel uplift| threshold
RNG = np.random.default_rng(42)


@dataclass
class CaseResult:
    placement_id: str
    barcode: str
    tool_name: str
    sku_category: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_test_weeks: int
    actual_units: float
    cf_units: float
    uplift_units: float
    relative_uplift: float
    parallel_corr: float
    placebo_uplift_pct: float
    permutation_pvalue: float
    gate1_parallel: bool
    gate2_placebo: bool
    gate3_permutation: bool
    confidence: str
    reason: str


def load_data():
    act = pd.read_parquet(RAW / "activations.parquet")
    sam = pd.read_parquet(RAW / "samokat_sales.parquet")
    lav = pd.read_parquet(RAW / "lavka_sales.parquet")
    # normalise barcode dtypes
    act["barcode"] = act["barcode"].astype(str)
    sam["gtin"] = sam["gtin"].astype(str)
    lav["barcode"] = lav["barcode"].astype(str)
    # weekly aggregation
    sam["date"] = pd.to_datetime(sam["date"])
    lav["date"] = pd.to_datetime(lav["date"])
    # Snap to Monday explicitly (avoid pandas W-MON period ambiguity)
    sam["week"] = (sam["date"] - pd.to_timedelta(sam["date"].dt.weekday, unit="D")).dt.normalize()
    lav["week"] = (lav["date"] - pd.to_timedelta(lav["date"].dt.weekday, unit="D")).dt.normalize()
    sam_w = sam.groupby(["gtin", "week"], as_index=False)["sales_quantity"].sum().rename(
        columns={"gtin": "barcode", "sales_quantity": "S"}
    )
    lav_w = lav.groupby(["barcode", "week"], as_index=False)["sales_quantity"].sum().rename(
        columns={"sales_quantity": "L"}
    )
    act["start_date"] = pd.to_datetime(act["start_date"])
    act["end_date"] = pd.to_datetime(act["end_date"])
    return act, sam_w, lav_w


def get_series(df: pd.DataFrame, barcode: str, col: str, week_index: pd.DatetimeIndex) -> pd.Series:
    sub = df[df["barcode"] == barcode][["week", col]].set_index("week")[col]
    return sub.reindex(week_index, fill_value=0.0)


def estimate(
    barcode: str, start: pd.Timestamp, end: pd.Timestamp,
    sam_w: pd.DataFrame, lav_w: pd.DataFrame,
    pre_weeks: int = PRE_WEEKS,
) -> dict | None:
    """Core LACE estimator. Returns dict with uplift + diagnostics, or None if not feasible."""
    start = start.normalize() - pd.Timedelta(days=start.weekday())  # snap to Monday
    end = end.normalize() - pd.Timedelta(days=end.weekday())
    if end < start:
        end = start
    pre_start = start - pd.Timedelta(weeks=pre_weeks)
    weeks = pd.date_range(pre_start, end, freq="7D")  # all Mondays (pre_start is already Monday)
    if len(weeks) < pre_weeks + 1:
        return None
    pre_idx = weeks[weeks < start]
    test_idx = weeks[weeks >= start]
    S = get_series(sam_w, barcode, "S", weeks)
    L = get_series(lav_w, barcode, "L", weeks)
    if S.loc[pre_idx].sum() == 0 or L.loc[pre_idx].sum() == 0:
        return None  # no overlap
    mu_S_pre = S.loc[pre_idx].mean()
    mu_L_pre = L.loc[pre_idx].mean()
    if mu_L_pre <= 0 or mu_S_pre <= 0:
        return None
    S_cf = mu_S_pre * (L.loc[test_idx] / mu_L_pre)
    actual = S.loc[test_idx].sum()
    cf = S_cf.sum()
    uplift_units = actual - cf
    rel_uplift = uplift_units / cf if cf > 0 else np.nan
    # gate 1: parallel trends
    if S.loc[pre_idx].std() == 0 or L.loc[pre_idx].std() == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(S.loc[pre_idx], L.loc[pre_idx])[0, 1])
    return {
        "actual": actual, "cf": cf, "uplift_units": uplift_units, "rel_uplift": rel_uplift,
        "parallel_corr": corr, "n_test_weeks": len(test_idx),
        "pre_idx": pre_idx, "test_idx": test_idx,
    }


def placebo_in_time(barcode, start, sam_w, lav_w, n_weeks):
    """Run LACE on a window BEFORE the real activation, treating it as 'test'."""
    placebo_end = start - pd.Timedelta(weeks=PLACEBO_GAP + 1)
    placebo_start = placebo_end - pd.Timedelta(weeks=n_weeks - 1)
    return estimate(barcode, placebo_start, placebo_end, sam_w, lav_w)


def permutation_pvalue(observed_rel_uplift, peers, start, end, sam_w, lav_w):
    """Run LACE on peer SKUs (same category, not activated in window) and compute p-value."""
    peer_uplifts = []
    for peer in peers:
        r = estimate(peer, start, end, sam_w, lav_w)
        if r is not None and np.isfinite(r["rel_uplift"]):
            peer_uplifts.append(r["rel_uplift"])
    if len(peer_uplifts) < 10:
        return np.nan, len(peer_uplifts)
    peer_arr = np.array(peer_uplifts)
    # one-sided test
    if observed_rel_uplift >= 0:
        p = (peer_arr >= observed_rel_uplift).mean()
    else:
        p = (peer_arr <= observed_rel_uplift).mean()
    return float(p), len(peer_uplifts)


def classify(gate1, gate2, gate3) -> str:
    n = sum([gate1, gate2, gate3])
    if n == 3:
        return "high"
    if n == 2:
        return "medium"
    return "low"


def run_case(row, sam_w, lav_w, peers_by_cat, activated_in_window) -> CaseResult | None:
    barcode = row["barcode"]
    start, end = row["start_date"], row["end_date"]
    res = estimate(barcode, start, end, sam_w, lav_w)
    if res is None:
        return None
    # Gate 1
    gate1 = bool(np.isfinite(res["parallel_corr"]) and res["parallel_corr"] >= PARALLEL_CORR_MIN)
    # Gate 2: in-time placebo
    pres = placebo_in_time(barcode, start, sam_w, lav_w, PLACEBO_WEEKS)
    placebo_pct = pres["rel_uplift"] if pres is not None else np.nan
    gate2 = bool(np.isfinite(placebo_pct) and abs(placebo_pct) <= PLACEBO_TOL)
    # Gate 3: permutation p-value
    cat = row["sku_category"]
    peers_all = peers_by_cat.get(cat, [])
    forbidden = activated_in_window.get((start, end), set()) | {barcode}
    available = [p for p in peers_all if p not in forbidden]
    if len(available) > N_PERMUTATIONS:
        peers = list(RNG.choice(available, N_PERMUTATIONS, replace=False))
    else:
        peers = available
    pval, n_peers = permutation_pvalue(res["rel_uplift"], peers, start, end, sam_w, lav_w)
    gate3 = bool(np.isfinite(pval) and pval < 0.20)
    confidence = classify(gate1, gate2, gate3)
    if confidence == "high" and (np.isnan(pval) or pval >= 0.10):
        confidence = "medium"
    return CaseResult(
        placement_id=row["placement_id"], barcode=barcode, tool_name=row["tool_name"],
        sku_category=cat, start_date=start, end_date=end,
        n_test_weeks=res["n_test_weeks"],
        actual_units=res["actual"], cf_units=res["cf"],
        uplift_units=res["uplift_units"], relative_uplift=res["rel_uplift"],
        parallel_corr=res["parallel_corr"],
        placebo_uplift_pct=placebo_pct, permutation_pvalue=pval,
        gate1_parallel=gate1, gate2_placebo=gate2, gate3_permutation=gate3,
        confidence=confidence,
        reason=f"n_peers={n_peers}",
    )


def build_peer_index(act: pd.DataFrame, sam_w: pd.DataFrame, lav_w: pd.DataFrame):
    """For each category, list barcodes that appear in BOTH Samokat and Lavka."""
    sam_codes = set(sam_w["barcode"].unique())
    lav_codes = set(lav_w["barcode"].unique())
    both = sam_codes & lav_codes
    peers_by_cat: dict[str, list[str]] = {}
    for cat, g in act[["sku_category", "barcode"]].dropna().drop_duplicates().groupby("sku_category"):
        peers_by_cat[cat] = [b for b in g["barcode"].unique() if b in both]
    activated_in_window: dict[tuple, set] = {}
    for _, r in act.iterrows():
        key = (r["start_date"].normalize() - pd.Timedelta(days=r["start_date"].weekday()),
               r["end_date"].normalize() - pd.Timedelta(days=r["end_date"].weekday()))
        activated_in_window.setdefault(key, set()).add(r["barcode"])
    return peers_by_cat, activated_in_window


def bootstrap_ci(arr, alpha=0.10, n_boot=1000):
    if len(arr) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(7)
    samples = [np.median(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def tool_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tool, cat), g in per_case.groupby(["tool_name", "sku_category"]):
        good = g[g["confidence"].isin(["high", "medium"])]
        rows.append({
            "tool_name": tool, "sku_category": cat,
            "n_cases_total": len(g), "n_cases_used": len(good),
            "median_uplift_pct_all": g["relative_uplift"].median(),
            "median_uplift_pct_good": good["relative_uplift"].median() if len(good) else np.nan,
            "ci_low": bootstrap_ci(good["relative_uplift"].dropna().values)[0],
            "ci_high": bootstrap_ci(good["relative_uplift"].dropna().values)[1],
            "n_high_conf": int((g["confidence"] == "high").sum()),
        })
    df = pd.DataFrame(rows).sort_values(["tool_name", "sku_category"])
    return df


def main():
    print("Loading data...")
    act, sam_w, lav_w = load_data()
    peers_by_cat, activated_in_window = build_peer_index(act, sam_w, lav_w)
    print(f"Activations: {len(act)} | Samokat-Lavka SKU overlap: {len(set(sam_w['barcode']) & set(lav_w['barcode']))}")

    cases = act.dropna(subset=["start_date", "end_date", "barcode", "tool_name"]).copy()
    cases = cases.drop_duplicates(subset=["placement_id", "barcode", "tool_name"])

    results = []
    skipped = 0
    for i, (_, row) in enumerate(cases.iterrows()):
        try:
            r = run_case(row, sam_w, lav_w, peers_by_cat, activated_in_window)
        except Exception as e:
            r = None
            print(f"  case {i} error: {e}")
        if r is None:
            skipped += 1
        else:
            results.append(r.__dict__)
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(cases)} cases ({skipped} skipped)")
    print(f"Done. {len(results)} usable cases, {skipped} skipped (no Lavka data / no overlap).")

    per_case = pd.DataFrame(results)
    per_case.to_csv(OUT / "lace_per_case.csv", index=False)

    summary = tool_summary(per_case)
    summary.to_csv(OUT / "lace_tool_summary.csv", index=False)

    # Tool-only (sum over categories)
    tool_only = []
    for tool, g in per_case.groupby("tool_name"):
        good = g[g["confidence"].isin(["high", "medium"])]
        tool_only.append({
            "tool_name": tool, "n_total": len(g), "n_used": len(good),
            "median_uplift_pct": good["relative_uplift"].median() if len(good) else np.nan,
            "ci_low": bootstrap_ci(good["relative_uplift"].dropna().values)[0],
            "ci_high": bootstrap_ci(good["relative_uplift"].dropna().values)[1],
            "sum_uplift_units": good["uplift_units"].sum(),
            "n_high_conf": int((g["confidence"] == "high").sum()),
        })
    tool_df = pd.DataFrame(tool_only).sort_values("median_uplift_pct", ascending=False)
    tool_df.to_csv(OUT / "lace_tool_only_summary.csv", index=False)
    print("\n=== LACE tool-level uplift (cases with confidence ≥ medium) ===")
    print(tool_df.to_string(index=False))


if __name__ == "__main__":
    main()
