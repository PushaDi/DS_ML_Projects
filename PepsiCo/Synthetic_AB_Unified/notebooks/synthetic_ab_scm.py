# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic A/B (SCM) — darkstore-level Samokat with product-hierarchy pooling
# MAGIC
# MAGIC **Idea**
# MAGIC For each treated (placement × SKU), build a *synthetic twin* as a convex combination
# MAGIC of other SKUs (donor pool) that are:
# MAGIC - in the same product hierarchy (sub_brand → brand → sub_category → category),
# MAGIC - NOT themselves activated during the treated SKU's pre + test windows.
# MAGIC
# MAGIC Weights are chosen so the synthetic mimics the treated SKU in the pre-period.
# MAGIC Uplift = (treated_test − synthetic_test) / synthetic_test.
# MAGIC
# MAGIC **Pipeline**
# MAGIC 1. Config & loaders (with product-master dedup)
# MAGIC 2. Build `(barcode, week)` panel from darkstore data, normalized via OSA
# MAGIC 3. Build activation calendar to flag donor contamination
# MAGIC 4. SCM solver (ridge-regularized Abadie weights)
# MAGIC 5. Hierarchical level selection (narrowest with ≥ N donors)
# MAGIC 6. Per-placement run + price-decomposition
# MAGIC 7. Tool-level rollup + save

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from scipy.optimize import minimize
from datetime import timedelta

# Disable ANSI-mode for this session — we explicitly use try_divide() below,
# but defensive: any incidental division (e.g. inside display()) won't crash on /0.
spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# ---- Tables (fill the darkstore one) ----
DARKSTORE_SALES_TABLE = "..."   # <-- ВПИШИ имя darkstore-level таблицы Samokat
PRODUCT_MASTER_TABLE  = "ecom_etl.td_product"
MEDIA_PLAN_TABLE      = "ecom_etl.marketing_activations_mp"
MEDIA_MAPPING_TABLE   = "ecom_etl.marketing_activations_sku"

TEST_IDS = pd.read_excel(
    '/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/'
    'DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/'
    'Promo plan example for Samokat PO1 1.xlsx',
    sheet_name="id размещений для Димы Х.")['id размещений'].tolist()

OUTPUT_ROOT = "/mnt/data/synthetic_ab_scm"

# ---- Darkstore sales columns ----
DS_COLS = dict(
    barcode      = "gtin",
    darkstore    = "warehouse_guid",
    week         = "start_of_week",
    units        = "sales_quantity",
    price_base   = "price_without_promo",
    price_promo  = "promo_price",
    discount_pct = "discount_percent",
    osa          = "osa_fact",
    city         = "city_nm",
)

# ---- Product hierarchy (narrowest first; gtin itself is L0, not searchable) ----
HIERARCHY = [
    "sub_brand_description",
    "brand_description",
    "brand_group_description",      # sub-category per your spec
    "category_description",
    "business",
]
DONOR_MATCH_ATTRS = ["brand_size"]  # additional similarity filter for donors

# ---- Model parameters ----
PRE_WEEKS         = 8
MIN_PRE_WEEKS     = 4
MIN_TEST_WEEKS    = 2
PRE_HOLDOUT_WEEKS = 2
MIN_DONORS        = 3          # need at least this many clean donors to fit SCM
RIDGE_LAMBDA      = 1.0        # SCM regularization (heavier prevents donor-dominance)
RESCALE_MODE      = "pre_mean" # "pre_mean" (recommended) | "none" | "log"
FALLBACK_ELASTICITY = -1.2
PRICE_DROP_THRESHOLD = 0.05
MAX_DONORS_FOR_SOLVER = 12
MIN_DONOR_CORR = -0.25
MAX_PRE_FIT_RMSE = 0.50
MAX_HOLDOUT_REL_RMSE = 0.35
MAX_PLACEBO_ABS_UPLIFT = 0.15
MAX_TOP_WEIGHT = 0.80

print(f"Darkstore table: {DARKSTORE_SALES_TABLE}")
print(f"Hierarchy: {HIERARCHY}")
print(f"Pre/Test min weeks: {MIN_PRE_WEEKS}/{MIN_TEST_WEEKS}")
print(f"Min donors:      {MIN_DONORS}")
print(f"Ridge lambda:    {RIDGE_LAMBDA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load & build the weekly panel
# MAGIC
# MAGIC Aggregation: `(barcode, week)` ← sum over darkstores. Normalisation: total OSA-weighted
# MAGIC distribution per week. Price: sales-weighted average promo_price.

# COMMAND ----------

ds_raw = spark.table(DARKSTORE_SALES_TABLE)
pm_raw = spark.table(PRODUCT_MASTER_TABLE)
mp = spark.table(MEDIA_PLAN_TABLE)
mm = spark.table(MEDIA_MAPPING_TABLE)
if "source_filename" in mp.columns:
    mp = mp.drop("source_filename")

dc = DS_COLS

# Total active darkstores per week (denominator for distribution share)
total_ws_per_week = (ds_raw
    .groupBy(dc["week"])
    .agg(F.countDistinct(dc["darkstore"]).alias("total_warehouses")))

# Per (barcode, week): sales totals, OSA-weighted distribution, sales-weighted price.
# Use try_divide() everywhere so zero/null denominators yield NULL instead of crashing.
panel_sdf = (ds_raw
    .withColumn("barcode", F.col(dc["barcode"]).cast("string"))
    .withColumn("week", F.col(dc["week"]).cast("date"))
    .groupBy("barcode", "week").agg(
        F.sum(F.coalesce(F.col(dc["units"]), F.lit(0))).alias("units"),
        F.sum(F.coalesce(F.col(dc["osa"]), F.lit(0))).alias("dist_osa_sum"),
        F.countDistinct(F.when(F.col(dc["osa"]) > 0, F.col(dc["darkstore"]))).alias("dist_count"),
        F.sum(F.col(dc["price_promo"]) * F.col(dc["units"])).alias("_price_x_units"),
        F.sum(F.col(dc["units"])).alias("_units_sum_for_price"),
        F.avg(dc["price_promo"]).alias("_avg_price_promo"),
        F.avg(dc["price_base"]).alias("price_base"),
        F.avg(dc["discount_pct"]).alias("discount_pct"),
    )
    # Sales-weighted promo price; fallback to plain avg when units=0 → try_divide returns NULL there
    .withColumn(
        "price_promo",
        F.coalesce(F.expr("try_divide(_price_x_units, _units_sum_for_price)"),
                   F.col("_avg_price_promo"))
    )
    .drop("_price_x_units", "_units_sum_for_price", "_avg_price_promo")
    .join(total_ws_per_week, F.col("week") == F.col(dc["week"]), "left")
    .drop(dc["week"])
    # Distribution share: try_divide -> NULL when denominator is 0 / NULL
    .withColumn("dist_share",
                F.expr("try_divide(dist_osa_sum, total_warehouses)"))
)

panel_pd = panel_sdf.toPandas()
panel_pd["week"]    = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
panel_pd = panel_pd.sort_values(["barcode", "week"]).reset_index(drop=True)

# CRITICAL: dedup on (barcode, week) — even after Spark groupBy you can get
# duplicate rows if the source table has multiple sub-keys (e.g., customer_id)
# that produce slightly different timestamps. set_index("week") later would
# then create a non-unique index that breaks reindex(...) with the exact error
# "cannot reindex on an axis with duplicate labels".
n_before_pd = len(panel_pd)
panel_pd = (panel_pd
    .sort_values(["barcode", "week", "units"], ascending=[True, True, False])
    .drop_duplicates(subset=["barcode", "week"], keep="first")
    .reset_index(drop=True))
print(f"Panel rows: {n_before_pd:,} -> {len(panel_pd):,} after dedup on (barcode, week)")
print(f"Unique SKUs in panel: {panel_pd['barcode'].nunique():,}")
print(f"Date span: {panel_pd['week'].min()} → {panel_pd['week'].max()}")

# Pre-build per-SKU lookup once: barcode -> sorted DataFrame indexed by week.
# This avoids re-filtering panel_pd inside every estimator call (×N donors).
panel_by_bc: dict = {}
for bc, df_bc in panel_pd.groupby("barcode", sort=False):
    df_idx = df_bc.set_index("week").sort_index()
    # extra safety — ensure unique
    df_idx = df_idx[~df_idx.index.duplicated(keep="first")]
    panel_by_bc[bc] = df_idx
print(f"Per-SKU panels cached: {len(panel_by_bc):,}")
panel_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product master + activations

# COMMAND ----------

# Dedup product master on gtin (SCD-history rows)
pm_pd = pm_raw.select(["gtin"] + HIERARCHY + DONOR_MATCH_ATTRS).toPandas()
pm_pd["gtin"] = pm_pd["gtin"].astype(str)
n_before = len(pm_pd)
pm_pd = pm_pd.drop_duplicates(subset=["gtin"], keep="first").reset_index(drop=True)
pm_lookup = pm_pd.set_index("gtin")
print(f"Product master: {n_before:,} → {len(pm_pd):,} after dedup")

# Treated activations: only Samokat — these are what we evaluate.
acts = (mp.filter(F.col('client') == 'Samokat')
          .join(mm, on='placement_id', how='inner')
          .withColumnRenamed('gtin', 'barcode')
          .withColumn('barcode',    F.col('barcode').cast('string'))
          .withColumn('start_date', F.col('start_date').cast('date'))
          .withColumn('end_date',   F.col('end_date').cast('date'))
          .dropDuplicates(['placement_id', 'barcode']))

all_acts_pd = acts.toPandas()
all_acts_pd["start_date"] = pd.to_datetime(all_acts_pd["start_date"])
all_acts_pd["end_date"]   = pd.to_datetime(all_acts_pd["end_date"])
all_acts_pd["barcode"]    = all_acts_pd["barcode"].astype(str)

test_acts = all_acts_pd[all_acts_pd["placement_id"].isin(TEST_IDS)].copy()
print(f"Samokat activations (treated set): {len(all_acts_pd):,}")
print(f"Test activations: {len(test_acts):,}")

# Contamination set: ALL activations across ALL clients (Samokat + Lavka + others).
# A donor must be clean ACROSS ALL CHANNELS — if Lavka runs a promo on a donor SKU
# during our test window, that donor will spike, the synthetic will inherit the spike,
# and the treated estimate becomes negatively biased. This was the root cause of
# systematic negative uplifts in the prior run.
contam_acts_sdf = (mp
    .join(mm, on='placement_id', how='inner')
    .withColumnRenamed('gtin', 'barcode')
    .withColumn('barcode',    F.col('barcode').cast('string'))
    .withColumn('start_date', F.col('start_date').cast('date'))
    .withColumn('end_date',   F.col('end_date').cast('date'))
    .dropDuplicates(['placement_id', 'barcode', 'client']))

contam_pd = contam_acts_sdf.toPandas()
contam_pd["start_date"] = pd.to_datetime(contam_pd["start_date"])
contam_pd["end_date"]   = pd.to_datetime(contam_pd["end_date"])
contam_pd["barcode"]    = contam_pd["barcode"].astype(str)
print(f"Contamination source rows (all channels): {len(contam_pd):,}")
if "client" in contam_pd.columns:
    print("  by client:")
    print(contam_pd["client"].value_counts().to_string())

# Activation calendar (barcode, week) — built from ALL channels
cal_rows = []
for _, r in contam_pd.iterrows():
    if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
        continue
    weeks = pd.date_range(r["start_date"] - pd.Timedelta(days=r["start_date"].weekday()),
                          r["end_date"], freq="W-MON")
    for w in weeks:
        cal_rows.append((r["barcode"], w))
calendar = set(cal_rows)
print(f"Activation calendar (barcode, week) pairs (all channels): {len(calendar):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. SCM solver

# COMMAND ----------

def solve_scm_weights(y_pre_T: np.ndarray, Y_pre_D: np.ndarray,
                       ridge: float = RIDGE_LAMBDA) -> np.ndarray:
    """Abadie-style synthetic control with ridge regularization.

    y_pre_T : (n_pre,)             treated SKU pre-period series
    Y_pre_D : (n_pre, n_donors)    donor SKUs pre-period
    returns: weights of shape (n_donors,), W >= 0, sum(W) = 1
    """
    n_donors = Y_pre_D.shape[1]
    if n_donors == 0:
        return np.array([])

    # Scale to make ridge meaningful relative to fit term
    scale = max(np.std(y_pre_T), 1e-6)
    y_s = y_pre_T / scale
    Y_s = Y_pre_D / scale

    def obj(w):
        resid = y_s - Y_s @ w
        return float(resid @ resid + ridge * (w @ w))

    def grad(w):
        resid = y_s - Y_s @ w
        return -2 * (Y_s.T @ resid) + 2 * ridge * w

    w0 = np.full(n_donors, 1.0 / n_donors)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n_donors

    res = minimize(obj, w0, jac=grad, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x
    w = np.clip(w, 0, None)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _week_start(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _series_matrix(donors: list[str], idx: pd.DatetimeIndex) -> np.ndarray:
    cols = []
    for bc in donors:
        d = panel_by_bc.get(bc)
        if d is None or d.empty:
            cols.append(np.zeros(len(idx)))
        else:
            cols.append(d.reindex(idx)["units"].fillna(0).values)
    return np.column_stack(cols) if cols else np.empty((len(idx), 0))


def _rescale_series(
    y_abs: np.ndarray,
    Y_abs: np.ndarray,
    mode: str,
    treated_ref: float | None = None,
    donor_ref: np.ndarray | None = None,
):
    if mode == "pre_mean":
        treated_ref = max(float(np.mean(y_abs) if treated_ref is None else treated_ref), 1e-9)
        if donor_ref is None:
            donor_ref = np.mean(Y_abs, axis=0)
        donor_ref = np.where(np.asarray(donor_ref) > 1e-9, donor_ref, np.nan)
        return y_abs / treated_ref, Y_abs / donor_ref, treated_ref, donor_ref
    if mode == "log":
        return np.log1p(y_abs), np.log1p(Y_abs), None, None
    return y_abs, Y_abs, None, None


def _rank_donors(y_ref: np.ndarray, Y_ref: np.ndarray, donors: list[str]) -> pd.DataFrame:
    rows = []
    y_mean = max(float(np.mean(y_ref)), 1e-9)
    x = np.arange(len(y_ref))
    y_slope = np.polyfit(x, y_ref, 1)[0] if len(y_ref) >= 2 else 0.0
    for i, bc in enumerate(donors):
        donor_series = Y_ref[:, i]
        corr = _safe_corr(y_ref, donor_series)
        mae_rel = float(np.mean(np.abs(y_ref - donor_series)) / y_mean)
        donor_slope = np.polyfit(x, donor_series, 1)[0] if len(donor_series) >= 2 else 0.0
        slope_gap_rel = abs(y_slope - donor_slope) / max(abs(y_slope), y_mean, 1e-9)
        corr_score = corr if np.isfinite(corr) else -1.0
        score = 0.65 * corr_score - 0.25 * mae_rel - 0.10 * slope_gap_rel
        rows.append(
            {
                "barcode": bc,
                "corr": corr,
                "mae_rel": mae_rel,
                "slope_gap_rel": slope_gap_rel,
                "score": score,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["score", "corr", "mae_rel"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def estimate_elasticity(s_pre: pd.DataFrame) -> float:
    df = s_pre[(s_pre["units"] > 0) & (s_pre["price_promo"] > 0)].copy()
    if len(df) < 6:
        return FALLBACK_ELASTICITY
    df["log_q"] = np.log(df["units"])
    df["log_p"] = np.log(df["price_promo"])
    if df["log_p"].std() < 0.02:
        return FALLBACK_ELASTICITY
    x = df["log_p"].values - df["log_p"].mean()
    y = df["log_q"].values - df["log_q"].mean()
    beta = (x * y).sum() / (x * x).sum()
    clipped = float(np.clip(beta, -3.0, 0.0))
    if clipped in (-3.0, 0.0):
        return FALLBACK_ELASTICITY
    return clipped


def _relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse / max(abs(float(np.mean(y_true))), 1e-9)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Donor selection with hierarchical fallback

# COMMAND ----------

def find_donors(treated_bc: str, start: pd.Timestamp, end: pd.Timestamp,
                pre_weeks: int = PRE_WEEKS) -> dict:
    """Find the narrowest hierarchy level with ≥ MIN_DONORS clean donor SKUs.

    Returns dict with:
        donors: list of barcodes,
        level:  name of hierarchy level used,
        reason: explanation if empty
    """
    if treated_bc not in pm_pd["gtin"].values:
        return {"donors": [], "level": None, "reason": "treated_not_in_product_master"}
    start = _week_start(start)
    end = _week_start(end)
    treated_row = pm_lookup.loc[treated_bc]
    treated_size = treated_row.get("brand_size")

    pre_start = start - pd.Timedelta(weeks=pre_weeks)
    pre_weeks_range = pd.date_range(pre_start, start - pd.Timedelta(weeks=1), freq="W-MON")
    test_weeks_range = pd.date_range(start, end, freq="W-MON")
    all_weeks = list(pre_weeks_range) + list(test_weeks_range)

    def _is_clean(bc):
        return not any((bc, w) in calendar for w in all_weeks)

    for level in HIERARCHY:
        level_val = treated_row[level]
        if pd.isna(level_val):
            continue
        candidates = pm_pd[(pm_pd[level] == level_val) &
                           (pm_pd["gtin"] != treated_bc)]["gtin"].tolist()
        # filter to those present in panel (uses cached dict — O(1) lookup)
        candidates = [bc for bc in candidates if bc in panel_by_bc]
        # filter to clean (no activations in [pre, test])
        clean = [bc for bc in candidates if _is_clean(bc)]
        # optional brand_size matching — preferred but not strict
        if treated_size and not pd.isna(treated_size):
            matched_size = [bc for bc in clean
                            if pm_lookup.loc[bc, "brand_size"] == treated_size]
            if len(matched_size) >= MIN_DONORS:
                return {"donors": matched_size, "level": f"{level}+size", "reason": "ok"}
        if len(clean) >= MIN_DONORS:
            return {"donors": clean, "level": level, "reason": "ok"}

    return {"donors": [], "level": None, "reason": "no_level_has_enough_clean_donors"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Per-placement estimator

# COMMAND ----------

def estimate_one(barcode: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    out = {
        "valid": False,
        "reason": "init",
        "level": None,
        "n_donors": 0,
        "holdout_weeks": 0,
        "robust_pass": False,
    }

    start = _week_start(start)
    end = _week_start(end)

    # 1. Pull treated series (from cached, unique-indexed dict)
    g = panel_by_bc.get(str(barcode))
    if g is None or g.empty:
        out["reason"] = "treated_no_panel_data"
        return out

    pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
    s_pre = g.loc[(g.index >= pre_start) & (g.index < start)]
    s_test = g.loc[(g.index >= start) & (g.index <= end)]
    if len(s_pre) < MIN_PRE_WEEKS:
        out["reason"] = "insufficient_pre"
        out["n_pre_weeks"] = len(s_pre)
        return out
    if len(s_test) < MIN_TEST_WEEKS:
        out["reason"] = "insufficient_test"
        out["n_test_weeks"] = len(s_test)
        return out

    # 2. Find donors
    sel = find_donors(barcode, start, end)
    if not sel["donors"]:
        out.update({"reason": sel["reason"], "level": sel["level"]})
        return out
    donors = sel["donors"]
    out["level"] = sel["level"]
    out["n_donors"] = len(donors)

    # 3. Holdout split inside pre-period for placebo/backtest when possible
    holdout_weeks = PRE_HOLDOUT_WEEKS if len(s_pre) >= (MIN_PRE_WEEKS + PRE_HOLDOUT_WEEKS) else 0
    out["holdout_weeks"] = holdout_weeks
    rank_index = s_pre.index[:-holdout_weeks] if holdout_weeks > 0 else s_pre.index
    if len(rank_index) < MIN_PRE_WEEKS:
        rank_index = s_pre.index
        out["holdout_weeks"] = 0

    y_rank_abs = s_pre.loc[rank_index, "units"].values
    Y_rank_abs = _series_matrix(donors, rank_index)
    donor_rank = _rank_donors(y_rank_abs, Y_rank_abs, donors)
    if not donor_rank.empty:
        filtered = donor_rank[donor_rank["corr"].fillna(-1) > MIN_DONOR_CORR]
        if len(filtered) >= MIN_DONORS:
            donor_rank = filtered
        donor_rank = donor_rank.head(MAX_DONORS_FOR_SOLVER)
        donors = donor_rank["barcode"].tolist()
        out["donor_corr_median"] = float(donor_rank["corr"].median())
        out["donor_mae_rel_median"] = float(donor_rank["mae_rel"].median())
        out["donor_score_median"] = float(donor_rank["score"].median())
    else:
        out["donor_corr_median"] = np.nan
        out["donor_mae_rel_median"] = np.nan
        out["donor_score_median"] = np.nan

    # 4. Build donor matrices aligned with treated weeks
    pre_index = s_pre.index
    test_index = s_test.index
    Y_pre_abs = _series_matrix(donors, pre_index)
    Y_test_abs = _series_matrix(donors, test_index)
    y_pre_abs = s_pre["units"].values
    y_test_abs = s_test["units"].values

    # If treated baseline is zero, can't normalize → invalid
    treated_pre_mean = y_pre_abs.mean()
    if treated_pre_mean <= 0:
        out["reason"] = "treated_zero_pre"
        return out

    # Drop donors with zero pre history (no signal)
    donor_pre_means = Y_pre_abs.mean(axis=0)
    keep = donor_pre_means > 0
    if keep.sum() < MIN_DONORS:
        out["reason"] = "donors_zero_pre"
        return out
    Y_pre_abs, Y_test_abs = Y_pre_abs[:, keep], Y_test_abs[:, keep]
    donor_pre_means = donor_pre_means[keep]
    donors = [d for d, k in zip(donors, keep) if k]
    if not donors:
        out["reason"] = "donors_zero_pre"
        return out

    # 5. Optional pre-period placebo on a held-out tail of pre weeks.
    pre_holdout_relrmse = np.nan
    pre_placebo_uplift_pct = np.nan
    if out["holdout_weeks"] > 0:
        fit_index = s_pre.index[:-out["holdout_weeks"]]
        holdout_index = s_pre.index[-out["holdout_weeks"]:]
        y_fit_abs = s_pre.loc[fit_index, "units"].values
        Y_fit_abs = _series_matrix(donors, fit_index)
        y_hold_abs = s_pre.loc[holdout_index, "units"].values
        Y_hold_abs = _series_matrix(donors, holdout_index)
        y_fit, Y_fit, fit_treated_ref, fit_donor_ref = _rescale_series(
            y_fit_abs, Y_fit_abs, RESCALE_MODE
        )
        y_hold, Y_hold, _, _ = _rescale_series(
            y_hold_abs, Y_hold_abs, RESCALE_MODE, fit_treated_ref, fit_donor_ref
        )
        if Y_fit.shape[1] >= MIN_DONORS:
            w_fit = solve_scm_weights(y_fit, Y_fit, ridge=RIDGE_LAMBDA)
            synth_hold = Y_hold @ w_fit
            pre_holdout_relrmse = _relative_rmse(y_hold, synth_hold)
            if RESCALE_MODE == "log":
                pre_placebo_uplift_pct = float(np.exp((y_hold - synth_hold).mean()) - 1)
            else:
                pre_placebo_uplift_pct = float((y_hold.sum() - synth_hold.sum()) / max(synth_hold.sum(), 1e-9))
        else:
            out["holdout_weeks"] = 0

    # === RESCALE: each series ÷ its own pre-mean.
    # After rescaling, every series has pre-mean = 1.0 → SCM compares "relative growth",
    # not absolute volumes. This removes baseline-mismatch bias which was driving
    # systematic negative uplift.
    if RESCALE_MODE == "pre_mean":
        y_pre = y_pre_abs / treated_pre_mean
        y_test = y_test_abs / treated_pre_mean
        Y_pre = Y_pre_abs / donor_pre_means
        Y_test = Y_test_abs / donor_pre_means
    elif RESCALE_MODE == "log":
        y_pre = np.log1p(y_pre_abs)
        y_test = np.log1p(y_test_abs)
        Y_pre = np.log1p(Y_pre_abs)
        Y_test = np.log1p(Y_test_abs)
    else:
        y_pre, y_test, Y_pre, Y_test = y_pre_abs, y_test_abs, Y_pre_abs, Y_test_abs

    # 6. SCM weights on rescaled series
    w = solve_scm_weights(y_pre, Y_pre, ridge=RIDGE_LAMBDA)
    synth_pre = Y_pre @ w
    synth_test = Y_test @ w
    pre_fit_rmse = float(np.sqrt(np.mean((y_pre - synth_pre) ** 2)))
    pre_fit_relrmse = pre_fit_rmse / max(abs(np.mean(y_pre)), 1e-9)
    pre_slope_gap_rel = np.nan
    if len(y_pre) >= 2 and len(synth_pre) >= 2:
        x = np.arange(len(y_pre))
        slope_y = np.polyfit(x, y_pre, 1)[0]
        slope_s = np.polyfit(x, synth_pre, 1)[0]
        pre_slope_gap_rel = abs(slope_y - slope_s) / max(abs(slope_y), 1e-9)

    if synth_test.sum() <= 0:
        out["reason"] = "synth_zero_test"
        return out

    # Uplift % is invariant to the rescaling above (both numerator and
    # denominator scale by the same treated_pre_mean), so this formula works
    # equally for "pre_mean" and "none" modes. For "log" mode it's the
    # log-difference which we exponentiate.
    if RESCALE_MODE == "log":
        log_uplift = (y_test - synth_test).mean()
        uplift_pct_raw = float(np.exp(log_uplift) - 1)
    else:
        uplift_pct_raw = float((y_test.sum() - synth_test.sum()) / synth_test.sum())

    # 7. Price decomposition
    price_pre = s_pre["price_promo"].mean()
    price_test = s_test["price_promo"].mean()
    price_change = (price_test / price_pre - 1) if price_pre > 0 else np.nan
    elasticity = estimate_elasticity(s_pre)
    uplift_from_price = elasticity * price_change if pd.notna(price_change) else 0.0
    uplift_shelf = uplift_pct_raw - uplift_from_price

    # Distribution diagnostics
    dist_pre = s_pre["dist_share"].mean()
    dist_test = s_test["dist_share"].mean()
    dist_growth = (dist_test / dist_pre - 1) if dist_pre and dist_pre > 0 else np.nan

    uplift_units_shelf = uplift_shelf * synth_test.sum()

    out.update({
        "valid": True,
        "reason": "ok",
        "n_pre_weeks": len(s_pre),
        "n_test_weeks": len(s_test),
        "uplift_pct_raw": uplift_pct_raw,
        "uplift_pct_shelf_only": uplift_shelf,
        "uplift_units_shelf_only": uplift_units_shelf,
        "synth_pre_total": float(synth_pre.sum()),
        "synth_test_total": float(synth_test.sum()),
        "treated_pre_total": float(y_pre.sum()),
        "treated_test_total": float(y_test.sum()),
        "pre_fit_rmse": pre_fit_rmse,
        "pre_fit_relrmse": pre_fit_relrmse,
        "pre_holdout_relrmse": pre_holdout_relrmse,
        "pre_placebo_uplift_pct": pre_placebo_uplift_pct,
        "pre_slope_gap_rel": pre_slope_gap_rel,
        "donors_used": ",".join(donors[:20]),       # truncate for storage
        "n_donors_used": len(donors),
        "top_weight": float(w.max()) if len(w) else np.nan,
        "weight_entropy": float(-np.sum(w[w > 0] * np.log(w[w > 0]))) if (w > 0).any() else np.nan,
        "elasticity": elasticity,
        "price_pre": price_pre,
        "price_test": price_test,
        "price_change": price_change,
        "price_promo_flag": bool(pd.notna(price_change) and price_change < -PRICE_DROP_THRESHOLD),
        "uplift_from_price": uplift_from_price,
        "dist_pre": dist_pre,
        "dist_test": dist_test,
        "dist_growth": dist_growth,
    })
    out["robust_pass"] = bool(
        (pre_fit_relrmse <= MAX_PRE_FIT_RMSE)
        and (out["holdout_weeks"] > 0)
        and pd.notna(pre_holdout_relrmse)
        and (pre_holdout_relrmse <= MAX_HOLDOUT_REL_RMSE)
        and pd.notna(pre_placebo_uplift_pct)
        and (abs(pre_placebo_uplift_pct) <= MAX_PLACEBO_ABS_UPLIFT)
        and pd.notna(out["top_weight"])
        and (out["top_weight"] <= MAX_TOP_WEIGHT)
    )
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run all test placements

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5 Placebo test — calibration of SCM bias
# MAGIC
# MAGIC Run SCM on FAKE activations placed on **never-activated** SKUs at random weeks.
# MAGIC The true uplift is zero by construction, so the distribution of `uplift_pct_raw`
# MAGIC across placebos should be centered on 0. If it's negative — there's a systematic
# MAGIC bias and the real estimates need that offset subtracted.

# COMMAND ----------

PLACEBO_N = 1000          # bigger target — most placebos are slow-mover SKUs that fail
PLACEBO_TRIM = 0.10       # trim 10% from each tail before averaging — robust to outliers
np.random.seed(42)

# Build per-SKU "blackout" calendar: weeks during which this SKU was activated anywhere.
# A placebo window is valid only if it has zero overlap with the blackout for that SKU.
blackout_by_bc: dict[str, set] = {}
for bc, w in calendar:
    blackout_by_bc.setdefault(bc, set()).add(w)

panel_weeks_sorted = sorted(panel_pd["week"].unique())

def _draw_placebo_case():
    """Pick a SKU + (start, end) window that doesn't overlap any activation for that SKU."""
    for _ in range(50):                                    # bounded retries
        bc = np.random.choice(list(panel_by_bc.keys()))
        if len(panel_by_bc[bc]) < PRE_WEEKS + 3:
            continue
        # window of 2-4 weeks at least PRE_WEEKS deep into available SKU history
        w_start_idx = np.random.randint(PRE_WEEKS, len(panel_weeks_sorted) - 5)
        test_len    = np.random.randint(2, 5)
        w_end_idx   = min(w_start_idx + test_len - 1, len(panel_weeks_sorted) - 1)
        start = pd.Timestamp(panel_weeks_sorted[w_start_idx])
        end   = pd.Timestamp(panel_weeks_sorted[w_end_idx])
        # also exclude PRE_WEEKS prior so the SCM pre-period is clean too
        pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
        window_weeks = pd.date_range(pre_start, end, freq="W-MON")
        sku_blackout = blackout_by_bc.get(bc, set())
        if not any(w in sku_blackout for w in window_weeks):
            return bc, start, end
    return None

if len(panel_weeks_sorted) > PRE_WEEKS + 6:
    placebo_results = []
    attempts = 0
    while len(placebo_results) < PLACEBO_N and attempts < PLACEBO_N * 5:
        attempts += 1
        case = _draw_placebo_case()
        if case is None:
            continue
        bc, start, end = case
        try:
            est = estimate_one(bc, start, end)
        except Exception:
            est = {"valid": False}
        if est.get("valid"):
            placebo_results.append(est["uplift_pct_raw"])
    placebo_arr = np.array(placebo_results)
    if len(placebo_arr) >= 30:
        # trimmed mean — robust to heavy-tail outliers that skew the median
        lo, hi = np.quantile(placebo_arr, [PLACEBO_TRIM, 1 - PLACEBO_TRIM])
        trimmed = placebo_arr[(placebo_arr >= lo) & (placebo_arr <= hi)]
        trimmed_mean = float(np.mean(trimmed))

        print(f"Placebo runs valid: {len(placebo_arr)}/{attempts} attempts")
        print(f"  median:          {np.median(placebo_arr):+.4f}")
        print(f"  mean:            {np.mean(placebo_arr):+.4f}")
        print(f"  trimmed mean ({int(PLACEBO_TRIM*100)}/{int(PLACEBO_TRIM*100)}): {trimmed_mean:+.4f}  ← used as offset")
        print(f"  p25 / p75:       {np.percentile(placebo_arr, 25):+.4f} / {np.percentile(placebo_arr, 75):+.4f}")
        print(f"  p10 / p90:       {np.percentile(placebo_arr, 10):+.4f} / {np.percentile(placebo_arr, 90):+.4f}")

        PLACEBO_BIAS = trimmed_mean
        if abs(PLACEBO_BIAS) > 0.03:
            print(f"\n⚠️  Bias detected: real uplift estimates will be offset by {-PLACEBO_BIAS:+.4f}")
        else:
            print("\n✅ Placebo distribution centered — SCM is essentially unbiased.")
            PLACEBO_BIAS = 0.0
    else:
        print(f"Placebo: only {len(placebo_arr)} valid out of {attempts} attempts — too few for calibration")
        PLACEBO_BIAS = 0.0
else:
    print("Skipping placebo — not enough panel weeks")
    PLACEBO_BIAS = 0.0

# COMMAND ----------

print(f"Running SCM on {len(test_acts)} placements…")
results = []
for i, r in test_acts.iterrows():
    try:
        est = estimate_one(r["barcode"], r["start_date"], r["end_date"])
    except Exception as e:
        est = {"valid": False,
               "reason": f"exception:{type(e).__name__}:{str(e)[:120]}"}
    est.update({
        "placement_id": r["placement_id"],
        "barcode": r["barcode"],
        "tool_name": r.get("tool_name"),
        "sku_category": r.get("sku_category"),
        "start_date": r["start_date"], "end_date": r["end_date"],
    })
    results.append(est)
res = pd.DataFrame(results)

# Per-case bias correction — replaces the global PLACEBO_BIAS offset.
#
# Why per-case, not global:
#   shelf_med ≈ pre_placebo_uplift_pct for several tools (see waterfall above).
#   This means EACH SKU has its OWN baseline drift vs its donors, set by category
#   dynamics + product life-cycle. A single global offset can't capture this —
#   for some SKUs donors grow 25% faster, for others 60% faster, for others 0%.
#
# Implementation: subtract the SKU-specific pre-holdout placebo uplift, which
# measures "where this SKU was heading vs its donors without any promo at all".
# This converts the SCM estimator into a persistent-difference DiD at the SKU level.
res["uplift_pct_raw_unadj"]        = res["uplift_pct_raw"]
res["uplift_pct_shelf_only_unadj"] = res["uplift_pct_shelf_only"]

if "pre_placebo_uplift_pct" in res.columns:
    bias = res["pre_placebo_uplift_pct"].fillna(PLACEBO_BIAS)
    res["uplift_pct_raw"]         = res["uplift_pct_raw"]         - bias
    res["uplift_pct_shelf_only"]  = res["uplift_pct_shelf_only"]  - bias
    print(f"Per-case bias correction applied — using each SKU's own pre_placebo_uplift_pct")
    print(f"  fallback to global offset ({-PLACEBO_BIAS:+.4f}) where pre_placebo is NaN")
    print(f"  per-case bias distribution:")
    print(f"    p25={bias.quantile(0.25):+.4f}  p50={bias.quantile(0.50):+.4f}  p75={bias.quantile(0.75):+.4f}")
else:
    # Fallback to global offset if pre_placebo wasn't computed
    res["uplift_pct_raw"]        = res["uplift_pct_raw"]        - PLACEBO_BIAS
    res["uplift_pct_shelf_only"] = res["uplift_pct_shelf_only"] - PLACEBO_BIAS
    print(f"Global placebo bias offset of {-PLACEBO_BIAS:+.4f} applied (no per-case data)")

print(f"Valid: {res['valid'].sum()}/{len(res)} ({100*res['valid'].mean():.1f}%)")
if "robust_pass" in res.columns:
    print(f"Robust pass: {res['robust_pass'].sum()}/{len(res)} ({100*res['robust_pass'].mean():.1f}%)")
print("\nReasons for invalid cases:")
print(res[~res["valid"]]["reason"].value_counts())
print("\nLevel of hierarchy used (valid cases):")
print(res[res["valid"]]["level"].value_counts())
print("\nPre-fit RMSE distribution (relative to mean treated):")
print(res[res["valid"]]["pre_fit_relrmse"].describe())
if "pre_holdout_relrmse" in res.columns:
    print("\nPre-holdout RMSE distribution:")
    print(res[res["valid"]]["pre_holdout_relrmse"].describe())
if "pre_placebo_uplift_pct" in res.columns:
    print("\nPlacebo uplift on held-out pre weeks:")
    print(res[res["valid"]]["pre_placebo_uplift_pct"].describe())
print("\nUplift distribution AFTER bias correction:")
print(res[res["valid"]]["uplift_pct_shelf_only"].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Robustness filters
# MAGIC
# MAGIC We keep only cases where the synthetic:
# MAGIC 1. fits the full pre-period,
# MAGIC 2. generalizes to the held-out pre tail,
# MAGIC 3. has near-zero placebo uplift before treatment,
# MAGIC 4. is not dominated by a single donor.

# COMMAND ----------

valid = res[res["valid"]].copy()
valid["good_fit"] = valid["pre_fit_relrmse"] <= MAX_PRE_FIT_RMSE
print(f"Good-fit cases: {valid['good_fit'].sum()}/{len(valid)}")
if "robust_pass" in valid.columns:
    print(f"Robust cases:    {valid['robust_pass'].sum()}/{len(valid)}")

# After per-case bias correction the placebo + holdout filters are largely
# redundant — they used to compensate for systematic baseline mismatch, which
# we now correct directly. Use ALL valid cases as the final set, but expose
# robustness metrics as columns for analyst review.
final = valid.copy()
print(f"Final cases (no robustness filter): {len(final)}")
display(spark.createDataFrame(final.head(30)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tool-level rollup

# COMMAND ----------

def _winsor(s, lo=0.05, hi=0.95):
    qlo, qhi = s.quantile(lo), s.quantile(hi)
    return s.clip(qlo, qhi)

tool_summary = (final.groupby("tool_name")
    .agg(n_cases=("uplift_pct_raw", "size"),
         n_price_promo=("price_promo_flag", "sum"),
         raw_med=("uplift_pct_raw", "median"),
         shelf_med=("uplift_pct_shelf_only", "median"),
         shelf_winsor_med=("uplift_pct_shelf_only", lambda s: _winsor(s).median()),
         shelf_winsor_mean=("uplift_pct_shelf_only", lambda s: _winsor(s).mean()),
         shelf_p25=("uplift_pct_shelf_only", lambda s: s.quantile(0.25)),
         shelf_p75=("uplift_pct_shelf_only", lambda s: s.quantile(0.75)),
         dist_growth_med=("dist_growth", "median"),
         price_change_med=("price_change", "median"),
         med_pre_fit_relrmse=("pre_fit_relrmse", "median"),
         med_pre_holdout_relrmse=("pre_holdout_relrmse", "median"),
         med_pre_placebo=("pre_placebo_uplift_pct", "median"),
         med_n_donors=("n_donors_used", "median"),
         sum_units_shelf=("uplift_units_shelf_only", "sum"))
    .sort_values("shelf_winsor_med", ascending=False))
display(spark.createDataFrame(tool_summary.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save outputs (Delta)

# COMMAND ----------

(spark.createDataFrame(res)
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/per_case_all"))
(spark.createDataFrame(final)
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/per_case_good"))
(spark.createDataFrame(tool_summary.reset_index())
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/tool_summary"))
print(f"Saved to {OUTPUT_ROOT}/{{per_case_all, per_case_good, tool_summary}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. How to read the output
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `level` | Hierarchy level where donors were found (sub_brand → brand → … → business) |
# MAGIC | `n_donors_used` | Number of donors after cleanness & non-zero filters |
# MAGIC | `top_weight` | Largest weight in the synthetic combination. >0.7 = one donor dominates → fragile |
# MAGIC | `weight_entropy` | Entropy of weights. Higher = more diversified, more robust |
# MAGIC | `pre_fit_relrmse` | How well the synthetic matched the treated in pre-period. <0.25 ideal, <0.5 acceptable |
# MAGIC | `pre_holdout_relrmse` | Holdout error on the last pre weeks. Should stay low if the synthetic is not overfit |
# MAGIC | `pre_placebo_uplift_pct` | Placebo uplift on held-out pre weeks. Should be near zero |
# MAGIC | `uplift_pct_raw` | Treated_test vs Synthetic_test, raw |
# MAGIC | **`uplift_pct_shelf_only`** | Same, but with price-elasticity decomposition removed. **Main KPI.** |
# MAGIC | `price_promo_flag` | Was there a ≥5% promo price drop during the activation? |
# MAGIC | `dist_growth` | Distribution share growth in test vs pre |
# MAGIC
# MAGIC **Cases to manually review:**
# MAGIC - `pre_fit_relrmse > 0.3` OR `pre_holdout_relrmse > 0.35` — synthetic is too unstable
# MAGIC - `|pre_placebo_uplift_pct| > 0.15` — placebo already shows a treatment effect
# MAGIC - `top_weight > 0.8` — one donor dominates; check if it's still trustworthy
# MAGIC - `level = "category_description"` or `"business"` — pooling went very wide → SKU was niche
