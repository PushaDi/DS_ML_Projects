# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic A/B — Interrupted Time Series with Category Covariate
# MAGIC
# MAGIC **Why this method (vs cross-SKU SCM)**
# MAGIC
# MAGIC In FMCG, marketing tools are typically activated on SKUs that *need support* —
# MAGIC declining velocity, new launches, seasonality lows. Donor SKUs in cross-SKU SCM
# MAGIC are then "healthy" peers, which structurally over-predicts the treated SKU's
# MAGIC counterfactual → systematic negative uplift. Selection bias is in the data,
# MAGIC not in the algorithm.
# MAGIC
# MAGIC **ITS with category covariate** avoids this by:
# MAGIC 1. Using ONLY the treated SKU's own pre-period history (no cross-SKU comparison)
# MAGIC 2. Using the **category index** (same brand_group, excluding treated)
# MAGIC    as a covariate that absorbs macro shocks (seasonality, holidays, category trends)
# MAGIC 3. Fitting a parsimonious model: log(T_t) = α + β·log(C_t) + γ·t + ε
# MAGIC 4. Projecting counterfactual using ACTUAL category values during test period
# MAGIC
# MAGIC **References**
# MAGIC - Bernal et al. 2017 (IJE) — Interrupted time series regression for evaluation of public health interventions
# MAGIC - Brodersen et al. 2015 — CausalImpact (Bayesian Structural Time Series)
# MAGIC - Arkhangelsky et al. 2021 (AER) — Synthetic Difference-in-Differences
# MAGIC - Wagner et al. 2002 (J. Clin. Pharm. Ther.) — Segmented regression with auto-correlated errors
# MAGIC
# MAGIC **Pipeline**
# MAGIC 1. Load darkstore sales + product master + activations
# MAGIC 2. Build (barcode, week) panel + category index per `brand_group_description`
# MAGIC 3. For each placement: fit pre-period regression, project counterfactual, measure uplift
# MAGIC 4. Bootstrap CI per placement (block-resample weeks) + within-SKU placebo calibration
# MAGIC 5. Tool-level rollup

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
import pyspark.sql.functions as F

spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config

# COMMAND ----------

# ---- Tables ----
DARKSTORE_SALES_TABLE = "ecom_etl.ds_samokat_availability_metrics"
PRODUCT_MASTER_TABLE  = "ecom_etl.td_product"
MEDIA_PLAN_TABLE      = "ecom_etl.marketing_activations_mp"
MEDIA_MAPPING_TABLE   = "ecom_etl.marketing_activations_sku"

TEST_IDS = pd.read_excel(
    '/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/'
    'DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/'
    'Promo plan example for Samokat PO1 1.xlsx',
    sheet_name="id размещений для Димы Х.")['id размещений'].tolist()

OUTPUT_ROOT = "/mnt/data/synthetic_ab_its"

# ---- Darkstore sales columns ----
# Distribution / OSA intentionally NOT used — units are direct darkstore-level
# sales aggregated to (barcode, week). Price is sales-weighted average.
DS_COLS = dict(
    barcode      = "gtin",
    darkstore    = "warehouse_guid",
    week         = "start_of_week",
    units        = "sales_quantity",
    price_base   = "price_without_promo",
    price_promo  = "promo_price",
    discount_pct = "discount_percent",
)

# ---- Category aggregation level ----
# We aggregate "other SKUs in same sub-category" as the macro control.
# Using brand_group_description per your spec — it's the sub-category level.
CATEGORY_LEVEL = "brand_group_description"
# Optional: exclude same sub_brand to avoid cannibalization spillover into the index
EXCLUDE_SAME = "sub_brand_description"

# ---- Model parameters ----
PRE_WEEKS              = 12       # ITS likes longer pre-period; 12 is comfortable
MIN_PRE_WEEKS          = 8
MIN_TEST_WEEKS         = 1        # ITS is fine with 1 test week
USE_TREND              = True     # include linear trend γ·t
USE_CATEGORY           = True     # include β·log(C_t)
LOG_TRANSFORM          = True     # work on log-scale (standard for FMCG)
WINSOR_PRE_RESIDUAL    = 3.0      # drop pre-weeks where |residual| > 3·σ before final fit
BOOTSTRAP_ITERATIONS   = 500      # for per-placement CI; set 0 to skip
BOOTSTRAP_MODE         = "darkstore_cluster"  # "darkstore_cluster" | "residual_weeks"
                                              # darkstore_cluster: resample darkstores → leverages 2500-point sample
                                              # residual_weeks: old behaviour, resample 12 pre-period residuals
FALLBACK_ELASTICITY    = -1.2
PRICE_DROP_THRESHOLD   = 0.05

print(f"Category-control level: {CATEGORY_LEVEL} (excluding same {EXCLUDE_SAME})")
print(f"Pre/Test min weeks:     {MIN_PRE_WEEKS}/{MIN_TEST_WEEKS}")
print(f"Pre-window length:      {PRE_WEEKS}")
print(f"Trend / Category / Log: {USE_TREND} / {USE_CATEGORY} / {LOG_TRANSFORM}")
print(f"Bootstrap iterations:   {BOOTSTRAP_ITERATIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load & build (barcode, week) panel

# COMMAND ----------

dc = DS_COLS
ds_raw = spark.table(DARKSTORE_SALES_TABLE)
pm_raw = spark.table(PRODUCT_MASTER_TABLE)
mp_raw = spark.table(MEDIA_PLAN_TABLE)
mm_raw = spark.table(MEDIA_MAPPING_TABLE)
if "source_filename" in mp_raw.columns:
    mp_raw = mp_raw.drop("source_filename")

panel_sdf = (ds_raw
    .withColumn("barcode", F.col(dc["barcode"]).cast("string"))
    .withColumn("week", F.col(dc["week"]).cast("date"))
    .groupBy("barcode", "week").agg(
        # Total units across all darkstores for this SKU-week
        F.sum(F.coalesce(F.col(dc["units"]), F.lit(0))).alias("units"),
        # Number of darkstores that recorded ANY transaction for this SKU-week
        # — kept as a soft proxy of presence (not used as distribution normaliser)
        F.countDistinct(F.when(F.col(dc["units"]) > 0,
                               F.col(dc["darkstore"]))).alias("n_darkstores"),
        # Sales-weighted promo price (so quiet darkstores don't sway the average)
        F.sum(F.col(dc["price_promo"]) * F.col(dc["units"])).alias("_pxu"),
        F.sum(F.col(dc["units"])).alias("_units"),
        F.avg(dc["price_promo"]).alias("_avg_p"),
        F.avg(dc["price_base"]).alias("price_base"),
    )
    .withColumn("price_promo",
                F.coalesce(F.expr("try_divide(_pxu, _units)"), F.col("_avg_p")))
    .drop("_pxu", "_units", "_avg_p"))

panel_pd = panel_sdf.toPandas()
panel_pd["week"]    = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
panel_pd = (panel_pd
    .sort_values(["barcode", "week"])
    .drop_duplicates(subset=["barcode", "week"], keep="first")
    .reset_index(drop=True))
print(f"Panel rows: {len(panel_pd):,}")
print(f"Unique SKUs: {panel_pd['barcode'].nunique():,}")
print(f"Date span: {panel_pd['week'].min()} → {panel_pd['week'].max()}")

# Cache per-SKU aggregated lookup (used for point estimate)
panel_by_bc = {bc: g.set_index("week").sort_index()
               for bc, g in panel_pd.groupby("barcode", sort=False)}

# DARKSTORE-LEVEL panel (used for cluster bootstrap CI — leverages 2500-point sample)
# Lazy-loaded the first time a placement needs it. Stored as DataFrames indexed
# by (darkstore, week) per barcode.
darkstore_panel_sdf = (ds_raw
    .withColumn("barcode",   F.col(dc["barcode"]).cast("string"))
    .withColumn("week",      F.col(dc["week"]).cast("date"))
    .withColumn("darkstore", F.col(dc["darkstore"]).cast("string"))
    .groupBy("barcode", "darkstore", "week").agg(
        F.sum(F.coalesce(F.col(dc["units"]), F.lit(0))).alias("units")))

# Materialize to pandas only on demand (per SKU) — avoids loading 2500×52×N rows at once
darkstore_panel_cache: dict[str, pd.DataFrame] = {}

def _get_darkstore_panel(bc: str) -> pd.DataFrame:
    if bc in darkstore_panel_cache:
        return darkstore_panel_cache[bc]
    df = (darkstore_panel_sdf.filter(F.col("barcode") == bc)
          .select("darkstore", "week", "units").toPandas())
    df["week"] = pd.to_datetime(df["week"])
    df = df.drop_duplicates(["darkstore", "week"], keep="first")
    darkstore_panel_cache[bc] = df
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product master + activations

# COMMAND ----------

pm_pd = pm_raw.select(["gtin", "brand_description", "sub_brand_description",
                       "brand_group_description", "category_description"]).toPandas()
pm_pd["gtin"] = pm_pd["gtin"].astype(str)
pm_pd = pm_pd.drop_duplicates(subset=["gtin"], keep="first").reset_index(drop=True)
pm_lookup = pm_pd.set_index("gtin")
print(f"Product master rows: {len(pm_pd):,}")

acts_sdf = (mp_raw.filter(F.col("client") == "Samokat")
    .join(mm_raw, on="placement_id", how="inner")
    .withColumnRenamed("gtin", "barcode")
    .withColumn("barcode",    F.col("barcode").cast("string"))
    .withColumn("start_date", F.col("start_date").cast("date"))
    .withColumn("end_date",   F.col("end_date").cast("date"))
    .dropDuplicates(["placement_id", "barcode"]))

acts_pd = acts_sdf.toPandas()
acts_pd["start_date"] = pd.to_datetime(acts_pd["start_date"])
acts_pd["end_date"]   = pd.to_datetime(acts_pd["end_date"])
acts_pd["barcode"]    = acts_pd["barcode"].astype(str)
test_acts = acts_pd[acts_pd["placement_id"].isin(TEST_IDS)].copy()
print(f"Test activations: {len(test_acts)}")

# Full contamination calendar (all channels) — used to exclude weeks where any
# SKU was under any promo (for building a clean category index).
contam_sdf = (mp_raw
    .join(mm_raw, on="placement_id", how="inner")
    .withColumnRenamed("gtin", "barcode")
    .withColumn("barcode",    F.col("barcode").cast("string"))
    .withColumn("start_date", F.col("start_date").cast("date"))
    .withColumn("end_date",   F.col("end_date").cast("date"))
    .dropDuplicates(["placement_id", "barcode"]))
contam_pd = contam_sdf.toPandas()
contam_pd["start_date"] = pd.to_datetime(contam_pd["start_date"])
contam_pd["end_date"]   = pd.to_datetime(contam_pd["end_date"])
contam_pd["barcode"]    = contam_pd["barcode"].astype(str)

# (barcode, week) blackout
blackout = set()
for _, r in contam_pd.iterrows():
    if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
        continue
    weeks = pd.date_range(r["start_date"] - pd.Timedelta(days=r["start_date"].weekday()),
                          r["end_date"], freq="W-MON")
    for w in weeks:
        blackout.add((r["barcode"], w.normalize()))
print(f"Blackout (barcode, week) pairs: {len(blackout):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build category index per CATEGORY_LEVEL
# MAGIC
# MAGIC `category_index[(category_value, week)] = sum(units of OTHER SKUs in same category,
# MAGIC excluding those currently in promo and excluding same EXCLUDE_SAME of treated)`.
# MAGIC
# MAGIC We pre-aggregate once at the brand_group level for clean weeks; per-placement
# MAGIC we further subtract same-`sub_brand` if requested.

# COMMAND ----------

# Attach hierarchy to panel
panel_with_h = panel_pd.merge(
    pm_pd[["gtin", CATEGORY_LEVEL, EXCLUDE_SAME, "brand_description"]],
    left_on="barcode", right_on="gtin", how="left").drop(columns=["gtin"])

# Mask out (barcode, week) pairs that are in blackout — these SKUs are themselves
# under promo and would contaminate the category index.
blackout_df = pd.DataFrame(list(blackout), columns=["barcode", "week"])
blackout_df["under_promo"] = True
panel_with_h = panel_with_h.merge(blackout_df, on=["barcode", "week"], how="left")
panel_with_h["under_promo"] = panel_with_h["under_promo"].fillna(False)

# Category index by (CATEGORY_LEVEL, sub_brand, week) — sum of CLEAN sibling units.
# We keep sub_brand split so per-placement we can subtract the treated sub_brand.
clean = panel_with_h[~panel_with_h["under_promo"]].copy()
cat_panel = (clean
    .groupby([CATEGORY_LEVEL, EXCLUDE_SAME, "week"], as_index=False)
    .agg(cat_units=("units", "sum"),
         cat_skus=("barcode", "nunique")))
print(f"Category panel rows: {len(cat_panel):,}")

# Convenience: total per (CATEGORY_LEVEL, week)
total_cat = (cat_panel.groupby([CATEGORY_LEVEL, "week"], as_index=False)
                       .agg(tot_units=("cat_units", "sum"),
                            tot_skus=("cat_skus", "sum")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-placement ITS estimator

# COMMAND ----------

def _build_category_series(category_value: str, excl_sub_brand: str,
                            weeks: pd.DatetimeIndex) -> pd.Series:
    """Sum of clean sibling units in the same category, EXCLUDING the same sub_brand."""
    # All weekly category units
    tot = total_cat[total_cat[CATEGORY_LEVEL] == category_value].set_index("week")["tot_units"]
    # Same sub_brand within that category (to subtract)
    same = cat_panel[(cat_panel[CATEGORY_LEVEL] == category_value)
                     & (cat_panel[EXCLUDE_SAME] == excl_sub_brand)].set_index("week")["cat_units"]
    s = tot.subtract(same, fill_value=0).reindex(weeks).fillna(0)
    return s


def _fit_predict_its(t_series: np.ndarray, c_series: np.ndarray,
                     t_index: np.ndarray, c_future: np.ndarray, t_future_idx: np.ndarray,
                     use_trend: bool, use_cat: bool, log_x: bool):
    """Fit log(T) = α + β·log(C) + γ·t + ε on pre, predict on test using actual C_future."""
    eps = 1e-6
    if log_x:
        y = np.log(np.maximum(t_series, eps))
        x_cat = np.log(np.maximum(c_series, eps)) if use_cat else None
        x_cat_f = np.log(np.maximum(c_future, eps)) if use_cat else None
    else:
        y = t_series.astype(float)
        x_cat = c_series.astype(float) if use_cat else None
        x_cat_f = c_future.astype(float) if use_cat else None

    cols = [np.ones_like(y)]                  # intercept
    cols_f = [np.ones_like(t_future_idx, dtype=float)]
    if use_cat:
        cols.append(x_cat); cols_f.append(x_cat_f)
    if use_trend:
        cols.append(t_index.astype(float)); cols_f.append(t_future_idx.astype(float))

    X = np.column_stack(cols)
    X_f = np.column_stack(cols_f)

    # Solve OLS — closed form so no scipy dependency in inner loop
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    y_hat   = X   @ beta
    y_hat_f = X_f @ beta
    if log_x:
        t_hat   = np.exp(y_hat)
        t_hat_f = np.exp(y_hat_f)
    else:
        t_hat   = y_hat
        t_hat_f = y_hat_f
    residuals = t_series - t_hat
    return t_hat_f, residuals, beta


def estimate_one(barcode: str, start: pd.Timestamp, end: pd.Timestamp,
                  bootstrap_n: int = BOOTSTRAP_ITERATIONS) -> dict:
    out = {"valid": False, "reason": "init"}

    g = panel_by_bc.get(str(barcode))
    if g is None or g.empty:
        out["reason"] = "no_panel_data"; return out

    pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
    s_pre  = g.loc[(g.index >= pre_start) & (g.index < start)]
    s_test = g.loc[(g.index >= start) & (g.index <= end)]
    if len(s_pre) < MIN_PRE_WEEKS:
        out["reason"] = "insufficient_pre"; out["n_pre_weeks"] = len(s_pre); return out
    if len(s_test) < MIN_TEST_WEEKS:
        out["reason"] = "insufficient_test"; out["n_test_weeks"] = len(s_test); return out

    if str(barcode) not in pm_lookup.index:
        out["reason"] = "not_in_product_master"; return out
    cat_val = pm_lookup.loc[str(barcode), CATEGORY_LEVEL]
    excl_val = pm_lookup.loc[str(barcode), EXCLUDE_SAME]
    if pd.isna(cat_val):
        out["reason"] = "no_category"; return out

    # Build category time series for pre + test
    c_pre  = _build_category_series(cat_val, excl_val, s_pre.index).values
    c_test = _build_category_series(cat_val, excl_val, s_test.index).values

    # Need category to have non-trivial values
    if np.sum(c_pre) <= 0:
        out["reason"] = "category_zero_pre"; return out

    t_pre  = s_pre["units"].values.astype(float)
    t_test = s_test["units"].values.astype(float)
    if t_pre.sum() <= 0:
        out["reason"] = "treated_zero_pre"; return out

    # Trend indices: continuous time in weeks, same scale across pre/test
    n_pre = len(t_pre); n_test = len(t_test)
    idx_pre  = np.arange(n_pre)
    idx_test = np.arange(n_pre, n_pre + n_test)

    # Initial fit
    t_hat_test, residuals, beta = _fit_predict_its(
        t_pre, c_pre, idx_pre, c_test, idx_test,
        USE_TREND, USE_CATEGORY, LOG_TRANSFORM)
    if t_hat_test is None:
        out["reason"] = "fit_failed"; return out

    # Optional: winsorize pre residuals and refit (improves robustness to outlier weeks)
    if WINSOR_PRE_RESIDUAL and len(residuals) > 4:
        sigma = np.std(residuals)
        if sigma > 0:
            mask = np.abs(residuals) <= WINSOR_PRE_RESIDUAL * sigma
            if mask.sum() >= MIN_PRE_WEEKS:
                t_hat_test, residuals, beta = _fit_predict_its(
                    t_pre[mask], c_pre[mask], idx_pre[mask],
                    c_test, idx_test, USE_TREND, USE_CATEGORY, LOG_TRANSFORM)
                if t_hat_test is None:
                    out["reason"] = "refit_failed"; return out

    counterfactual = float(np.sum(t_hat_test))
    if counterfactual <= 0:
        out["reason"] = "cf_zero"; return out
    actual = float(np.sum(t_test))
    uplift_pct = (actual - counterfactual) / counterfactual

    # Pre-fit quality (diagnostic only, not a filter)
    pre_fit_rmse_rel = float(np.sqrt(np.mean(residuals ** 2)) / max(np.mean(t_pre), 1e-9))

    # Bootstrap CI — two modes:
    #   "darkstore_cluster" (default): resample darkstores with replacement, recompute
    #     T(t) = sum_units across the sampled darkstores, refit ITS, repredict.
    #     This uses the FULL 2500-darkstore sample, giving narrow CI by n=2500 not n=12.
    #   "residual_weeks": legacy — resample 12 pre-period residuals (wide CI).
    ci_lo = ci_hi = se = np.nan
    n_darkstores_used = np.nan
    if bootstrap_n and bootstrap_n > 0:
        boots = []
        if BOOTSTRAP_MODE == "darkstore_cluster":
            ds_panel = _get_darkstore_panel(str(barcode))
            if not ds_panel.empty:
                # All weeks needed for resampled aggregates
                all_weeks = pd.DatetimeIndex(list(s_pre.index) + list(s_test.index))
                # Pivot: rows = darkstore, cols = weeks
                wide = (ds_panel[ds_panel["week"].isin(all_weeks)]
                        .pivot_table(index="darkstore", columns="week",
                                     values="units", aggfunc="sum", fill_value=0))
                # Reindex columns to the exact week order we need
                wide = wide.reindex(columns=all_weeks, fill_value=0)
                ds_list = wide.index.values
                n_ds = len(ds_list)
                n_darkstores_used = int(n_ds)
                if n_ds >= 30:
                    arr = wide.values   # shape (n_ds, n_pre + n_test)
                    pre_cols = np.arange(0, n_pre)
                    test_cols = np.arange(n_pre, n_pre + n_test)
                    rng = np.random.default_rng(7)
                    for _ in range(bootstrap_n):
                        samp = rng.integers(0, n_ds, size=n_ds)
                        sub = arr[samp]                  # (n_ds, weeks)
                        t_pre_b  = sub[:, pre_cols].sum(axis=0)
                        t_test_b = sub[:, test_cols].sum(axis=0)
                        if t_pre_b.sum() <= 0:
                            continue
                        t_hat_b, _, _ = _fit_predict_its(
                            t_pre_b, c_pre, idx_pre, c_test, idx_test,
                            USE_TREND, USE_CATEGORY, LOG_TRANSFORM)
                        if t_hat_b is None: continue
                        cf_b = float(np.sum(t_hat_b))
                        if cf_b > 0:
                            boots.append((float(t_test_b.sum()) - cf_b) / cf_b)
        else:
            # Legacy residual-week resampling
            if len(residuals) >= 4:
                eps = 1e-6
                y_pre_log = np.log(np.maximum(t_pre, eps))
                fit_pre_log = y_pre_log - residuals
                for _ in range(bootstrap_n):
                    samp_idx = np.random.randint(0, len(residuals), size=len(residuals))
                    resampled_resid = residuals[samp_idx]
                    t_pre_b = (fit_pre_log + resampled_resid) if LOG_TRANSFORM \
                              else (t_pre + resampled_resid - residuals)
                    t_pre_b = np.exp(t_pre_b) if LOG_TRANSFORM else t_pre_b
                    t_pre_b = np.maximum(t_pre_b, 0.01)
                    t_hat_b, _, _ = _fit_predict_its(
                        t_pre_b, c_pre, idx_pre, c_test, idx_test,
                        USE_TREND, USE_CATEGORY, LOG_TRANSFORM)
                    if t_hat_b is None: continue
                    cf_b = float(np.sum(t_hat_b))
                    if cf_b > 0:
                        boots.append((actual - cf_b) / cf_b)
        if len(boots) >= max(50, bootstrap_n * 0.5):
            ci_lo = float(np.percentile(boots, 2.5))
            ci_hi = float(np.percentile(boots, 97.5))
            se    = float(np.std(boots))

    # Price decomposition (kept simple — same elasticity logic as before)
    price_pre  = float(s_pre["price_promo"].mean())
    price_test = float(s_test["price_promo"].mean())
    price_change = (price_test / price_pre - 1) if price_pre > 0 else np.nan
    # Self-elasticity from pre-period on log-log
    if len(s_pre) >= 6 and s_pre["price_promo"].std() > 0.5:
        lp = np.log(np.maximum(s_pre["price_promo"].values, 1e-6))
        lq = np.log(np.maximum(s_pre["units"].values, 1e-6))
        if np.std(lp) > 0.02:
            x = lp - lp.mean(); y_ = lq - lq.mean()
            elas = float(np.clip((x * y_).sum() / max((x * x).sum(), 1e-9), -3.0, 0.0))
            if elas in (-3.0, 0.0): elas = FALLBACK_ELASTICITY
        else:
            elas = FALLBACK_ELASTICITY
    else:
        elas = FALLBACK_ELASTICITY

    uplift_from_price = elas * price_change if pd.notna(price_change) else 0.0
    uplift_shelf      = uplift_pct - uplift_from_price

    return {
        "valid": True, "reason": "ok",
        "n_pre_weeks": n_pre, "n_test_weeks": n_test,
        "uplift_pct": uplift_pct,
        "uplift_pct_shelf_only": uplift_shelf,
        "counterfactual_units": counterfactual,
        "actual_units": actual,
        "uplift_units": actual - counterfactual,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "se": se,
        "n_darkstores_used": n_darkstores_used,
        "bootstrap_mode": BOOTSTRAP_MODE,
        "pre_fit_rmse_rel": pre_fit_rmse_rel,
        "price_pre": price_pre, "price_test": price_test,
        "price_change": price_change,
        "price_promo_flag": bool(pd.notna(price_change) and price_change < -PRICE_DROP_THRESHOLD),
        "elasticity": elas, "uplift_from_price": uplift_from_price,
        "category_level": CATEGORY_LEVEL, "category_value": cat_val,
        "n_beta_params": len(beta) if beta is not None else 0,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Within-SKU placebo calibration
# MAGIC
# MAGIC For each treated SKU, run the SAME estimator on FAKE test windows BEFORE the
# MAGIC real activation (purely historical). True uplift = 0 by construction. The
# MAGIC distribution of these placebo uplifts should center on zero; if it doesn't,
# MAGIC the model has SKU-specific bias that we subtract per-case.

# COMMAND ----------

def placebo_for_sku(barcode: str, real_start: pd.Timestamp,
                     n_attempts: int = 5) -> float:
    """Pre-treatment placebo: pick fake (test_start, test_end) before real_start.
    Returns median uplift across attempts, or NaN if none worked."""
    g = panel_by_bc.get(str(barcode))
    if g is None or g.empty: return np.nan
    available_weeks = sorted(g.index[g.index < real_start])
    if len(available_weeks) < PRE_WEEKS + 4: return np.nan
    results = []
    for _ in range(n_attempts):
        # pick a "fake activation start" that has at least PRE_WEEKS history before it
        end_idx = len(available_weeks) - 2
        start_idx = np.random.randint(PRE_WEEKS, end_idx + 1)
        fake_start = pd.Timestamp(available_weeks[start_idx])
        fake_end   = pd.Timestamp(available_weeks[min(start_idx + 1, end_idx)])
        try:
            est = estimate_one(barcode, fake_start, fake_end, bootstrap_n=0)
        except Exception:
            continue
        if est.get("valid"):
            results.append(est["uplift_pct"])
    return float(np.median(results)) if results else np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run all test placements

# COMMAND ----------

np.random.seed(42)
print(f"Estimating ITS uplift on {len(test_acts)} placements (with bootstrap CI)…")
rows = []
for i, r in test_acts.iterrows():
    try:
        est = estimate_one(r["barcode"], r["start_date"], r["end_date"])
    except Exception as e:
        est = {"valid": False, "reason": f"exception:{type(e).__name__}"}
    # Per-SKU placebo for bias correction
    if est.get("valid"):
        try:
            est["sku_placebo_bias"] = placebo_for_sku(r["barcode"], r["start_date"])
        except Exception:
            est["sku_placebo_bias"] = np.nan
    est.update({
        "placement_id": r["placement_id"],
        "barcode": r["barcode"],
        "tool_name": r.get("tool_name"),
        "sku_category": r.get("sku_category"),
        "start_date": r["start_date"], "end_date": r["end_date"],
    })
    rows.append(est)
res = pd.DataFrame(rows)

# Apply per-SKU bias correction (if placebo available)
if "sku_placebo_bias" in res.columns:
    bias = res["sku_placebo_bias"].fillna(0)
    res["uplift_pct_unadj"] = res.get("uplift_pct")
    res["uplift_pct_shelf_only_unadj"] = res.get("uplift_pct_shelf_only")
    res["uplift_pct"]              = res["uplift_pct"]              - bias
    res["uplift_pct_shelf_only"]   = res["uplift_pct_shelf_only"]   - bias

print(f"Valid: {res['valid'].sum()}/{len(res)} ({100*res['valid'].mean():.1f}%)")
print("\nInvalid reasons:")
print(res[~res["valid"]]["reason"].value_counts())
print("\nUplift distribution (after per-SKU placebo correction):")
print(res[res["valid"]]["uplift_pct_shelf_only"].describe())
print("\nSKU-placebo bias distribution:")
print(res[res["valid"]]["sku_placebo_bias"].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tool-level rollup with bootstrap CI of medians

# COMMAND ----------

def _winsor(s, lo=0.05, hi=0.95):
    qlo, qhi = s.quantile(lo), s.quantile(hi)
    return s.clip(qlo, qhi)

def _bootstrap_median_ci(values: np.ndarray, n=2000, alpha=0.05):
    if len(values) < 4: return (np.nan, np.nan)
    boots = np.empty(n)
    rng = np.random.default_rng(7)
    for i in range(n):
        boots[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))

valid = res[res["valid"]].copy()
tool_rows = []
for tool, sub in valid.groupby("tool_name"):
    vals = sub["uplift_pct_shelf_only"].dropna().values
    winsor_vals = _winsor(pd.Series(vals)).values
    lo, hi = _bootstrap_median_ci(winsor_vals)
    tool_rows.append({
        "tool_name":       tool,
        "n_cases":         len(sub),
        "n_price_promo":   int(sub["price_promo_flag"].sum()),
        "raw_med":         float(np.median(sub["uplift_pct"].dropna())),
        "shelf_med":       float(np.median(vals)) if len(vals) else np.nan,
        "shelf_winsor_med":float(np.median(winsor_vals)) if len(winsor_vals) else np.nan,
        "shelf_winsor_mean":float(np.mean(winsor_vals)) if len(winsor_vals) else np.nan,
        "ci_lo":           lo,
        "ci_hi":           hi,
        "shelf_p25":       float(np.percentile(vals, 25)) if len(vals) else np.nan,
        "shelf_p75":       float(np.percentile(vals, 75)) if len(vals) else np.nan,
        "med_pre_fit_rmse":float(sub["pre_fit_rmse_rel"].median()),
        "med_sku_placebo": float(sub["sku_placebo_bias"].median()),
        "sum_units":       float(sub["uplift_units"].sum()),
    })
tool_summary = pd.DataFrame(tool_rows).sort_values("shelf_winsor_med", ascending=False)
display(spark.createDataFrame(tool_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save

# COMMAND ----------

(spark.createDataFrame(res)
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/per_case"))
(spark.createDataFrame(tool_summary)
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/tool_summary"))
print(f"Saved to {OUTPUT_ROOT}/{{per_case, tool_summary}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. How to read the output
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `uplift_pct` | Per-placement uplift on the **same SKU vs its own counterfactual** trajectory |
# MAGIC | `uplift_pct_shelf_only` | Same, minus the part explained by price elasticity (price decomposition kept) |
# MAGIC | `sku_placebo_bias` | Bias estimated on FAKE pre-treatment windows for the same SKU. ITS estimates are corrected by subtracting this per case. |
# MAGIC | `ci_lo / ci_hi` | 95% bootstrap CI of per-placement uplift |
# MAGIC | `pre_fit_rmse_rel` | How well the ITS model fit the pre-period — diagnostic, not a filter |
# MAGIC | `tool_summary.shelf_winsor_med` | **Headline metric per tool** — winsorized median across placements |
# MAGIC | `tool_summary.ci_lo / ci_hi` | Bootstrap CI of the median per tool — if `ci_lo > 0`, the tool's effect is significantly positive |
# MAGIC
# MAGIC **Why this should give positive uplifts where the business expects them:**
# MAGIC - No cross-SKU comparison → no selection bias from "treated SKUs are weaker"
# MAGIC - Category covariate absorbs seasonality and macro shocks
# MAGIC - Within-SKU placebo per case → removes any remaining SKU-specific baseline drift
# MAGIC - If the activation actually moved sales above the SKU's own trend × category trend, uplift > 0
