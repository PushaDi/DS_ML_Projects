# Databricks notebook source
# MAGIC %md
# MAGIC # Industrial Synthetic A/B for Samokat
# MAGIC
# MAGIC This is a clean, production-oriented rewrite of the SCM notebook.
# MAGIC
# MAGIC It keeps the original business task: estimate uplift for every `(placement_id, SKU)`
# MAGIC using synthetic controls. The implementation adds the safeguards normally required
# MAGIC before a synthetic A/B readout can be trusted:
# MAGIC
# MAGIC 1. Raw point-level sales as the primary outcome (`units`).
# MAGIC 2. All-channel contamination calendar for donor cleaning.
# MAGIC 3. Product-hierarchy fallback with donor coverage checks.
# MAGIC 4. Donor ranking by pre-period correlation, level gap and trend similarity.
# MAGIC 5. Ridge selection by pre-period holdout.
# MAGIC 6. Local in-space placebo for empirical p-values.
# MAGIC 7. Global random placebo calibration for structural bias correction.
# MAGIC 8. Price decomposition into total effect and shelf-only effect.
# MAGIC 9. Sensitivity readout across transformations and pre-period lengths.
# MAGIC 10. Strict `robust_pass` gate for the final business table.

# COMMAND ----------

import math
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

spark.conf.set("spark.sql.ansi.enabled", "false")
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# ---- Tables ----
DARKSTORE_SALES_TABLE = "..."   # <-- fill darkstore-level Samokat table name
PRODUCT_MASTER_TABLE  = "ecom_etl.td_product"
MEDIA_PLAN_TABLE      = "ecom_etl.marketing_activations_mp"
MEDIA_MAPPING_TABLE   = "ecom_etl.marketing_activations_sku"

TEST_IDS = pd.read_excel(
    "/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/"
    "DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/"
    "Promo plan example for Samokat PO1 1.xlsx",
    sheet_name="id размещений для Димы Х.",
)["id размещений"].tolist()

OUTPUT_ROOT = "/mnt/data/synthetic_ab_scm_industrial"

# ---- Darkstore sales columns ----
DS_COLS = {
    "barcode":      "gtin",
    "darkstore":    "warehouse_guid",
    "week":         "start_of_week",
    "units":        "sales_quantity",
    "price_base":   "price_without_promo",
    "price_promo":  "promo_price",
    "discount_pct": "discount_percent",
    "osa":          "osa_fact",
    "city":         "city_nm",
}

# ---- Product hierarchy, narrowest to widest ----
HIERARCHY = [
    "sub_brand_description",
    "brand_description",
    "brand_group_description",
    "category_description",
    "business",
]
DONOR_MATCH_ATTRS = ["brand_size"]

# ---- Core model parameters ----
PRIMARY_OUTCOME = "units"
SECONDARY_OUTCOMES = ["log_units"]
PRE_WEEKS = 8
SENSITIVITY_PRE_WEEKS = [6, 8, 10]
MIN_PRE_WEEKS = 5
MIN_TEST_WEEKS = 2
PRE_HOLDOUT_WEEKS = 2

MIN_DONORS = 4
MAX_DONORS_FOR_SOLVER = 20
RIDGE_GRID = [0.05, 0.2, 1.0, 5.0, 20.0]
DEFAULT_RIDGE = 1.0

MIN_TREATED_PRE_COVERAGE = 0.70
MIN_TREATED_TEST_COVERAGE = 0.50
MIN_DONOR_PRE_COVERAGE = 0.70
MIN_DONOR_TEST_COVERAGE = 0.50
MIN_DONOR_CORR = -0.20

FALLBACK_ELASTICITY = -1.2
PRICE_DROP_THRESHOLD = 0.05
NEGATIVE_UNITS_MODE = "clip_to_zero"  # "clip_to_zero" | "keep"
MISSING_SKU_WEEKS_AS_ZERO = True       # point-level extracts often omit zero-sale SKU-weeks
MIN_TREATED_PRE_ACTIVE_WEEKS = 3
MIN_DONOR_PRE_ACTIVE_WEEKS = 3

# ---- Robustness gates ----
MAX_PRE_FIT_REL_RMSE = 0.45
MAX_HOLDOUT_REL_RMSE = 0.35
MAX_PRE_PLACEBO_ABS = 0.15
MAX_TOP_WEIGHT = 0.75
MIN_EFFECT_SIGN_STABILITY = 0.67
MAX_LOCAL_PLACEBO_PVALUE = 0.20

GLOBAL_PLACEBO_N = 250
LOCAL_PLACEBO_MAX_DONORS = 30
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print(f"Darkstore table: {DARKSTORE_SALES_TABLE}")
print(f"Primary outcome: {PRIMARY_OUTCOME}")
print(f"Pre weeks: {PRE_WEEKS}; sensitivity: {SENSITIVITY_PRE_WEEKS}")
print(f"Ridge grid: {RIDGE_GRID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load weekly panel

# COMMAND ----------

def week_start_expr(col_name):
    return F.date_trunc("week", F.col(col_name)).cast("date")


ds_raw = spark.table(DARKSTORE_SALES_TABLE)
pm_raw = spark.table(PRODUCT_MASTER_TABLE)
mp_raw = spark.table(MEDIA_PLAN_TABLE)
mm_raw = spark.table(MEDIA_MAPPING_TABLE)
if "source_filename" in mp_raw.columns:
    mp_raw = mp_raw.drop("source_filename")

dc = DS_COLS

units_expr = F.coalesce(F.col(dc["units"]), F.lit(0.0))
if NEGATIVE_UNITS_MODE == "clip_to_zero":
    units_expr = F.when(units_expr < 0, F.lit(0.0)).otherwise(units_expr)

osa_expr = (
    F.greatest(F.coalesce(F.col(dc["osa"]), F.lit(0.0)).cast("double"), F.lit(0.0))
    if dc["osa"] in ds_raw.columns
    else F.when(F.col(dc["darkstore"]).isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0))
)

ds_prepared = (
    ds_raw
    .withColumn("barcode", F.col(dc["barcode"]).cast("string"))
    .withColumn("week", week_start_expr(dc["week"]))
    .withColumn("_units_clean", units_expr.cast("double"))
    .withColumn("_osa_clean", osa_expr)
    .withColumn("_point_present", F.when(F.col(dc["darkstore"]).isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)))
    .withColumn("_price_promo", F.col(dc["price_promo"]).cast("double"))
    .withColumn("_price_base", F.col(dc["price_base"]).cast("double"))
)

total_ws_per_week = (
    ds_prepared
    .groupBy("week")
    .agg(F.countDistinct(dc["darkstore"]).alias("total_warehouses"))
)

city_count_agg = (
    F.countDistinct(F.when(F.col("_point_present") > 0, F.col(dc["city"]))).alias("city_count")
    if dc["city"] in ds_raw.columns
    else F.lit(None).cast("double").alias("city_count")
)

panel_sdf = (
    ds_prepared
    .groupBy("barcode", "week")
    .agg(
        F.sum("_units_clean").alias("units"),
        F.sum("_osa_clean").alias("dist_osa_sum"),
        F.countDistinct(F.when(F.col("_point_present") > 0, F.col(dc["darkstore"]))).alias("dist_count"),
        city_count_agg,
        F.sum(F.col("_price_promo") * F.col("_units_clean")).alias("_promo_x_units"),
        F.sum("_units_clean").alias("_units_for_price"),
        F.avg("_price_promo").alias("_avg_price_promo"),
        F.avg("_price_base").alias("price_base"),
        F.avg(dc["discount_pct"]).alias("discount_pct"),
    )
    .join(total_ws_per_week, on="week", how="left")
    .withColumn(
        "price_promo",
        F.coalesce(F.expr("try_divide(_promo_x_units, _units_for_price)"), F.col("_avg_price_promo")),
    )
    .withColumn("dist_share", F.expr("try_divide(dist_osa_sum, total_warehouses)"))
    .withColumn("log_units", F.log1p(F.coalesce(F.col("units"), F.lit(0.0))))
    .drop("_promo_x_units", "_units_for_price", "_avg_price_promo")
)

panel_pd = panel_sdf.toPandas()
panel_pd["week"] = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
for col in ["units", "dist_osa_sum", "dist_count", "city_count", "total_warehouses"]:
    panel_pd[col] = pd.to_numeric(panel_pd[col], errors="coerce").fillna(0.0)

for col in ["log_units"]:
    panel_pd[col] = pd.to_numeric(panel_pd[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

before = len(panel_pd)
panel_pd = (
    panel_pd
    .sort_values(["barcode", "week", "units"], ascending=[True, True, False])
    .drop_duplicates(["barcode", "week"], keep="first")
    .reset_index(drop=True)
)

panel_by_bc = {}
for bc, df_bc in panel_pd.groupby("barcode", sort=False):
    df_idx = df_bc.set_index("week").sort_index()
    df_idx = df_idx[~df_idx.index.duplicated(keep="first")]
    panel_by_bc[bc] = df_idx

all_weeks = pd.DatetimeIndex(sorted(panel_pd["week"].dropna().unique()))

print(f"Panel rows: {before:,} -> {len(panel_pd):,}")
print(f"SKUs: {len(panel_by_bc):,}")
print(f"Date span: {panel_pd['week'].min()} -> {panel_pd['week'].max()}")
display(panel_sdf.orderBy("barcode", "week").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product master and activation calendars

# COMMAND ----------

pm_cols = ["gtin"] + HIERARCHY + DONOR_MATCH_ATTRS
pm_pd = pm_raw.select(pm_cols).toPandas()
pm_pd["gtin"] = pm_pd["gtin"].astype(str)
pm_pd["_non_nulls"] = pm_pd[HIERARCHY + DONOR_MATCH_ATTRS].notna().sum(axis=1)
pm_pd = (
    pm_pd
    .sort_values(["gtin", "_non_nulls"], ascending=[True, False])
    .drop_duplicates("gtin", keep="first")
    .drop(columns=["_non_nulls"])
    .reset_index(drop=True)
)
pm_lookup = pm_pd.set_index("gtin", drop=False)
pm_gtins = set(pm_pd["gtin"])

media_joined = (
    mp_raw
    .join(mm_raw, on="placement_id", how="inner")
    .withColumnRenamed("gtin", "barcode")
    .withColumn("barcode", F.col("barcode").cast("string"))
    .withColumn("start_date", F.col("start_date").cast("date"))
    .withColumn("end_date", F.col("end_date").cast("date"))
    .dropDuplicates(["placement_id", "barcode", "client"])
)

treated_sdf = (
    media_joined
    .filter(F.col("client") == "Samokat")
    .dropDuplicates(["placement_id", "barcode"])
)

all_acts_pd = media_joined.toPandas()
test_acts = treated_sdf.toPandas()
for df in [all_acts_pd, test_acts]:
    df["barcode"] = df["barcode"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

test_acts = test_acts[test_acts["placement_id"].isin(TEST_IDS)].copy()

def week_start(ts):
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


def weeks_between(start, end):
    if pd.isna(start) or pd.isna(end):
        return pd.DatetimeIndex([])
    return pd.date_range(week_start(start), week_start(end), freq="W-MON")


cal_rows = []
act_lookup_rows = []
for _, r in all_acts_pd.iterrows():
    for w in weeks_between(r["start_date"], r["end_date"]):
        cal_rows.append((r["barcode"], w))
        act_lookup_rows.append((r["barcode"], w, r.get("placement_id"), r.get("client")))

activation_calendar = set(cal_rows)
activation_week_df = pd.DataFrame(act_lookup_rows, columns=["barcode", "week", "placement_id", "client"])

print(f"Product master rows: {len(pm_pd):,}")
print(f"All activation rows: {len(all_acts_pd):,}")
print(f"Test activation rows: {len(test_acts):,}")
print(f"Activation calendar pairs: {len(activation_calendar):,}")
if "client" in all_acts_pd.columns:
    print(all_acts_pd["client"].value_counts(dropna=False).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Core helpers

# COMMAND ----------

def clean_array(x):
    arr = np.asarray(x, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def rel_rmse(y, yhat):
    y = clean_array(y)
    yhat = clean_array(yhat)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return rmse / max(abs(float(np.mean(y))), 1e-9)


def safe_corr(a, b):
    a = clean_array(a)
    b = clean_array(b)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_ratio(num, den):
    den = float(den)
    if not np.isfinite(den) or abs(den) < 1e-9:
        return np.nan
    return float(num) / den


def outcome_values(df, idx, outcome_col):
    aligned = df.reindex(idx)
    if MISSING_SKU_WEEKS_AS_ZERO:
        present = np.ones(len(idx), dtype=bool)
    else:
        present = aligned[outcome_col].notna().values
    y = aligned[outcome_col].fillna(0.0).values.astype(float)
    return clean_array(y), float(present.mean()) if len(present) else 0.0


def metric_values(df, idx, col):
    aligned = df.reindex(idx)
    return aligned[col].fillna(0.0).values.astype(float)


def solve_scm_weights(y_pre, Y_pre, ridge):
    y_pre = clean_array(y_pre)
    Y_pre = np.asarray(Y_pre, dtype=float)
    Y_pre[~np.isfinite(Y_pre)] = 0.0
    n_donors = Y_pre.shape[1]
    if n_donors == 0:
        return np.array([])

    scale = max(float(np.std(y_pre)), 1e-6)
    y_s = y_pre / scale
    Y_s = Y_pre / scale

    def obj(w):
        resid = y_s - Y_s @ w
        return float(resid @ resid + ridge * (w @ w))

    def grad(w):
        resid = y_s - Y_s @ w
        return -2 * (Y_s.T @ resid) + 2 * ridge * w

    w0 = np.full(n_donors, 1.0 / n_donors)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n_donors
    res = minimize(
        obj,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 700, "ftol": 1e-10, "disp": False},
    )
    w = np.clip(res.x if res.success else w0, 0, None)
    return w / w.sum() if w.sum() > 0 else w0


def rank_donors(y_ref, Y_ref, donors):
    y_ref = clean_array(y_ref)
    rows = []
    y_mean = max(float(np.mean(np.abs(y_ref))), 1e-9)
    x = np.arange(len(y_ref))
    y_slope = np.polyfit(x, y_ref, 1)[0] if len(y_ref) >= 2 else 0.0

    for i, bc in enumerate(donors):
        d = clean_array(Y_ref[:, i])
        corr = safe_corr(y_ref, d)
        mae_rel = float(np.mean(np.abs(y_ref - d)) / y_mean)
        d_slope = np.polyfit(x, d, 1)[0] if len(d) >= 2 else 0.0
        slope_gap = abs(y_slope - d_slope) / max(abs(y_slope), y_mean, 1e-9)
        corr_score = corr if np.isfinite(corr) else -1.0
        score = 0.65 * corr_score - 0.25 * mae_rel - 0.10 * slope_gap
        rows.append({
            "barcode": bc,
            "corr": corr,
            "mae_rel": mae_rel,
            "slope_gap_rel": slope_gap,
            "score": score,
        })

    if not rows:
        return pd.DataFrame(columns=["barcode", "corr", "mae_rel", "slope_gap_rel", "score"])
    return pd.DataFrame(rows).sort_values(["score", "corr", "mae_rel"], ascending=[False, False, True]).reset_index(drop=True)


def build_matrix(donors, idx, outcome_col):
    cols = []
    coverages = []
    for bc in donors:
        df = panel_by_bc.get(bc)
        if df is None:
            cols.append(np.zeros(len(idx)))
            coverages.append(0.0)
            continue
        y, cov = outcome_values(df, idx, outcome_col)
        cols.append(y)
        coverages.append(cov)
    if not cols:
        return np.empty((len(idx), 0)), np.array([])
    return np.column_stack(cols), np.asarray(coverages, dtype=float)


def estimate_elasticity(s_pre):
    df = s_pre[(s_pre["price_promo"] > 0) & (s_pre[PRIMARY_OUTCOME] > 0)].copy()
    if len(df) < 6:
        return FALLBACK_ELASTICITY
    df["log_q"] = np.log(df[PRIMARY_OUTCOME])
    df["log_p"] = np.log(df["price_promo"])
    if df["log_p"].std() < 0.02:
        return FALLBACK_ELASTICITY
    x = df["log_p"].values - df["log_p"].mean()
    y = df["log_q"].values - df["log_q"].mean()
    beta = safe_ratio((x * y).sum(), (x * x).sum())
    if not np.isfinite(beta):
        return FALLBACK_ELASTICITY
    beta = float(np.clip(beta, -3.0, 0.0))
    return FALLBACK_ELASTICITY if beta in (-3.0, 0.0) else beta

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Treated history diagnostic

# COMMAND ----------

# Point-level extracts often omit zero-sale SKU-weeks. This diagnostic shows
# whether invalid cases are caused by genuinely sparse treated history.
treated_history_rows = []
for _, r in test_acts.iterrows():
    bc = str(r["barcode"])
    g = panel_by_bc.get(bc)
    if g is None:
        treated_history_rows.append({
            "placement_id": r["placement_id"],
            "barcode": bc,
            "has_panel": False,
            "pre_active_weeks": 0,
            "test_active_weeks": 0,
        })
        continue
    start = week_start(r["start_date"])
    end = week_start(r["end_date"])
    pre_idx = pd.date_range(start - pd.Timedelta(weeks=PRE_WEEKS), start - pd.Timedelta(weeks=1), freq="W-MON")
    test_idx = pd.date_range(start, end, freq="W-MON")
    y_pre, _ = outcome_values(g, pre_idx, PRIMARY_OUTCOME)
    y_test, _ = outcome_values(g, test_idx, PRIMARY_OUTCOME)
    treated_history_rows.append({
        "placement_id": r["placement_id"],
        "barcode": bc,
        "has_panel": True,
        "pre_active_weeks": int(np.sum(y_pre > 0)),
        "test_active_weeks": int(np.sum(y_test > 0)),
        "pre_units": float(y_pre.sum()),
        "test_units": float(y_test.sum()),
    })

treated_history_diag = pd.DataFrame(treated_history_rows)
print("Treated history diagnostic:")
print(treated_history_diag[["has_panel", "pre_active_weeks", "test_active_weeks"]].describe(include="all").to_string())
display(spark.createDataFrame(treated_history_diag.head(50)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Donor pool

# COMMAND ----------

def is_clean_donor(bc, weeks):
    return not any((bc, w) in activation_calendar for w in weeks)


def donor_coverage_ok(bc, pre_idx, test_idx, outcome_col):
    df = panel_by_bc.get(bc)
    if df is None:
        return False
    y_pre, pre_cov = outcome_values(df, pre_idx, outcome_col)
    _, test_cov = outcome_values(df, test_idx, outcome_col)
    pre_active_weeks = int(np.sum(y_pre > 0))
    return (
        pre_cov >= MIN_DONOR_PRE_COVERAGE
        and test_cov >= MIN_DONOR_TEST_COVERAGE
        and pre_active_weeks >= MIN_DONOR_PRE_ACTIVE_WEEKS
    )


def find_donor_pool(treated_bc, start, end, outcome_col, pre_weeks):
    if treated_bc not in pm_gtins:
        return {"donors": [], "level": None, "reason": "treated_not_in_product_master"}
    if treated_bc not in panel_by_bc:
        return {"donors": [], "level": None, "reason": "treated_not_in_panel"}

    start = week_start(start)
    end = week_start(end)
    pre_idx = pd.date_range(start - pd.Timedelta(weeks=pre_weeks), start - pd.Timedelta(weeks=1), freq="W-MON")
    test_idx = pd.date_range(start, end, freq="W-MON")
    clean_weeks = list(pre_idx) + list(test_idx)

    treated_row = pm_lookup.loc[treated_bc]
    treated_size = treated_row.get("brand_size")

    for level in HIERARCHY:
        level_val = treated_row.get(level)
        if pd.isna(level_val):
            continue
        candidates = pm_pd.loc[
            (pm_pd[level] == level_val) & (pm_pd["gtin"] != treated_bc),
            "gtin",
        ].astype(str).tolist()
        candidates = [bc for bc in candidates if bc in panel_by_bc]
        candidates = [bc for bc in candidates if is_clean_donor(bc, clean_weeks)]
        candidates = [bc for bc in candidates if donor_coverage_ok(bc, pre_idx, test_idx, outcome_col)]

        if treated_size is not None and not pd.isna(treated_size):
            size_matched = [bc for bc in candidates if pm_lookup.loc[bc].get("brand_size") == treated_size]
            if len(size_matched) >= MIN_DONORS:
                return {"donors": size_matched, "level": f"{level}+brand_size", "reason": "ok"}
        if len(candidates) >= MIN_DONORS:
            return {"donors": candidates, "level": level, "reason": "ok"}

    return {"donors": [], "level": None, "reason": "no_clean_donor_pool"}


def treated_pre_contamination(barcode, placement_id, start, pre_idx):
    rows = activation_week_df[
        (activation_week_df["barcode"] == str(barcode))
        & (activation_week_df["week"].isin(pre_idx))
    ]
    return int(len(rows.drop_duplicates(["placement_id", "week"])))


def treated_concurrent_count(barcode, placement_id, test_idx):
    rows = activation_week_df[
        (activation_week_df["barcode"] == str(barcode))
        & (activation_week_df["week"].isin(test_idx))
    ]
    if "placement_id" in rows.columns:
        rows = rows[rows["placement_id"] != placement_id]
    return int(len(rows.drop_duplicates(["placement_id", "week"])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Estimator

# COMMAND ----------

def choose_ridge(y_train, Y_train, y_hold, Y_hold):
    if len(y_hold) == 0 or Y_hold.shape[1] == 0:
        return DEFAULT_RIDGE, np.nan, np.nan

    rows = []
    for ridge in RIDGE_GRID:
        w = solve_scm_weights(y_train, Y_train, ridge)
        pred = Y_hold @ w
        holdout_rmse = rel_rmse(y_hold, pred)
        holdout_placebo = safe_ratio(y_hold.sum() - pred.sum(), pred.sum())
        score = holdout_rmse + 0.50 * abs(holdout_placebo if np.isfinite(holdout_placebo) else 9.9) + 0.05 * float(w.max())
        rows.append((ridge, score, holdout_rmse, holdout_placebo))
    best = sorted(rows, key=lambda x: x[1])[0]
    return float(best[0]), float(best[2]), float(best[3])


def compute_local_placebo(donors, pre_idx, test_idx, outcome_col, ridge, max_donors=LOCAL_PLACEBO_MAX_DONORS):
    placebo = []
    selected = donors[:max_donors]
    for fake_bc in selected:
        donor_minus = [d for d in selected if d != fake_bc]
        if len(donor_minus) < MIN_DONORS:
            continue
        fake_df = panel_by_bc.get(fake_bc)
        y_pre, pre_cov = outcome_values(fake_df, pre_idx, outcome_col)
        y_test, test_cov = outcome_values(fake_df, test_idx, outcome_col)
        if pre_cov < MIN_DONOR_PRE_COVERAGE or test_cov < MIN_DONOR_TEST_COVERAGE:
            continue
        Y_pre, _ = build_matrix(donor_minus, pre_idx, outcome_col)
        Y_test, _ = build_matrix(donor_minus, test_idx, outcome_col)
        if Y_pre.shape[1] < MIN_DONORS:
            continue
        w = solve_scm_weights(y_pre, Y_pre, ridge)
        synth_test = Y_test @ w
        eff = safe_ratio(y_test.sum() - synth_test.sum(), synth_test.sum())
        if np.isfinite(eff):
            placebo.append(eff)
    return np.asarray(placebo, dtype=float)


def estimate_one(row, outcome_col=PRIMARY_OUTCOME, pre_weeks=PRE_WEEKS, run_local_placebo=True):
    barcode = str(row["barcode"])
    placement_id = row.get("placement_id")
    start = week_start(row["start_date"])
    end = week_start(row["end_date"])

    out = {
        "valid": False,
        "reason": "init",
        "placement_id": placement_id,
        "barcode": barcode,
        "tool_name": row.get("tool_name"),
        "sku_category": row.get("sku_category"),
        "start_date": start,
        "end_date": end,
        "outcome_col": outcome_col,
        "pre_weeks_config": pre_weeks,
        "robust_pass": False,
    }

    g = panel_by_bc.get(barcode)
    if g is None or g.empty:
        out["reason"] = "treated_no_panel_data"
        return out

    pre_idx = pd.date_range(start - pd.Timedelta(weeks=pre_weeks), start - pd.Timedelta(weeks=1), freq="W-MON")
    test_idx = pd.date_range(start, end, freq="W-MON")
    if len(test_idx) < MIN_TEST_WEEKS:
        out["reason"] = "test_window_too_short"
        return out

    y_pre, treated_pre_cov = outcome_values(g, pre_idx, outcome_col)
    y_test, treated_test_cov = outcome_values(g, test_idx, outcome_col)
    treated_pre_active_weeks = int(np.sum(y_pre > 0))
    treated_test_active_weeks = int(np.sum(y_test > 0))
    if len(pre_idx) < MIN_PRE_WEEKS or treated_pre_cov < MIN_TREATED_PRE_COVERAGE:
        out.update({"reason": "insufficient_treated_pre", "treated_pre_coverage": treated_pre_cov})
        return out
    if treated_test_cov < MIN_TREATED_TEST_COVERAGE:
        out.update({"reason": "insufficient_treated_test", "treated_test_coverage": treated_test_cov})
        return out
    if treated_pre_active_weeks < MIN_TREATED_PRE_ACTIVE_WEEKS:
        out.update({
            "reason": "insufficient_treated_pre_active_weeks",
            "treated_pre_active_weeks": treated_pre_active_weeks,
            "treated_pre_coverage": treated_pre_cov,
        })
        return out
    if np.mean(y_pre) <= 0:
        out["reason"] = "treated_zero_pre"
        return out

    pre_contam = treated_pre_contamination(barcode, placement_id, start, pre_idx)
    concurrent_count = treated_concurrent_count(barcode, placement_id, test_idx)

    pool = find_donor_pool(barcode, start, end, outcome_col, pre_weeks)
    if not pool["donors"]:
        out.update({"reason": pool["reason"], "level": pool["level"]})
        return out

    donors = pool["donors"]
    holdout_weeks = PRE_HOLDOUT_WEEKS if len(pre_idx) >= (MIN_PRE_WEEKS + PRE_HOLDOUT_WEEKS) else 0
    fit_idx = pre_idx[:-holdout_weeks] if holdout_weeks else pre_idx
    hold_idx = pre_idx[-holdout_weeks:] if holdout_weeks else pd.DatetimeIndex([])

    y_fit, _ = outcome_values(g, fit_idx, outcome_col)
    Y_fit_raw, _ = build_matrix(donors, fit_idx, outcome_col)
    ranked = rank_donors(y_fit, Y_fit_raw, donors)
    filtered = ranked[ranked["corr"].fillna(-1.0) >= MIN_DONOR_CORR]
    if len(filtered) >= MIN_DONORS:
        ranked = filtered
    ranked = ranked.head(MAX_DONORS_FOR_SOLVER)
    donors = ranked["barcode"].astype(str).tolist()

    if len(donors) < MIN_DONORS:
        out["reason"] = "too_few_ranked_donors"
        return out

    Y_pre, donor_pre_cov = build_matrix(donors, pre_idx, outcome_col)
    Y_test, donor_test_cov = build_matrix(donors, test_idx, outcome_col)
    keep = (donor_pre_cov >= MIN_DONOR_PRE_COVERAGE) & (donor_test_cov >= MIN_DONOR_TEST_COVERAGE) & (Y_pre.mean(axis=0) > 0)
    if keep.sum() < MIN_DONORS:
        out["reason"] = "too_few_donors_after_coverage"
        return out
    Y_pre, Y_test = Y_pre[:, keep], Y_test[:, keep]
    donors = [d for d, k in zip(donors, keep) if k]
    ranked = ranked[ranked["barcode"].isin(donors)]

    y_train = y_pre[:-holdout_weeks] if holdout_weeks else y_pre
    Y_train = Y_pre[:-holdout_weeks, :] if holdout_weeks else Y_pre
    y_hold = y_pre[-holdout_weeks:] if holdout_weeks else np.array([])
    Y_hold = Y_pre[-holdout_weeks:, :] if holdout_weeks else np.empty((0, Y_pre.shape[1]))
    ridge, holdout_rel_rmse, pre_placebo = choose_ridge(y_train, Y_train, y_hold, Y_hold)

    w = solve_scm_weights(y_pre, Y_pre, ridge)
    synth_pre = Y_pre @ w
    synth_test = Y_test @ w
    if synth_test.sum() <= 0:
        out["reason"] = "synth_zero_test"
        return out

    effect_raw = safe_ratio(y_test.sum() - synth_test.sum(), synth_test.sum())
    pre_fit = rel_rmse(y_pre, synth_pre)
    top_weight = float(w.max()) if len(w) else np.nan
    entropy = float(-np.sum(w[w > 0] * np.log(w[w > 0]))) if (w > 0).any() else np.nan

    s_pre = g.reindex(pre_idx)
    s_test = g.reindex(test_idx)
    price_pre = float(s_pre["price_promo"].mean())
    price_test = float(s_test["price_promo"].mean())
    price_change = (price_test / price_pre - 1) if price_pre > 0 else np.nan
    elasticity = estimate_elasticity(s_pre)
    uplift_from_price = elasticity * price_change if np.isfinite(price_change) else 0.0
    effect_shelf = effect_raw - uplift_from_price

    treated_test_units = metric_values(g, test_idx, "units").sum()
    active_points_pre = float(s_pre["dist_count"].mean()) if "dist_count" in s_pre.columns else np.nan
    active_points_test = float(s_test["dist_count"].mean()) if "dist_count" in s_test.columns else np.nan
    active_point_growth = (active_points_test / active_points_pre - 1) if active_points_pre > 0 else np.nan

    if outcome_col == "units":
        expected_units = float(synth_test.sum())
    else:
        expected_units = np.nan
    uplift_units_shelf = effect_shelf * expected_units if np.isfinite(expected_units) else np.nan

    local_placebo = np.array([])
    local_pvalue = np.nan
    placebo_p10 = np.nan
    placebo_p90 = np.nan
    if run_local_placebo:
        local_placebo = compute_local_placebo(donors, pre_idx, test_idx, outcome_col, ridge)
        if len(local_placebo) >= MIN_DONORS:
            local_pvalue = float((1 + np.sum(np.abs(local_placebo) >= abs(effect_raw))) / (len(local_placebo) + 1))
            placebo_p10 = float(np.percentile(local_placebo, 10))
            placebo_p90 = float(np.percentile(local_placebo, 90))

    out.update({
        "valid": True,
        "reason": "ok",
        "level": pool["level"],
        "n_pre_weeks": len(pre_idx),
        "n_test_weeks": len(test_idx),
        "treated_pre_coverage": treated_pre_cov,
        "treated_test_coverage": treated_test_cov,
        "treated_pre_active_weeks": treated_pre_active_weeks,
        "treated_test_active_weeks": treated_test_active_weeks,
        "treated_pre_contamination_weeks": pre_contam,
        "treated_concurrent_activation_weeks": concurrent_count,
        "n_donors_candidate": len(pool["donors"]),
        "n_donors_used": len(donors),
        "donors_used": ",".join(donors[:30]),
        "donor_corr_median": float(ranked["corr"].median()) if len(ranked) else np.nan,
        "donor_mae_rel_median": float(ranked["mae_rel"].median()) if len(ranked) else np.nan,
        "ridge_selected": ridge,
        "pre_fit_rel_rmse": pre_fit,
        "holdout_rel_rmse": holdout_rel_rmse,
        "pre_placebo_uplift": pre_placebo,
        "top_weight": top_weight,
        "weight_entropy": entropy,
        "uplift_pct_raw": effect_raw,
        "uplift_pct_shelf_only": effect_shelf,
        "uplift_units_shelf_only": uplift_units_shelf,
        "treated_test_units": float(treated_test_units),
        "expected_test_units": expected_units,
        "synth_pre_total": float(synth_pre.sum()),
        "synth_test_total": float(synth_test.sum()),
        "treated_pre_total": float(y_pre.sum()),
        "treated_test_total": float(y_test.sum()),
        "local_placebo_n": int(len(local_placebo)),
        "local_placebo_pvalue": local_pvalue,
        "local_placebo_p10": placebo_p10,
        "local_placebo_p90": placebo_p90,
        "price_pre": price_pre,
        "price_test": price_test,
        "price_change": price_change,
        "price_promo_flag": bool(np.isfinite(price_change) and price_change < -PRICE_DROP_THRESHOLD),
        "elasticity": elasticity,
        "uplift_from_price": uplift_from_price,
        "active_points_pre": active_points_pre,
        "active_points_test": active_points_test,
        "active_point_growth": active_point_growth,
    })
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Global placebo calibration

# COMMAND ----------

activated_barcodes = {bc for bc, _ in activation_calendar}
never_activated = [bc for bc in panel_by_bc.keys() if bc not in activated_barcodes and bc in pm_gtins]
panel_weeks_sorted = list(all_weeks)

global_placebo_rows = []
if len(panel_weeks_sorted) > PRE_WEEKS + 8 and len(never_activated) >= 30:
    for i in range(GLOBAL_PLACEBO_N):
        bc = str(np.random.choice(never_activated))
        start_pos = np.random.randint(PRE_WEEKS, len(panel_weeks_sorted) - 4)
        duration = np.random.randint(MIN_TEST_WEEKS, 5)
        fake_start = pd.Timestamp(panel_weeks_sorted[start_pos])
        fake_end = pd.Timestamp(panel_weeks_sorted[min(start_pos + duration - 1, len(panel_weeks_sorted) - 1)])
        fake_row = {
            "placement_id": f"placebo_{i}",
            "barcode": bc,
            "tool_name": "GLOBAL_PLACEBO",
            "sku_category": None,
            "start_date": fake_start,
            "end_date": fake_end,
        }
        est = estimate_one(fake_row, outcome_col=PRIMARY_OUTCOME, pre_weeks=PRE_WEEKS, run_local_placebo=False)
        if est.get("valid"):
            global_placebo_rows.append(est)

global_placebo = pd.DataFrame(global_placebo_rows)
if len(global_placebo):
    GLOBAL_BIAS_RAW = float(global_placebo["uplift_pct_raw"].median())
    GLOBAL_BIAS_SHELF = float(global_placebo["uplift_pct_shelf_only"].median())
    print(f"Global placebo valid: {len(global_placebo):,}/{GLOBAL_PLACEBO_N:,}")
    print(f"Global raw bias:   {GLOBAL_BIAS_RAW:+.4f}")
    print(f"Global shelf bias: {GLOBAL_BIAS_SHELF:+.4f}")
    print(global_placebo["uplift_pct_raw"].describe())
else:
    GLOBAL_BIAS_RAW = 0.0
    GLOBAL_BIAS_SHELF = 0.0
    print("Global placebo skipped or no valid placebo cases.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run target placements

# COMMAND ----------

results = []
for _, r in test_acts.iterrows():
    try:
        results.append(estimate_one(r, outcome_col=PRIMARY_OUTCOME, pre_weeks=PRE_WEEKS, run_local_placebo=True))
    except Exception as e:
        results.append({
            "valid": False,
            "reason": f"exception:{type(e).__name__}:{str(e)[:160]}",
            "placement_id": r.get("placement_id"),
            "barcode": str(r.get("barcode")),
            "tool_name": r.get("tool_name"),
            "sku_category": r.get("sku_category"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
        })

res = pd.DataFrame(results)

if len(res) and "uplift_pct_raw" in res.columns:
    res["uplift_pct_raw_unadjusted"] = res["uplift_pct_raw"]
    res["uplift_pct_shelf_only_unadjusted"] = res["uplift_pct_shelf_only"]
    res.loc[res["valid"], "uplift_pct_raw"] = res.loc[res["valid"], "uplift_pct_raw"] - GLOBAL_BIAS_RAW
    res.loc[res["valid"], "uplift_pct_shelf_only"] = res.loc[res["valid"], "uplift_pct_shelf_only"] - GLOBAL_BIAS_SHELF

print(f"Valid: {res['valid'].sum()}/{len(res)} ({100 * res['valid'].mean():.1f}%)")
print("Invalid reasons:")
print(res.loc[~res["valid"], "reason"].value_counts().to_string())
print("Pre-fit:")
print(res.loc[res["valid"], "pre_fit_rel_rmse"].describe().to_string())
display(spark.createDataFrame(res.head(50)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Sensitivity checks

# COMMAND ----------

sensitivity_rows = []
valid_keys = res.loc[res["valid"], ["placement_id", "barcode"]].drop_duplicates()
valid_key_set = set(map(tuple, valid_keys.values.tolist()))

for _, r in test_acts.iterrows():
    key = (r["placement_id"], str(r["barcode"]))
    if key not in valid_key_set:
        continue
    effects = []
    for outcome_col in [PRIMARY_OUTCOME] + SECONDARY_OUTCOMES:
        for pre_weeks in SENSITIVITY_PRE_WEEKS:
            try:
                est = estimate_one(r, outcome_col=outcome_col, pre_weeks=pre_weeks, run_local_placebo=False)
                if est.get("valid") and np.isfinite(est.get("uplift_pct_shelf_only", np.nan)):
                    effects.append(est["uplift_pct_shelf_only"])
                    sensitivity_rows.append({
                        "placement_id": r["placement_id"],
                        "barcode": str(r["barcode"]),
                        "outcome_col": outcome_col,
                        "pre_weeks": pre_weeks,
                        "valid": True,
                        "uplift_pct_shelf_only": est["uplift_pct_shelf_only"],
                        "pre_fit_rel_rmse": est["pre_fit_rel_rmse"],
                        "holdout_rel_rmse": est["holdout_rel_rmse"],
                    })
            except Exception as e:
                sensitivity_rows.append({
                    "placement_id": r["placement_id"],
                    "barcode": str(r["barcode"]),
                    "outcome_col": outcome_col,
                    "pre_weeks": pre_weeks,
                    "valid": False,
                    "reason": f"{type(e).__name__}:{str(e)[:100]}",
                })

sensitivity = pd.DataFrame(sensitivity_rows)
if len(sensitivity):
    sens_summary = (
        sensitivity[sensitivity["valid"]]
        .groupby(["placement_id", "barcode"])
        .agg(
            sensitivity_n=("uplift_pct_shelf_only", "size"),
            sensitivity_med=("uplift_pct_shelf_only", "median"),
            sensitivity_p25=("uplift_pct_shelf_only", lambda s: s.quantile(0.25)),
            sensitivity_p75=("uplift_pct_shelf_only", lambda s: s.quantile(0.75)),
            sign_stability=("uplift_pct_shelf_only", lambda s: max((s > 0).mean(), (s < 0).mean())),
        )
        .reset_index()
    )
    res = res.merge(sens_summary, on=["placement_id", "barcode"], how="left")
else:
    res["sensitivity_n"] = 0
    res["sign_stability"] = np.nan

display(spark.createDataFrame(sensitivity.head(50))) if len(sensitivity) else print("No sensitivity rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Robust readout

# COMMAND ----------

def robust_gate(row):
    if not bool(row.get("valid", False)):
        return False
    checks = [
        row.get("pre_fit_rel_rmse", np.inf) <= MAX_PRE_FIT_REL_RMSE,
        row.get("holdout_rel_rmse", np.inf) <= MAX_HOLDOUT_REL_RMSE,
        abs(row.get("pre_placebo_uplift", np.inf)) <= MAX_PRE_PLACEBO_ABS,
        row.get("top_weight", np.inf) <= MAX_TOP_WEIGHT,
        row.get("n_donors_used", 0) >= MIN_DONORS,
        row.get("sign_stability", 0.0) >= MIN_EFFECT_SIGN_STABILITY,
    ]
    pvalue = row.get("local_placebo_pvalue", np.nan)
    if np.isfinite(pvalue):
        checks.append(pvalue <= MAX_LOCAL_PLACEBO_PVALUE)
    return bool(all(checks))


res["robust_pass"] = res.apply(robust_gate, axis=1)
res["needs_manual_review"] = (
    (res["valid"])
    & (
        (res["treated_concurrent_activation_weeks"].fillna(0) > 0)
        | (res["treated_pre_contamination_weeks"].fillna(0) > 0)
        | (res["price_promo_flag"].fillna(False))
        | (res["level"].isin(["category_description", "business"]))
    )
)

final = res[res["robust_pass"]].copy()

print(f"Valid:  {res['valid'].sum()}/{len(res)}")
print(f"Robust: {final.shape[0]}/{len(res)}")
print("Robust by tool:")
print(final["tool_name"].value_counts(dropna=False).to_string())
display(spark.createDataFrame(final.head(50))) if len(final) else print("No robust cases under current thresholds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Rollups

# COMMAND ----------

def winsor(s, lo=0.05, hi=0.95):
    if len(s) == 0:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


if len(final):
    tool_summary = (
        final
        .groupby("tool_name")
        .agg(
            n_cases=("uplift_pct_shelf_only", "size"),
            n_manual_review=("needs_manual_review", "sum"),
            n_price_promo=("price_promo_flag", "sum"),
            shelf_med=("uplift_pct_shelf_only", "median"),
            shelf_winsor_mean=("uplift_pct_shelf_only", lambda s: winsor(s).mean()),
            shelf_p25=("uplift_pct_shelf_only", lambda s: s.quantile(0.25)),
            shelf_p75=("uplift_pct_shelf_only", lambda s: s.quantile(0.75)),
            raw_med=("uplift_pct_raw", "median"),
            units_shelf_sum=("uplift_units_shelf_only", "sum"),
            local_pvalue_med=("local_placebo_pvalue", "median"),
            sign_stability_med=("sign_stability", "median"),
            pre_fit_med=("pre_fit_rel_rmse", "median"),
            holdout_fit_med=("holdout_rel_rmse", "median"),
            donors_med=("n_donors_used", "median"),
            top_weight_med=("top_weight", "median"),
            active_point_growth_med=("active_point_growth", "median"),
            price_change_med=("price_change", "median"),
        )
        .reset_index()
        .sort_values(["shelf_winsor_mean", "n_cases"], ascending=[False, False])
    )
else:
    tool_summary = pd.DataFrame()

display(spark.createDataFrame(tool_summary)) if len(tool_summary) else print("No tool summary: final is empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save outputs

# COMMAND ----------

def write_delta_if_not_empty(df, path):
    if df is None or len(df) == 0:
        print(f"Skip empty output: {path}")
        return
    spark.createDataFrame(df).write.mode("overwrite").format("delta").save(path)
    print(f"Saved: {path}")


write_delta_if_not_empty(res, f"{OUTPUT_ROOT}/per_case_all")
write_delta_if_not_empty(final, f"{OUTPUT_ROOT}/per_case_robust")
write_delta_if_not_empty(tool_summary, f"{OUTPUT_ROOT}/tool_summary")
write_delta_if_not_empty(global_placebo, f"{OUTPUT_ROOT}/global_placebo")
write_delta_if_not_empty(sensitivity, f"{OUTPUT_ROOT}/sensitivity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Output guide
# MAGIC
# MAGIC Main table: `{OUTPUT_ROOT}/per_case_robust`
# MAGIC
# MAGIC Main business KPI:
# MAGIC - `uplift_pct_shelf_only`: bias-corrected, point-level, price-decomposed uplift.
# MAGIC - `uplift_units_shelf_only`: absolute shelf-only units on the raw point-level sales scale.
# MAGIC
# MAGIC Trust diagnostics:
# MAGIC - `robust_pass`: case is included in final readout.
# MAGIC - `needs_manual_review`: result can be used, but should be inspected before a presentation.
# MAGIC - `pre_fit_rel_rmse`: full pre-period fit error.
# MAGIC - `holdout_rel_rmse`: fit on held-out pre weeks.
# MAGIC - `pre_placebo_uplift`: fake uplift on held-out pre weeks; should be near zero.
# MAGIC - `local_placebo_pvalue`: in-space placebo p-value from the selected donor pool.
# MAGIC - `sign_stability`: sign agreement across outcome and pre-period sensitivity variants.
# MAGIC - `top_weight`: dominance of the largest donor.
# MAGIC - `level`: hierarchy level used for donors; wider levels are less comparable.
# MAGIC - `active_point_growth`: diagnostic only; it is not part of the primary adjustment.
# MAGIC
# MAGIC Interpretation rule:
# MAGIC - Use `tool_summary` for business conclusions.
# MAGIC - Use per-SKU results only after checking diagnostics and `needs_manual_review`.
# MAGIC - If robust count is low for a tool, the correct conclusion is "insufficient clean evidence", not zero effect.
