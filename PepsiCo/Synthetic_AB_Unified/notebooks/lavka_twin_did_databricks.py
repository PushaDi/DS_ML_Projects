# Databricks notebook source
# MAGIC %md
# MAGIC # Lavka-Twin DiD v3 — Uplift estimation for Samokat marketing tools (Databricks)
# MAGIC
# MAGIC **Inputs (all Delta tables)**
# MAGIC - Samokat sales: daily, with numeric distribution column inside
# MAGIC - Lavka sales: **daily** granularity (aggregated to weekly here)
# MAGIC - Marketing / activations: Delta table with `placement_id, barcode, tool_name, start_date, end_date, ...`
# MAGIC - Test placement ids: CSV (optional — if you have a column inside activations marking test, skip the file)
# MAGIC
# MAGIC **Pipeline**
# MAGIC 1. Load + standardize → 2. Weekly panel → 3. Per-case Lavka-Twin DiD with **distribution & price decomposition** →
# MAGIC 4. Tool rollup → 5. Save Delta.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration — edit paths and column names here

# COMMAND ----------

# ---- Paths (all Delta unless noted) ----
SAMOKAT_SALES_TABLE = "/mnt/data/samokat_sales"      # daily, includes distribution column
LAVKA_SALES_TABLE   = "/mnt/data/lavka_sales"        # daily
ACTIVATIONS_TABLE   = "/mnt/data/activations"        # marketing placements
TEST_IDS_CSV        = "/mnt/data/test_placement_ids.csv"   # optional, can be ""
SCM_CONSENSUS_CSV   = ""                              # optional, sanity check against prior SCM/HBM run
OUTPUT_ROOT         = "/mnt/data/uplift_v3"

# ---- Column mapping (rename here if your schema differs) ----
SAMOKAT_COLS = {
    "barcode": "gtin",            # SKU id column in Samokat
    "date":    "date",
    "units":   "sales_quantity",
    "price":   "price",
    "city":    "city",
    "distribution": "numeric_distribution",   # <-- distribution column lives here
}
LAVKA_COLS = {
    "barcode":  "barcode",
    "date":     "date",
    "units":    "sales_quantity",
    "outlet":   "lavka_id",
}
ACTIVATIONS_COLS = {
    "placement_id": "placement_id",
    "barcode":      "barcode",
    "tool_name":    "tool_name",
    "start_date":   "start_date",
    "end_date":     "end_date",
    "sku_category": "sku_category",
    "is_test":      None,         # set to column name if activations have an is_test flag
}

# ---- Model parameters ----
PRE_WEEKS = 8
MIN_PRE_WEEKS = 4
PRICE_DROP_THRESHOLD = 0.05       # >5% drop flags a parallel price promo
FALLBACK_ELASTICITY = -1.2

print(f"Samokat: {SAMOKAT_SALES_TABLE}")
print(f"Lavka:   {LAVKA_SALES_TABLE}")
print(f"Acts:    {ACTIVATIONS_TABLE}")
print(f"Out:     {OUTPUT_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & loader
# MAGIC The loader auto-detects Delta vs Parquet vs CSV by path/format — same code works in dev (parquet) and prod (Delta).

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F, types as T
from pyspark.sql import DataFrame

spark.conf.set("spark.sql.session.timeZone", "UTC")


def _read(path: str) -> DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return spark.read.option("header", True).csv(path)
    if p.endswith(".parquet"):
        return spark.read.parquet(path)
    try:
        return spark.read.format("delta").load(path)
    except Exception:
        return spark.read.parquet(path)


def _week(col: str):
    """Snap date/timestamp to Monday-anchored ISO week start."""
    return F.date_trunc("week", F.col(col)).cast("date")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load and standardize

# COMMAND ----------

samokat_raw = _read(SAMOKAT_SALES_TABLE)
lavka_raw   = _read(LAVKA_SALES_TABLE)
acts_raw    = _read(ACTIVATIONS_TABLE)

print("Samokat schema:"); samokat_raw.printSchema()
print("Lavka schema:");   lavka_raw.printSchema()
print("Acts schema:");    acts_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Samokat weekly aggregation (with distribution in the same row)
# MAGIC
# MAGIC The distribution column is **averaged within the week** — for daily numeric distribution
# MAGIC this is the standard convention (week-mean of daily % present). Units are summed, price is averaged.

# COMMAND ----------

sc = SAMOKAT_COLS
samokat_w = (samokat_raw
    .withColumn("barcode", F.col(sc["barcode"]).cast("string"))
    .withColumn("week", _week(sc["date"]))
    .filter(F.col(sc["units"]) > 0)
    .groupBy("barcode", "week").agg(
        F.sum(sc["units"]).alias("samokat_units"),
        F.avg(sc["price"]).alias("samokat_price"),
        F.countDistinct(sc["city"]).alias("samokat_n_cities"),
        F.avg(sc["distribution"]).alias("numeric_distribution"),   # week-mean of daily distribution
    ))
display(samokat_w.orderBy("barcode", "week").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Lavka weekly aggregation (from daily)
# MAGIC
# MAGIC Two reasonable definitions of "outlets per week":
# MAGIC - **`countDistinct(outlet)` over the week** — total unique outlets touched
# MAGIC - **`avg(daily_distinct_outlets)`** — average number of selling outlets per day
# MAGIC
# MAGIC We use the second: per-day distinct count, then averaged. This better reflects "how wide
# MAGIC was Lavka's distribution during this week" and matches the per-distribution-point logic we
# MAGIC use for Samokat. Closes the conceptual gap between the two retailers.

# COMMAND ----------

lc = LAVKA_COLS
lavka_daily_outlets = (lavka_raw
    .withColumn("barcode", F.col(lc["barcode"]).cast("string"))
    .filter(F.col(lc["units"]) > 0)
    .groupBy("barcode", F.col(lc["date"]).alias("date")).agg(
        F.sum(lc["units"]).alias("daily_units"),
        F.countDistinct(lc["outlet"]).alias("daily_outlets"),
    )
    .withColumn("week", _week("date")))

lavka_w = (lavka_daily_outlets.groupBy("barcode", "week").agg(
        F.sum("daily_units").alias("lavka_units"),
        F.avg("daily_outlets").alias("lavka_n_outlets"),   # avg per-day distinct outlets
    ))
display(lavka_w.orderBy("barcode", "week").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Activations (Delta) + optional test filter

# COMMAND ----------

ac = ACTIVATIONS_COLS
acts = (acts_raw
    .withColumn("barcode",    F.col(ac["barcode"]).cast("string"))
    .withColumn("start_date", F.to_date(ac["start_date"]))
    .withColumn("end_date",   F.to_date(ac["end_date"]))
    .select(
        F.col(ac["placement_id"]).alias("placement_id"),
        "barcode",
        F.col(ac["tool_name"]).alias("tool_name"),
        F.col(ac["sku_category"]).alias("sku_category"),
        "start_date", "end_date",
        *([F.col(ac["is_test"]).alias("is_test")] if ac["is_test"] else []),
    ))

if ac["is_test"]:
    acts_test = acts.filter(F.col("is_test"))
elif TEST_IDS_CSV:
    test_ids_df = spark.read.option("header", True).csv(TEST_IDS_CSV)
    test_col = test_ids_df.columns[0]
    test_ids = [r[test_col] for r in test_ids_df.collect() if r[test_col]]
    print(f"Test placement_ids from CSV: {len(test_ids)}")
    acts_test = acts.filter(F.col("placement_id").isin(test_ids))
else:
    acts_test = acts
    print("No test filter — processing ALL activations")

print(f"Test activations: {acts_test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Unified weekly panel
# MAGIC
# MAGIC One row per `(barcode, week)`. Where distribution is missing (e.g. weeks with sales but no
# MAGIC distribution feed yet), we fall back to `samokat_n_cities` so the case isn't lost.

# COMMAND ----------

panel = (samokat_w
    .join(lavka_w, ["barcode", "week"], "left")
    .withColumn("numeric_distribution",
                F.coalesce("numeric_distribution",
                           F.col("samokat_n_cities").cast("double"))))
display(panel.orderBy("barcode", "week").limit(20))

panel_pd = panel.toPandas()
panel_pd["week"]    = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
print("Panel rows:", len(panel_pd))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Single-case estimator: Lavka-Twin DiD v3
# MAGIC
# MAGIC Three uplift outputs, each answering a different business question:
# MAGIC - `uplift_pct_raw` — gross %, the usual reporting number
# MAGIC - `uplift_pct_norm` — per-distribution-point (removes listing-expansion bias)
# MAGIC - `uplift_pct_shelf_only` — also subtracts price-discount effect via elasticity. **Main KPI.**

# COMMAND ----------

def _elasticity_pre(s_pre: pd.DataFrame) -> float:
    df = s_pre[(s_pre["samokat_units"] > 0) & (s_pre["samokat_price"] > 0)
               & (s_pre["numeric_distribution"] > 0)].copy()
    if len(df) < 4:
        return FALLBACK_ELASTICITY
    df["log_q"] = np.log(df["samokat_units"] / df["numeric_distribution"])
    df["log_p"] = np.log(df["samokat_price"])
    if df["log_p"].std() < 0.01:
        return FALLBACK_ELASTICITY
    x = df["log_p"].values - df["log_p"].mean()
    y = df["log_q"].values - df["log_q"].mean()
    beta = (x * y).sum() / (x * x).sum()
    return float(np.clip(beta, -3.0, 0.0))


def lavka_twin_did_v3(barcode: str, start: pd.Timestamp, end: pd.Timestamp,
                       panel: pd.DataFrame) -> dict:
    g = panel[panel["barcode"] == str(barcode)].set_index("week").sort_index()
    if g.empty:
        return {"valid": False, "reason": "no_data_for_sku"}

    pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
    s_pre  = g.loc[(g.index >= pre_start) & (g.index < start)]
    s_test = g.loc[(g.index >= start) & (g.index <= end)]

    if len(s_pre) < MIN_PRE_WEEKS or len(s_test) < 1:
        return {"valid": False, "reason": "insufficient_history"}
    if s_pre["lavka_units"].isna().all() or s_test["lavka_units"].isna().all():
        return {"valid": False, "reason": "no_lavka_twin"}

    # Per-distribution-point Samokat rates
    s_pre_pd  = (s_pre["samokat_units"]  / s_pre["numeric_distribution"]).mean()
    s_test_pd = (s_test["samokat_units"] / s_test["numeric_distribution"]).mean()
    # Per-outlet Lavka rates
    l_pre_po  = (s_pre["lavka_units"]    / s_pre["lavka_n_outlets"]).mean()
    l_test_po = (s_test["lavka_units"]   / s_test["lavka_n_outlets"]).mean()

    if not all(v and v > 0 for v in [s_pre_pd, l_pre_po]):
        return {"valid": False, "reason": "tiny_baseline"}

    # === DiD on normalized rates ===
    ratio_shift = l_test_po / l_pre_po
    s_exp_pd = s_pre_pd * ratio_shift
    uplift_norm = (s_test_pd - s_exp_pd) / s_exp_pd

    # Raw DiD (no normalization) for diagnostic
    s_exp_tot = s_pre["samokat_units"].mean() * (s_test["lavka_units"].mean()
                                                 / max(s_pre["lavka_units"].mean(), 1e-6))
    uplift_raw = (s_test["samokat_units"].mean() - s_exp_tot) / s_exp_tot if s_exp_tot > 0 else np.nan

    # === Price decomposition ===
    price_pre  = s_pre["samokat_price"].mean()
    price_test = s_test["samokat_price"].mean()
    price_change = (price_test / price_pre) - 1 if price_pre > 0 else np.nan
    elasticity = _elasticity_pre(s_pre)
    uplift_from_price = elasticity * price_change if pd.notna(price_change) else 0.0
    uplift_shelf = uplift_norm - uplift_from_price

    # Distribution diagnostics
    dist_pre  = s_pre["numeric_distribution"].mean()
    dist_test = s_test["numeric_distribution"].mean()
    dist_growth = (dist_test / dist_pre) - 1 if dist_pre > 0 else np.nan

    uplift_units = uplift_shelf * s_exp_pd * dist_test * len(s_test)

    return {
        "valid": True, "reason": "ok",
        "n_pre_weeks": len(s_pre), "n_test_weeks": len(s_test),
        "uplift_pct_raw": uplift_raw,
        "uplift_pct_norm": uplift_norm,
        "uplift_pct_shelf_only": uplift_shelf,
        "uplift_units_shelf_only": uplift_units,
        "dist_pre": dist_pre, "dist_test": dist_test, "dist_growth": dist_growth,
        "price_pre": price_pre, "price_test": price_test, "price_change": price_change,
        "price_promo_flag": bool(pd.notna(price_change) and price_change < -PRICE_DROP_THRESHOLD),
        "elasticity": elasticity, "uplift_from_price": uplift_from_price,
        "s_pre_pd": s_pre_pd, "s_test_pd": s_test_pd,
        "l_pre_po": l_pre_po, "l_test_po": l_test_po,
        "ratio_shift": ratio_shift,
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch run over all (test) activations

# COMMAND ----------

acts_pd = acts_test.toPandas()
acts_pd["start_date"] = pd.to_datetime(acts_pd["start_date"])
acts_pd["end_date"]   = pd.to_datetime(acts_pd["end_date"])

results = []
for _, r in acts_pd.iterrows():
    est = lavka_twin_did_v3(r["barcode"], r["start_date"], r["end_date"], panel_pd)
    est.update({
        "placement_id": r["placement_id"],
        "barcode": r["barcode"],
        "tool_name": r.get("tool_name"),
        "sku_category": r.get("sku_category"),
        "start_date": r["start_date"], "end_date": r["end_date"],
    })
    results.append(est)
res = pd.DataFrame(results)
print(f"Valid: {res['valid'].sum()}/{len(res)} ({100*res['valid'].mean():.1f}%)")
display(spark.createDataFrame(res.head(30)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tool-level rollup
# MAGIC
# MAGIC `shelf_med` is the headline KPI. Compare `raw_med` vs `shelf_med` — the gap is the part
# MAGIC of the historical uplift that was actually driven by parallel discounts or listing expansion.

# COMMAND ----------

valid = res[res["valid"]].copy()
tool_summary = (valid.groupby("tool_name")
                .agg(n_cases=("uplift_pct_raw", "size"),
                     n_price_promo=("price_promo_flag", "sum"),
                     raw_med=("uplift_pct_raw", "median"),
                     norm_med=("uplift_pct_norm", "median"),
                     shelf_med=("uplift_pct_shelf_only", "median"),
                     shelf_p25=("uplift_pct_shelf_only", lambda s: s.quantile(0.25)),
                     shelf_p75=("uplift_pct_shelf_only", lambda s: s.quantile(0.75)),
                     dist_growth_med=("dist_growth", "median"),
                     price_change_med=("price_change", "median"),
                     sum_units_shelf=("uplift_units_shelf_only", "sum"))
                .sort_values("shelf_med", ascending=False))
display(spark.createDataFrame(tool_summary.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. (Optional) Sanity check vs prior SCM / HBM consensus

# COMMAND ----------

if SCM_CONSENSUS_CSV:
    consensus = pd.read_csv(SCM_CONSENSUS_CSV) if SCM_CONSENSUS_CSV.endswith(".csv") \
                else _read(SCM_CONSENSUS_CSV).toPandas()
    consensus["barcode"] = consensus["barcode"].astype(str)
    cmp = valid.merge(
        consensus[["placement_id", "barcode", "tool_name", "scm_rel", "hbm_rel"]],
        on=["placement_id", "barcode", "tool_name"], how="inner")
    for col in ["uplift_pct_raw", "uplift_pct_norm", "uplift_pct_shelf_only"]:
        c_scm = cmp[[col, "scm_rel"]].corr().iloc[0, 1]
        c_hbm = cmp[[col, "hbm_rel"]].corr().iloc[0, 1]
        sa = ((cmp[col] > 0) == (cmp["scm_rel"] > 0)).mean()
        print(f"  {col:25s}  corr_SCM={c_scm:+.3f}  corr_HBM={c_hbm:+.3f}  sign_agree_SCM={sa:.1%}")
else:
    print("SCM_CONSENSUS_CSV empty — skipping sanity check")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save outputs (Delta)

# COMMAND ----------

(spark.createDataFrame(res)
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/per_case"))
(spark.createDataFrame(tool_summary.reset_index())
       .write.mode("overwrite").format("delta")
       .save(f"{OUTPUT_ROOT}/tool_summary"))
print(f"Wrote {OUTPUT_ROOT}/per_case and {OUTPUT_ROOT}/tool_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Reading the outputs
# MAGIC
# MAGIC | Column | Question it answers |
# MAGIC |---|---|
# MAGIC | `uplift_pct_raw` | Gross % growth — finance headline |
# MAGIC | `uplift_pct_norm` | Did existing distribution sell more? (listing-expansion removed) |
# MAGIC | **`uplift_pct_shelf_only`** | True effect of the marketing tool itself (price-discount also removed) |
# MAGIC | `price_promo_flag` | Was there a parallel ≥5% discount? (treat shelf-only cautiously when True) |
# MAGIC | `dist_growth` | How much did distribution expand during the window? Raw number overstates when >10% |
# MAGIC | `elasticity` | Per-SKU price elasticity used in the decomposition (fallback -1.2 if too noisy) |
