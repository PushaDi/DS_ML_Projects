# Databricks notebook source
# MAGIC %md
# MAGIC # Lavka-Twin DiD v4 — distribution-correct + Lavka-promo-filtered
# MAGIC
# MAGIC **Changes vs v3** (your last run):
# MAGIC 1. **Lavka-promo filter** — недели, когда в самой Лавке шла активация по тому же SKU,
# MAGIC    исключаются из twin'а. Источник: `media_plan` с `client='Yandex.Lavka'`.
# MAGIC 2. **Samokat distribution = SUM по городам**, а не AVG. `warehouse_count` —
# MAGIC    это число складов в городе-неделе; их нужно складывать, чтобы получить
# MAGIC    total warehouses, держащих SKU.
# MAGIC 3. **Условный sum для units** (а не ранний `filter(units > 0)`) — недели
# MAGIC    «есть на полке, но не продалось» больше не выпадают из panel'а.
# MAGIC 4. **Weighted average price** (взвешенная по продажам) вместо простого AVG —
# MAGIC    цена не перекашивается городами с минимальными продажами.
# MAGIC 5. **Lavka distribution diagnostic** — печатает гранулярность исходной таблицы,
# MAGIC    чтобы убедиться, что двойной AVG корректен для вашей схемы.
# MAGIC
# MAGIC **Inputs (all Delta tables)**
# MAGIC - Samokat sales: **weekly per city** (`start_of_week` granularity), includes `warehouse_count`
# MAGIC - Lavka sales: daily, aggregated to weekly here
# MAGIC - Marketing / activations: Delta, contains both Samokat and Lavka activations (client column)

# COMMAND ----------

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
path_components = notebook_path.split("/")
team_folder = path_components[2]
import sys
sys.path.append(f"/Workspace/eperfectstore-prod/{team_folder}/notebooks/eperfectstore-prod/e-com/COMMON_FUNCTIONS_AND_CONSTANTS_FOLDER/")
import common_functions_and_constants as CF
from udf_functions import *

# COMMAND ----------

import libify
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.storagelevel import StorageLevel
from datetime import datetime, timedelta
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os, re, warnings
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from scipy.spatial.distance import euclidean

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Marketing / activations (keeps Lavka activations too — needed for cleaning)

# COMMAND ----------

media_plan = (spark.table('ecom_etl.marketing_activations_mp')
              .filter(F.col('client').isin(['Samokat', 'Yandex.Lavka']))
              .drop('source_filename'))

media_mapping = spark.table('ecom_etl.marketing_activations_sku')

media_plan_mapped = (
    media_plan
    .join(media_mapping, on='placement_id', how='inner')
    .withColumn('start_date', F.col('start_date').cast('date'))
    .withColumn('end_date',   F.col('end_date').cast('date'))
)

mkt_final = (
    media_plan_mapped
    .filter(F.col('client') == 'Samokat')          # only Samokat activations are evaluated
    .withColumnRenamed('gtin', 'barcode')
    .dropDuplicates(subset=['barcode', 'placement_id'])
)

# Lavka activations as contamination mask — used to clean the twin
lavka_acts = (media_plan_mapped
    .filter(F.col('client') == 'Yandex.Lavka')
    .withColumnRenamed('gtin', 'barcode')
    .select('barcode', 'start_date', 'end_date')
    .dropna())

print(f"Samokat activations: {mkt_final.count()}")
print(f"Lavka activations (used as contamination mask): {lavka_acts.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Paths & column mapping

# COMMAND ----------

SAMOKAT_SALES_TABLE = "ecom_etl.ds_samokat_availability_metrics_city"   # weekly per city
LAVKA_SALES_TABLE   = "ecom_etl.ds_sim_inference_data"                  # daily
TEST_IDS_CSV = pd.read_excel(
    '/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/'
    'DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/'
    'Promo plan example for Samokat PO1 1.xlsx',
    sheet_name="id размещений для Димы Х.")['id размещений'].tolist()

SAMOKAT_COLS = {
    "barcode": "gtin",
    "date":    "start_of_week",
    "units":   "sales_quantity",
    "price":   "mode_promo_price",
    "city":    "city_nm",
    "distribution": "warehouse_count",
}
LAVKA_COLS = {
    "barcode":     "barcode",
    "date":        "date",
    "units":       "sales_quantity",
    "distribution": "distribution",
}
ACTIVATIONS_COLS = {
    "placement_id": "placement_id",
    "barcode":      "barcode",
    "tool_name":    "tool_name",
    "start_date":   "start_date",
    "end_date":     "end_date",
    "sku_category": "sku_category",
    "is_test":      None,
}

# Model params
PRE_WEEKS = 8
MIN_PRE_WEEKS = 4
MIN_TEST_WEEKS = 2
PRICE_DROP_THRESHOLD = 0.05
FALLBACK_ELASTICITY = -1.2
OUTPUT_ROOT = "/mnt/data/uplift_v4"   # adjust

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Imports & helpers

# COMMAND ----------

from pyspark.sql import functions as F, types as T
from pyspark.sql import DataFrame

spark.conf.set("spark.sql.session.timeZone", "UTC")

def _week(col: str):
    """Snap date/timestamp to Monday-anchored ISO week start."""
    return F.date_trunc("week", F.col(col)).cast("date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load raw

# COMMAND ----------

samokat_raw = spark.table(SAMOKAT_SALES_TABLE)
lavka_raw   = spark.table(LAVKA_SALES_TABLE)

print("Samokat schema:"); samokat_raw.printSchema()
print("Lavka schema:");   lavka_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Samokat weekly aggregation — FIXED
# MAGIC
# MAGIC - `warehouse_count`: **SUM** across cities (each city-week row contributes its warehouses)
# MAGIC - `units`: **conditional sum** — keeps zero-sale weeks alive
# MAGIC - `price`: **weighted by units** — price avg from cities with actual sales

# COMMAND ----------

sc = SAMOKAT_COLS
samokat_w = (samokat_raw
    .withColumn("barcode", F.col(sc["barcode"]).cast("string"))
    .withColumn("week", _week(sc["date"]))
    .groupBy("barcode", "week").agg(
        F.sum(F.coalesce(F.col(sc["units"]), F.lit(0))).alias("samokat_units"),
        # weighted price: sum(price*units)/sum(units); fallback to plain avg if units=0
        F.coalesce(
            F.sum(F.col(sc["price"]) * F.col(sc["units"])) /
                F.when(F.sum(F.col(sc["units"])) > 0, F.sum(F.col(sc["units"]))),
            F.avg(sc["price"])
        ).alias("samokat_price"),
        F.countDistinct(F.when(F.col(sc["units"]) > 0, F.col(sc["city"]))).alias("samokat_n_cities"),
        F.sum(F.coalesce(F.col(sc["distribution"]), F.lit(0))).alias("numeric_distribution"),
    ))
display(samokat_w.orderBy("barcode", "week").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Lavka — diagnostic + weekly aggregation
# MAGIC
# MAGIC First we check how many rows are at the `(barcode, date)` level. If it's exactly 1
# MAGIC per pair → daily is the bottom granularity and double-avg is fine.
# MAGIC If >1 → we need sum-then-avg (rows are split by city/region).

# COMMAND ----------

lc = LAVKA_COLS
rows_per_day = (lavka_raw
    .groupBy(F.col(lc["barcode"]).alias("barcode"), F.col(lc["date"]).alias("date"))
    .count())
print("Lavka rows per (barcode,date) distribution:")
rows_per_day.select("count").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC If `max(count) == 1` → the original loader is correct.
# MAGIC If `max(count) > 1` → switch the line below to `F.sum(lc["distribution"])` for the inner agg.

# COMMAND ----------

# Inner agg: per-day aggregation. Default sums, since multi-row-per-day means rows are sub-granular.
# If your data is already (barcode, date) unique, F.sum equals F.first and is still correct.
lavka_daily = (lavka_raw
    .withColumn("barcode", F.col(lc["barcode"]).cast("string"))
    .groupBy("barcode", F.col(lc["date"]).alias("date")).agg(
        F.sum(F.coalesce(F.col(lc["units"]), F.lit(0))).alias("daily_units"),
        F.sum(F.coalesce(F.col(lc["distribution"]), F.lit(0))).alias("daily_distribution"),
    )
    .withColumn("week", _week("date")))

lavka_w = (lavka_daily.groupBy("barcode", "week").agg(
        F.sum("daily_units").alias("lavka_units"),
        F.avg("daily_distribution").alias("lvk_numeric_distribution"),  # week-mean of daily distribution
    ))
display(lavka_w.orderBy("barcode", "week").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Lavka promo contamination mask
# MAGIC
# MAGIC Build a set of `(barcode, week)` pairs that overlap with any Lavka activation by
# MAGIC the same SKU — we cannot use these weeks for the twin baseline or twin shift.

# COMMAND ----------

lavka_promo_weeks = (lavka_acts
    .withColumn("barcode", F.col("barcode").cast("string"))
    # explode dates from start to end inclusive
    .withColumn("date",
        F.explode(F.sequence(F.col("start_date"), F.col("end_date"), F.expr("interval 1 day"))))
    .withColumn("week", _week("date"))
    .select("barcode", "week")
    .distinct()
    .withColumn("lavka_contaminated", F.lit(True)))

print(f"Contaminated (barcode, week) pairs: {lavka_promo_weeks.count()}")

# Clean twin: rows where Lavka was running its own promo are dropped
lavka_w_clean = (lavka_w
    .join(lavka_promo_weeks, ["barcode", "week"], "left")
    .filter(F.col("lavka_contaminated").isNull())
    .drop("lavka_contaminated"))

print(f"Lavka panel rows: total={lavka_w.count()}, after cleaning={lavka_w_clean.count()}")
print(f"Removed (contaminated by Lavka promo): {lavka_w.count() - lavka_w_clean.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Activations (test filter)

# COMMAND ----------

ac = ACTIVATIONS_COLS
acts = (mkt_final
    .withColumn("barcode",    F.col(ac["barcode"]).cast("string"))
    .withColumn("start_date", F.to_date(ac["start_date"]))
    .withColumn("end_date",   F.to_date(ac["end_date"]))
    .select(
        F.col(ac["placement_id"]).alias("placement_id"),
        "barcode",
        F.col(ac["tool_name"]).alias("tool_name"),
        F.col(ac["sku_category"]).alias("sku_category"),
        "start_date", "end_date",
    ))

acts_test = acts.filter(F.col("placement_id").isin(TEST_IDS_CSV))
print(f"Test activations: {acts_test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Unified weekly panel
# MAGIC
# MAGIC Note: Samokat side is **full** (never dropped), Lavka side is **cleaned** (promo weeks removed).
# MAGIC When the join produces a row with NaN Lavka — that week is a Lavka-contaminated one and the
# MAGIC per-case estimator will skip it inside pre/test windows.

# COMMAND ----------

panel = samokat_w.join(lavka_w_clean, ["barcode", "week"], "left")
display(panel.orderBy("barcode", "week").limit(20))

panel_pd = panel.toPandas()
panel_pd["week"]    = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
print("Panel rows:", len(panel_pd))
print(f"  with Lavka twin available: {panel_pd['lavka_units'].notna().sum()}")
print(f"  Lavka-contaminated (no twin this week): {panel_pd['lavka_units'].isna().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Single-case estimator: Lavka-Twin DiD v4
# MAGIC
# MAGIC Within the pre and test windows, weeks with missing Lavka twin (i.e. contaminated)
# MAGIC are dropped **before** computing means. If too few clean weeks remain, the case
# MAGIC is marked invalid with reason `lavka_contamination_too_high`.

# COMMAND ----------

def _elasticity_pre(s_pre: pd.DataFrame) -> float:
    df = s_pre[(s_pre["samokat_units"] > 0) & (s_pre["samokat_price"] > 0)
               & (s_pre["numeric_distribution"] > 0)].copy()
    if len(df) < 6:
        return FALLBACK_ELASTICITY
    df["log_q"] = np.log(df["samokat_units"] / df["numeric_distribution"])
    df["log_p"] = np.log(df["samokat_price"])
    if df["log_p"].std() < 0.02:
        return FALLBACK_ELASTICITY
    x = df["log_p"].values - df["log_p"].mean()
    y = df["log_q"].values - df["log_q"].mean()
    beta = (x * y).sum() / (x * x).sum()
    clipped = float(np.clip(beta, -3.0, 0.0))
    # If clip hit, fallback (the OLS estimate is pathological)
    if clipped in (-3.0, 0.0):
        return FALLBACK_ELASTICITY
    return clipped


def lavka_twin_did_v4(barcode: str, start: pd.Timestamp, end: pd.Timestamp,
                       panel: pd.DataFrame) -> dict:
    g = panel[panel["barcode"] == str(barcode)].set_index("week").sort_index()
    if g.empty:
        return {"valid": False, "reason": "no_data_for_sku"}

    pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
    s_pre_all  = g.loc[(g.index >= pre_start) & (g.index < start)]
    s_test_all = g.loc[(g.index >= start) & (g.index <= end)]

    # Clean weeks = where Lavka twin is available
    s_pre  = s_pre_all[s_pre_all["lavka_units"].notna()]
    s_test = s_test_all[s_test_all["lavka_units"].notna()]

    n_pre_contam  = len(s_pre_all)  - len(s_pre)
    n_test_contam = len(s_test_all) - len(s_test)

    if len(s_pre) < MIN_PRE_WEEKS:
        return {"valid": False, "reason": "insufficient_pre_clean_weeks",
                "n_pre_weeks_clean": len(s_pre), "n_pre_contaminated": n_pre_contam}
    if len(s_test) < MIN_TEST_WEEKS:
        return {"valid": False, "reason": "insufficient_test_clean_weeks",
                "n_test_weeks_clean": len(s_test), "n_test_contaminated": n_test_contam}

    # Per-distribution-point Samokat rates
    s_pre_pd  = (s_pre["samokat_units"]  / s_pre["numeric_distribution"]).mean()
    s_test_pd = (s_test["samokat_units"] / s_test["numeric_distribution"]).mean()
    # Per-distribution-point Lavka rates
    l_pre_po  = (s_pre["lavka_units"]    / s_pre["lvk_numeric_distribution"]).mean()
    l_test_po = (s_test["lavka_units"]   / s_test["lvk_numeric_distribution"]).mean()

    if not all(v and v > 0 and np.isfinite(v) for v in [s_pre_pd, l_pre_po, s_test_pd, l_test_po]):
        return {"valid": False, "reason": "tiny_or_invalid_baseline"}

    # DiD on normalized rates
    ratio_shift = l_test_po / l_pre_po
    s_exp_pd = s_pre_pd * ratio_shift
    uplift_norm = (s_test_pd - s_exp_pd) / s_exp_pd

    # Raw DiD (no normalization) for diagnostic
    s_exp_tot = s_pre["samokat_units"].mean() * (s_test["lavka_units"].mean()
                                                 / max(s_pre["lavka_units"].mean(), 1e-6))
    uplift_raw = (s_test["samokat_units"].mean() - s_exp_tot) / s_exp_tot if s_exp_tot > 0 else np.nan

    # Price decomposition
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
        "n_pre_contaminated": n_pre_contam, "n_test_contaminated": n_test_contam,
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
# MAGIC ## 11. Batch run

# COMMAND ----------

acts_pd = acts_test.toPandas()
acts_pd["start_date"] = pd.to_datetime(acts_pd["start_date"])
acts_pd["end_date"]   = pd.to_datetime(acts_pd["end_date"])

results = []
for _, r in acts_pd.iterrows():
    est = lavka_twin_did_v4(r["barcode"], r["start_date"], r["end_date"], panel_pd)
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
print("\nInvalid reasons:")
print(res[~res["valid"]]["reason"].value_counts())
display(spark.createDataFrame(res.head(30)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Tool-level rollup

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
                     contam_pre_med=("n_pre_contaminated", "median"),
                     contam_test_med=("n_test_contaminated", "median"),
                     sum_units_shelf=("uplift_units_shelf_only", "sum"))
                .sort_values("shelf_med", ascending=False))
display(spark.createDataFrame(tool_summary.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Save outputs

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
# MAGIC ## 14. Reading the outputs
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `uplift_pct_raw` | Gross % growth — finance headline |
# MAGIC | `uplift_pct_norm` | Per-distribution-point growth (listing expansion removed) |
# MAGIC | **`uplift_pct_shelf_only`** | True marketing-tool effect (price discount also removed) |
# MAGIC | `n_pre_contaminated` / `n_test_contaminated` | Weeks dropped due to a parallel Lavka promo |
# MAGIC | `price_promo_flag` | Was there a parallel Samokat discount ≥5 %? |
# MAGIC | `dist_growth` | Distribution growth during the window — raw overstates when >10 % |
# MAGIC | `elasticity` | Per-SKU price elasticity used in the decomposition (fallback −1.2) |
# MAGIC
# MAGIC **Invalid reasons cheat-sheet:**
# MAGIC - `insufficient_pre_clean_weeks` — Lavka was promoting too often in the pre-period → no clean baseline
# MAGIC - `insufficient_test_clean_weeks` — Lavka promo overlapped the entire activation window
# MAGIC - `tiny_or_invalid_baseline` — pre-period sales/distribution near zero
# MAGIC - `no_data_for_sku` — barcode not found in Samokat panel
