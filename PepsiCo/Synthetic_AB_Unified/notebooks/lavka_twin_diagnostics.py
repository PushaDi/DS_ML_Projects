# Databricks notebook source
# MAGIC %md
# MAGIC # Lavka-Twin DiD — Pre-flight Diagnostics
# MAGIC
# MAGIC Run this **before** the actual estimator (`lavka_twin_did_databricks_v4`). Walks the
# MAGIC pipeline from raw tables to the final panel and prints **PASS / WARN / FAIL** at every
# MAGIC stage. Each FAIL has a "→ how to fix" hint next to it.
# MAGIC
# MAGIC **Sections**
# MAGIC 1. Setup & raw tables
# MAGIC 2. Samokat sales — granularity, nulls, distribution, price
# MAGIC 3. Lavka sales — granularity, distribution semantics
# MAGIC 4. Marketing activations — Lavka activations existence (CRITICAL for v4)
# MAGIC 5. Barcode type / format compatibility across tables
# MAGIC 6. Lavka-promo mask construction & coverage
# MAGIC 7. Activation ↔ panel match (how many SKUs land on the panel)
# MAGIC 8. Per-tool invalidity audit (why cases fail — `Полка "Лучшее"` and Pepsi cases)
# MAGIC 9. Outlier scan & sum-vs-median disagreement (Приоритезация −682k case)
# MAGIC 10. Cheat-sheet of fixes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Helpers — minimal, no DiD math here

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import Window
from datetime import timedelta

OK   = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"

def check(label: str, ok: bool, detail: str = "", fix: str = ""):
    mark = OK if ok else FAIL
    print(f"{mark}  {label}")
    if detail: print(f"      {detail}")
    if (not ok) and fix:
        print(f"      → fix: {fix}")

def warn(label: str, detail: str, fix: str = ""):
    print(f"{WARN}  {label}")
    print(f"      {detail}")
    if fix: print(f"      → consider: {fix}")

def _week(col):
    return F.date_trunc("week", F.col(col)).cast("date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config (must match v4 exactly)

# COMMAND ----------

SAMOKAT_SALES_TABLE = "ecom_etl.ds_samokat_availability_metrics_city"
LAVKA_SALES_TABLE   = "ecom_etl.ds_sim_inference_data"
MEDIA_PLAN_TABLE    = "ecom_etl.marketing_activations_mp"
MEDIA_MAPPING_TABLE = "ecom_etl.marketing_activations_sku"

TEST_IDS = pd.read_excel(
    '/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/'
    'DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/'
    'Promo plan example for Samokat PO1 1.xlsx',
    sheet_name="id размещений для Димы Х.")['id размещений'].tolist()

SAMOKAT_COLS = dict(barcode="gtin", date="start_of_week", units="sales_quantity",
                    price="mode_promo_price", city="city_nm", distribution="warehouse_count")
LAVKA_COLS   = dict(barcode="barcode", date="date", units="sales_quantity",
                    distribution="distribution")

print(f"Samokat table:    {SAMOKAT_SALES_TABLE}")
print(f"Lavka table:      {LAVKA_SALES_TABLE}")
print(f"Media plan table: {MEDIA_PLAN_TABLE}")
print(f"Mapping table:    {MEDIA_MAPPING_TABLE}")
print(f"Test placements:  {len(TEST_IDS)}")

# COMMAND ----------

samokat_raw = spark.table(SAMOKAT_SALES_TABLE)
lavka_raw   = spark.table(LAVKA_SALES_TABLE)
mp_raw      = spark.table(MEDIA_PLAN_TABLE)
mm_raw      = spark.table(MEDIA_MAPPING_TABLE)

print("Samokat columns:", samokat_raw.columns)
print("Lavka columns:",   lavka_raw.columns)
print("Media plan columns:", mp_raw.columns)
print("Media mapping columns:", mm_raw.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SAMOKAT sales — sanity

# COMMAND ----------

print("=" * 70)
print("SECTION 2: Samokat sales table")
print("=" * 70)

sc = SAMOKAT_COLS
n_total = samokat_raw.count()
check("Samokat table non-empty", n_total > 0, f"{n_total:,} rows")

# Columns exist
missing = [v for v in sc.values() if v not in samokat_raw.columns]
check("All required columns present", not missing,
      f"missing: {missing}",
      f"adjust SAMOKAT_COLS to match real schema: {samokat_raw.columns}")

# Granularity: rows per (barcode, week, city)
gran = (samokat_raw
        .groupBy(sc["barcode"], sc["date"], sc["city"]).count()
        .agg(F.max("count").alias("max_dup"),
             F.avg("count").alias("avg_dup")).collect()[0])
check(
    "Samokat is unique on (barcode, start_of_week, city)",
    gran["max_dup"] == 1,
    f"max duplicates per key: {gran['max_dup']}, avg: {gran['avg_dup']:.2f}",
    "duplicate keys mean we are double-counting units. Add dedup or use windowed first().")

# Nulls in critical columns
nulls = samokat_raw.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in [sc["barcode"], sc["date"], sc["units"], sc["price"],
              sc["city"], sc["distribution"]]
]).collect()[0].asDict()
print(f"  Null counts: {nulls}")
check("No null barcodes",  nulls[sc["barcode"]] == 0)
check("No null weeks",      nulls[sc["date"]] == 0)
check("No null cities",     nulls[sc["city"]] == 0)
check("Distribution column has data",
      nulls[sc["distribution"]] < n_total * 0.1,
      f"{nulls[sc['distribution']]:,} nulls in {sc['distribution']} ({100*nulls[sc['distribution']]/n_total:.1f}%)",
      "if >50% null, you're losing distribution signal; check ETL of warehouse_count")
if nulls[sc["price"]] > 0:
    pct = 100*nulls[sc['price']]/n_total
    warn("Price column has nulls",
         f"{nulls[sc['price']]:,} ({pct:.1f}%) — likely weeks without promo. mode_promo_price is NULL outside promo windows.",
         "use coalesce(mode_promo_price, regular_price) — if regular_price exists in schema")

# Distribution distribution :)
print("\nWarehouse_count distribution (across all rows):")
samokat_raw.select(F.col(sc["distribution"])).describe().show()

# Date span
span = (samokat_raw.agg(
    F.min(sc["date"]).alias("min"),
    F.max(sc["date"]).alias("max"),
    F.countDistinct(sc["date"]).alias("n_weeks")).collect()[0])
print(f"  Date span: {span['min']} → {span['max']} ({span['n_weeks']} unique weeks)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Samokat: granularity of `start_of_week` — is it really weekly?

# COMMAND ----------

# If date_trunc("week", start_of_week) != start_of_week, the column isn't aligned to weeks
mismatch = samokat_raw.filter(
    F.date_trunc("week", F.col(sc["date"])).cast("date") != F.col(sc["date"]).cast("date")
).count()
check(
    "start_of_week values aligned to ISO week start",
    mismatch == 0,
    f"{mismatch} rows have start_of_week not equal to date_trunc('week', start_of_week)",
    "if non-zero, your aggregation re-truncates differently → check timezone/locale; "
    "the canonical Monday-start week is what date_trunc('week', ...) produces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. LAVKA sales — sanity & distribution semantics

# COMMAND ----------

print("=" * 70)
print("SECTION 3: Lavka sales table")
print("=" * 70)

lc = LAVKA_COLS
n_total_l = lavka_raw.count()
check("Lavka table non-empty", n_total_l > 0, f"{n_total_l:,} rows")

missing = [v for v in lc.values() if v not in lavka_raw.columns]
check("All required columns present", not missing,
      f"missing: {missing}",
      f"adjust LAVKA_COLS to match real schema: {lavka_raw.columns}")

# How many rows per (barcode, date)? This decides if double-avg in v4 is correct.
gran_l = (lavka_raw.groupBy(lc["barcode"], lc["date"]).count())
gran_stats = gran_l.agg(
    F.max("count").alias("max_dup"),
    F.avg("count").alias("avg_dup"),
    F.expr("percentile_approx(count, 0.5)").alias("p50"),
    F.expr("percentile_approx(count, 0.95)").alias("p95"),
).collect()[0]
print(f"  Rows per (barcode, date): max={gran_stats['max_dup']}, "
      f"avg={gran_stats['avg_dup']:.2f}, p50={gran_stats['p50']}, p95={gran_stats['p95']}")

if gran_stats["max_dup"] == 1:
    check("Lavka is unique on (barcode, date) — double-avg in v4 is OK", True)
else:
    warn("Lavka has multiple rows per (barcode, date) — likely split by city/region",
         f"max_dup={gran_stats['max_dup']}, p50={gran_stats['p50']}",
         "in v4 ячейка 6: use F.sum(distribution) inside the inner groupBy (per day), "
         "then F.avg(daily_distribution) outer. Currently v4 already uses sum() inner — OK.")

# Distribution semantics check: is it [0,1], [0,100], or counts of outlets?
dist_stats = lavka_raw.select(F.col(lc["distribution"])).describe().collect()
print("\nLavka 'distribution' column distribution:")
for r in dist_stats: print(" ", r.asDict())
dist_max = float(lavka_raw.agg(F.max(lc["distribution"])).collect()[0][0] or 0)
if dist_max <= 1.01:
    print("  → Looks like a fraction [0,1] — numeric distribution / 100")
elif dist_max <= 100.5:
    print("  → Looks like percentage [0,100]")
else:
    print("  → Looks like a COUNT (raw outlet count). Sum-by-day inside the week is correct.")

# Nulls
nulls_l = lavka_raw.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in [lc["barcode"], lc["date"], lc["units"], lc["distribution"]]
]).collect()[0].asDict()
print(f"\n  Null counts: {nulls_l}")
check("No null barcodes (Lavka)", nulls_l[lc["barcode"]] == 0)
check("Distribution has data",
      nulls_l[lc["distribution"]] < n_total_l * 0.1,
      f"{nulls_l[lc['distribution']]:,} nulls in distribution ({100*nulls_l[lc['distribution']]/n_total_l:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MARKETING ACTIVATIONS — `Yandex.Lavka` rows must exist
# MAGIC
# MAGIC This is **the** check for the broken Lavka-promo mask in v4.

# COMMAND ----------

print("=" * 70)
print("SECTION 4: Marketing activations (the Lavka-mask blocker)")
print("=" * 70)

mp = mp_raw.drop('source_filename') if 'source_filename' in mp_raw.columns else mp_raw
n_mp = mp.count()
n_mp_samokat = mp.filter(F.col('client') == 'Samokat').count()
n_mp_lavka   = mp.filter(F.col('client') == 'Yandex.Lavka').count()

check("media_plan has any rows", n_mp > 0, f"{n_mp:,} total rows")
check("media_plan has Samokat activations", n_mp_samokat > 0, f"{n_mp_samokat:,} rows")
check(
    "media_plan has Lavka activations",
    n_mp_lavka > 0,
    f"{n_mp_lavka:,} Lavka rows — THIS IS WHAT v4 USES TO CLEAN THE TWIN",
    "if 0, you have no Lavka activation feed → ask the marketing team for one or "
    "use a different proxy (e.g., promo_price drops in Lavka sales)")

# What client values actually exist?
print("\nClient values in media_plan:")
mp.groupBy('client').count().show()

if n_mp_lavka > 0:
    # Now the inner join — does it survive?
    mp_l = mp.filter(F.col('client') == 'Yandex.Lavka')
    n_lavka_in_mapping = (mp_l.join(mm_raw, on='placement_id', how='inner')).count()
    n_lavka_lost = n_mp_lavka - mp_l.join(mm_raw, on='placement_id', how='inner').select('placement_id').distinct().count()

    check(
        "Lavka activations survive inner join with mapping",
        n_lavka_in_mapping > 0,
        f"{n_lavka_in_mapping:,} Lavka (placement × sku) rows after join (vs {n_mp_lavka:,} before)",
        "if 0, mapping table has no Lavka SKUs — switch to LEFT join (see v4.1 patch)")

    if n_lavka_in_mapping == 0:
        # Show which Lavka placement_ids are missing from mapping
        missing_pids = mp_l.join(mm_raw, on='placement_id', how='left_anti').select('placement_id').distinct()
        print(f"\nLavka placement_ids missing from mapping ({missing_pids.count()} unique):")
        missing_pids.limit(10).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. BARCODE format compatibility across tables
# MAGIC
# MAGIC The Pepsi 0.5л / 1л case in the prior run had `no_lavka_twin` because barcodes
# MAGIC didn't match across Samokat and Lavka. Here we check it for **all** test SKUs.

# COMMAND ----------

print("=" * 70)
print("SECTION 5: Barcode compatibility")
print("=" * 70)

# Gather barcodes from each side as strings
sa_bcs = set(r[0] for r in samokat_raw.select(F.col(sc["barcode"]).cast("string")).distinct().collect())
lv_bcs = set(r[0] for r in lavka_raw.select(F.col(lc["barcode"]).cast("string")).distinct().collect())

print(f"Samokat distinct barcodes: {len(sa_bcs)}")
print(f"Lavka distinct barcodes:   {len(lv_bcs)}")
print(f"Intersection:              {len(sa_bcs & lv_bcs)}")
print(f"Samokat-only (no twin):    {len(sa_bcs - lv_bcs)}")
print(f"Lavka-only (irrelevant):   {len(lv_bcs - sa_bcs)}")

check(
    "≥80% of Samokat barcodes have a Lavka twin",
    len(sa_bcs & lv_bcs) >= 0.8 * len(sa_bcs),
    f"intersection rate = {100 * len(sa_bcs & lv_bcs) / max(len(sa_bcs), 1):.1f}%",
    "if <80%, check barcode formats: leading zeros, EAN-13 vs UPC-12, integer vs string")

# Check format patterns
def _format_summary(bcs, name):
    lens = pd.Series([len(b) for b in bcs])
    print(f"  {name}: lengths min={lens.min()}, max={lens.max()}, "
          f"p50={int(lens.median())}, most common length={lens.mode().tolist()[:3]}")
print()
_format_summary(sa_bcs, "Samokat barcodes")
_format_summary(lv_bcs, "Lavka barcodes")

# Probe specific Pepsi SKUs that failed last time
pepsi_probes = ['4600494000188', '4600494000416', '04600494000188', '04600494000416',
                '600494000188', '600494000416']
print("\nPepsi probe — which forms exist in each table:")
for b in pepsi_probes:
    in_sa = b in sa_bcs
    in_lv = b in lv_bcs
    print(f"  {b:>16s}  Samokat: {'YES' if in_sa else '—'}    Lavka: {'YES' if in_lv else '—'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Try padding to canonical 13-char and re-check

# COMMAND ----------

def _pad(b):
    return b.zfill(13) if b and len(b) <= 13 else b

sa_pad = {_pad(b) for b in sa_bcs}
lv_pad = {_pad(b) for b in lv_bcs}
ix_pad = len(sa_pad & lv_pad)
ix_raw = len(sa_bcs & lv_bcs)
print(f"Intersection RAW:      {ix_raw}")
print(f"Intersection PADDED13: {ix_pad}")
if ix_pad > ix_raw:
    warn("Padding to 13 chars increases the intersection",
         f"+{ix_pad - ix_raw} extra SKUs recovered with lpad(barcode, 13, '0')",
         "add F.lpad(barcode, 13, '0') BEFORE every barcode-based join in v4")
else:
    check("Padding doesn't change anything — barcode formats already aligned", True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LAVKA-PROMO MASK — build and inspect coverage

# COMMAND ----------

print("=" * 70)
print("SECTION 6: Lavka-promo mask construction")
print("=" * 70)

# Build the mask under v4's logic
mp_full = mp_raw.drop('source_filename') if 'source_filename' in mp_raw.columns else mp_raw
mp_mapped = (mp_full
    .join(mm_raw, on='placement_id', how='inner')
    .withColumn('start_date', F.col('start_date').cast('date'))
    .withColumn('end_date',   F.col('end_date').cast('date')))

lavka_acts_inner = (mp_mapped
    .filter(F.col('client') == 'Yandex.Lavka')
    .withColumnRenamed('gtin', 'barcode')
    .select('barcode', 'start_date', 'end_date')
    .dropna())

# And under v4.1's (LEFT join) logic for comparison
lavka_acts_left = (mp_full
    .filter(F.col('client') == 'Yandex.Lavka')
    .join(mm_raw, on='placement_id', how='left')
    .withColumnRenamed('gtin', 'barcode')
    .withColumn('start_date', F.col('start_date').cast('date'))
    .withColumn('end_date',   F.col('end_date').cast('date'))
    .select('barcode', 'start_date', 'end_date')
    .filter(F.col('barcode').isNotNull() & F.col('start_date').isNotNull()
            & F.col('end_date').isNotNull()))

n_inner = lavka_acts_inner.count()
n_left  = lavka_acts_left.count()
print(f"Lavka acts (inner join, v4 current):  {n_inner}")
print(f"Lavka acts (left join,  v4.1 fix):    {n_left}")

check(
    "Lavka activation mask is non-empty under v4's inner-join",
    n_inner > 0,
    f"got {n_inner} Lavka activations — if 0, that's why contam_pre/contam_test = 0 everywhere",
    f"switch inner→left: gets {n_left} rows instead. See cheat-sheet at the bottom.")

# Type check on barcode
if n_inner > 0 or n_left > 0:
    lavka_acts = lavka_acts_left if n_left > n_inner else lavka_acts_inner
    bcs_in_mask = set(r[0] for r in lavka_acts.select(F.col('barcode').cast('string')).distinct().collect())
    print(f"\nUnique barcodes in Lavka-mask: {len(bcs_in_mask)}")
    print(f"  ∩ Samokat barcodes: {len(bcs_in_mask & sa_bcs)}")
    print(f"  ∩ Lavka sales barcodes: {len(bcs_in_mask & lv_bcs)}")
    check("Mask barcodes intersect Samokat barcodes (otherwise mask filters nothing)",
          len(bcs_in_mask & sa_bcs) > 0,
          f"intersection={len(bcs_in_mask & sa_bcs)}",
          "if 0, barcode types/format differ between media_mapping and sales tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 If mask is non-empty: explode dates and count contaminated (barcode, week) pairs

# COMMAND ----------

if n_left > 0:
    contam = (lavka_acts_left
        .withColumn("barcode", F.col("barcode").cast("string"))
        .withColumn("date",
            F.explode(F.sequence(F.col("start_date"), F.col("end_date"), F.expr("interval 1 day"))))
        .withColumn("week", _week("date"))
        .select("barcode", "week")
        .distinct())
    n_contam_pairs = contam.count()
    print(f"Contaminated (barcode, week) pairs: {n_contam_pairs}")
    print("Sample:")
    contam.limit(10).show()
else:
    print("Skipped — mask is empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ACTIVATION ↔ PANEL match — how many test SKUs reach the panel?

# COMMAND ----------

print("=" * 70)
print("SECTION 7: Activation-to-panel match")
print("=" * 70)

# Test activations
mkt_test = (mp_mapped
    .filter(F.col('client') == 'Samokat')
    .withColumnRenamed('gtin', 'barcode')
    .withColumn('barcode', F.col('barcode').cast('string'))
    .filter(F.col('placement_id').isin(TEST_IDS))
    .dropDuplicates(['placement_id', 'barcode']))

n_tests = mkt_test.count()
print(f"Test activations (placement × barcode rows): {n_tests}")

# How many unique barcodes
n_test_bcs = mkt_test.select('barcode').distinct().count()
print(f"Unique test barcodes: {n_test_bcs}")

# How many of them have ANY data in Samokat sales table?
test_bc_set = set(r[0] for r in mkt_test.select('barcode').distinct().collect())
in_samokat = test_bc_set & sa_bcs
in_lavka   = test_bc_set & lv_bcs
print(f"  in Samokat sales:  {len(in_samokat)} / {len(test_bc_set)}")
print(f"  in Lavka sales:    {len(in_lavka)} / {len(test_bc_set)}")

check(
    "≥95% of test SKUs found in Samokat sales",
    len(in_samokat) >= 0.95 * len(test_bc_set),
    f"hit rate = {100*len(in_samokat)/len(test_bc_set):.1f}%",
    "if low, SKUs are activated but never sold in Samokat (catalogue issue) — drop them")
check(
    "≥80% of test SKUs found in Lavka sales (twins exist)",
    len(in_lavka) >= 0.80 * len(test_bc_set),
    f"hit rate = {100*len(in_lavka)/len(test_bc_set):.1f}%",
    "if low, twin-method is structurally infeasible for many SKUs → consider fallback method")

if len(test_bc_set - in_lavka) > 0:
    print(f"\nTest SKUs WITHOUT Lavka twin (sample 20 of {len(test_bc_set - in_lavka)}):")
    for b in list(test_bc_set - in_lavka)[:20]:
        print(f"  {b}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. INVALIDITY AUDIT — why cases fail

# COMMAND ----------

print("=" * 70)
print("SECTION 8: Why did cases fail in v4 (Polka Luchshee + Pepsi)")
print("=" * 70)

# Recompute aggregations exactly like v4 does (no DiD, just the panel)
samokat_w = (samokat_raw
    .withColumn("barcode", F.col(sc["barcode"]).cast("string"))
    .withColumn("week", _week(sc["date"]))
    .groupBy("barcode", "week").agg(
        F.sum(F.coalesce(F.col(sc["units"]), F.lit(0))).alias("samokat_units"),
        F.sum(F.coalesce(F.col(sc["distribution"]), F.lit(0))).alias("numeric_distribution"),
    ))

lavka_w = (lavka_raw
    .withColumn("barcode", F.col(lc["barcode"]).cast("string"))
    .groupBy("barcode", F.col(lc["date"]).alias("date")).agg(
        F.sum(F.coalesce(F.col(lc["units"]), F.lit(0))).alias("daily_units"),
        F.sum(F.coalesce(F.col(lc["distribution"]), F.lit(0))).alias("daily_dist"),
    )
    .withColumn("week", _week("date"))
    .groupBy("barcode", "week").agg(
        F.sum("daily_units").alias("lavka_units"),
        F.avg("daily_dist").alias("lvk_dist")))

panel = (samokat_w.join(lavka_w, ["barcode", "week"], "left")
                  .select("barcode", "week", "samokat_units", "numeric_distribution",
                          "lavka_units", "lvk_dist"))
panel_pd = panel.toPandas()
panel_pd["week"]    = pd.to_datetime(panel_pd["week"])
panel_pd["barcode"] = panel_pd["barcode"].astype(str)
print(f"Panel rows: {len(panel_pd)}")
print(f"  with Lavka twin: {panel_pd['lavka_units'].notna().sum()}")
print(f"  without Lavka twin: {panel_pd['lavka_units'].isna().sum()}")

# For each test activation, count clean pre and test weeks
acts_pd = mkt_test.toPandas()
acts_pd["start_date"] = pd.to_datetime(acts_pd["start_date"])
acts_pd["end_date"]   = pd.to_datetime(acts_pd["end_date"])

PRE_WEEKS = 8
audit = []
for _, r in acts_pd.iterrows():
    g = panel_pd[panel_pd["barcode"] == str(r["barcode"])].set_index("week").sort_index()
    pre_start = r["start_date"] - pd.Timedelta(weeks=PRE_WEEKS)
    pre_all  = g.loc[(g.index >= pre_start) & (g.index < r["start_date"])]
    test_all = g.loc[(g.index >= r["start_date"]) & (g.index <= r["end_date"])]
    audit.append({
        "placement_id": r["placement_id"],
        "barcode":       r["barcode"],
        "tool_name":     r.get("tool_name"),
        "n_pre_total":   len(pre_all),
        "n_pre_with_lavka":  pre_all["lavka_units"].notna().sum() if len(pre_all) else 0,
        "n_test_total":  len(test_all),
        "n_test_with_lavka": test_all["lavka_units"].notna().sum() if len(test_all) else 0,
        "sku_in_panel":  not g.empty,
    })
audit = pd.DataFrame(audit)

print("\n=== Per-tool validity audit ===")
audit_summary = audit.groupby("tool_name").agg(
    n=("placement_id", "size"),
    sku_in_panel_pct=("sku_in_panel", "mean"),
    pre_total_med=("n_pre_total", "median"),
    pre_lavka_med=("n_pre_with_lavka", "median"),
    test_total_med=("n_test_total", "median"),
    test_lavka_med=("n_test_with_lavka", "median"),
    valid_under_v4=("n_test_with_lavka",
        lambda s: ((s >= 2) & (audit.loc[s.index, "n_pre_with_lavka"] >= 4)).mean()),
).sort_values("n", ascending=False)
print(audit_summary.to_string())

# Focus on "Полка Лучшее"
polka = audit[audit["tool_name"] == 'Полка "Лучшее"']
print(f"\n=== Полка \"Лучшее\" deep-dive ({len(polka)} cases) ===")
if len(polka) > 0:
    print(polka[["placement_id", "barcode", "n_pre_total",
                 "n_pre_with_lavka", "n_test_total", "n_test_with_lavka"]].to_string())
    n_too_short = (polka["n_test_with_lavka"] < 2).sum()
    if n_too_short > 0:
        warn(f"Полка Лучшее: {n_too_short} cases have <2 clean test weeks",
             "this is why the tool disappeared from the v4 rollup",
             "lower MIN_TEST_WEEKS to 1 in v4, OR aggregate to daily for this tool only")
else:
    print("No 'Полка Лучшее' cases in test set — check test_ids list")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. OUTLIER scan — who drives Приоритезация −682k штук

# COMMAND ----------

print("=" * 70)
print("SECTION 9: Outlier scan for sum-vs-median sign disagreement")
print("=" * 70)

# We don't have the per-case uplift here (DiD lives in v4); but we can flag
# placements where (a) SKU is in Lavka twin set, (b) test distribution dropped sharply
# (b) is the usual cause of huge negative units estimates.

dist_change_audit = []
for _, r in acts_pd.iterrows():
    g = panel_pd[panel_pd["barcode"] == str(r["barcode"])].set_index("week").sort_index()
    pre_start = r["start_date"] - pd.Timedelta(weeks=PRE_WEEKS)
    pre  = g.loc[(g.index >= pre_start) & (g.index < r["start_date"])]
    test = g.loc[(g.index >= r["start_date"]) & (g.index <= r["end_date"])]
    if len(pre) == 0 or len(test) == 0: continue
    dp = pre["numeric_distribution"].mean()
    dt = test["numeric_distribution"].mean()
    pre_units = pre["samokat_units"].mean()
    test_units = test["samokat_units"].mean()
    dist_change_audit.append({
        "placement_id": r["placement_id"], "barcode": r["barcode"],
        "tool_name": r.get("tool_name"),
        "dist_pre": dp, "dist_test": dt,
        "dist_growth": (dt / dp - 1) if dp > 0 else np.nan,
        "units_pre": pre_units, "units_test": test_units,
    })
dca = pd.DataFrame(dist_change_audit)

# Distribution dropped >10% during activation
big_drop = dca[dca["dist_growth"] < -0.10].sort_values("dist_growth")
print(f"Activations where distribution dropped >10% during the window: {len(big_drop)}")
if len(big_drop) > 0:
    warn(
        "These cases likely produce huge negative uplift_units that pollute the tool-level sum",
        f"top 10 by dist_growth drop:",
        "exclude cases with dist_growth < -0.10 from the tool-level rollup, OR "
        "report shelf_med separately for these (they're 'delisting events', not marketing effects)")
    print(big_drop[["placement_id", "barcode", "tool_name",
                    "dist_pre", "dist_test", "dist_growth",
                    "units_pre", "units_test"]].head(10).to_string())

# Distribution increased >25% — listing expansion, raw is inflated
big_growth = dca[dca["dist_growth"] > 0.25].sort_values("dist_growth", ascending=False)
print(f"\nActivations where distribution grew >25% during the window: {len(big_growth)}")
if len(big_growth) > 0:
    print(big_growth[["placement_id", "barcode", "tool_name",
                      "dist_pre", "dist_test", "dist_growth"]].head(10).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. CHEAT-SHEET — fixes in priority order
# MAGIC
# MAGIC | # | Symptom | Section | Fix in v4 |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `contam_pre_med = contam_test_med = 0` (mask empty) | §4, §6 | If §4 says no Lavka activations at all → ask marketing for the feed. If §4 says yes but §6 says inner-join wipes them → **change `inner` to `left`** in `lavka_acts = ...join(media_mapping, ..., 'left')` |
# MAGIC | 2 | Pepsi SKUs `no_lavka_twin` | §5 | If §5.1 says padding helps → add `F.lpad(F.col('barcode'), 13, '0')` everywhere before joins on `barcode` |
# MAGIC | 3 | `Полка "Лучшее"` disappeared | §8 | If `n_test_with_lavka < 2` everywhere → lower `MIN_TEST_WEEKS` to 1 OR add a `short_activation` flag and process those daily |
# MAGIC | 4 | Приоритезация sum_units negative while shelf_med positive | §9 | Exclude cases with `dist_growth < -0.10` from rollup; they are SKU delistings, not marketing failures |
# MAGIC | 5 | High `n_price_promo` for a tool | (v4 output) | Report the tool's `shelf_med` AND a separate `shelf_med_no_promo` filtered to `price_promo_flag == False`. If they diverge, the tool's "effect" is partly the discount.|
# MAGIC | 6 | Wide `[shelf_p25, shelf_p75]` bands | (v4 output) | Add winsorization at [5%, 95%] before median in `tool_summary`. Reduces tails without dropping cases. |

# COMMAND ----------

print("\n\n=" * 35)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
print("Run each section, fix from top of cheat-sheet down, then re-run v4.")
