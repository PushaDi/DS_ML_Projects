# Databricks notebook source
# MAGIC %md
# MAGIC # Darkstore SCM — Pre-flight Diagnostics
# MAGIC
# MAGIC Sanity checks across darkstore-level Samokat sales + product master.
# MAGIC Run top-down; fix on first FAIL; re-run.
# MAGIC
# MAGIC **Sections**
# MAGIC 1. Setup
# MAGIC 2. Product master (`ecom_etl.td_product`) — hierarchy & coverage
# MAGIC 3. Darkstore sales — granularity, uniqueness, period
# MAGIC 4. OSA, price, sales — distributions & nulls
# MAGIC 5. Activations: which SKUs land on darkstore panel
# MAGIC 6. Activation coverage: full rollout vs partial (decides SCM vs TWFE)
# MAGIC 7. Donor pool feasibility per shrinkage level
# MAGIC 8. Pre-period cleanness for synthetic control
# MAGIC 9. Cheat-sheet of fixes

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.sql.functions as F

OK   = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"

def check(label, ok, detail="", fix=""):
    print(f"{OK if ok else FAIL}  {label}")
    if detail: print(f"      {detail}")
    if (not ok) and fix: print(f"      → fix: {fix}")

def warn(label, detail, fix=""):
    print(f"{WARN}  {label}")
    print(f"      {detail}")
    if fix: print(f"      → consider: {fix}")

def _week(col):
    return F.date_trunc("week", F.col(col)).cast("date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

DARKSTORE_SALES_TABLE = "..."   # <-- ВПИШИ имя таблицы с darkstore-level продажами
PRODUCT_MASTER_TABLE  = "ecom_etl.td_product"
MEDIA_PLAN_TABLE      = "ecom_etl.marketing_activations_mp"
MEDIA_MAPPING_TABLE   = "ecom_etl.marketing_activations_sku"

TEST_IDS = pd.read_excel(
    '/Workspace/eperfectstore-prod/Dmitry_Khloptsov/notebooks/eperfectstore-prod/e-com/'
    'DATA_SCIENCE/MKT ROI CALCULATOR/AB Synthetic Test/TEST 2/'
    'Promo plan example for Samokat PO1 1.xlsx',
    sheet_name="id размещений для Димы Х.")['id размещений'].tolist()

# Darkstore sales columns (this matches your schema)
DS_COLS = dict(
    barcode      = "gtin",
    darkstore    = "warehouse_guid",
    week         = "start_of_week",
    units        = "sales_quantity",
    units_vol    = "sales_quantity_vol",
    price_base   = "price_without_promo",
    price_promo  = "promo_price",
    discount_pct = "discount_percent",
    osa          = "osa_fact",
    osa_plan     = "osa_plan",
    city         = "city_nm",
    customer     = "customer_id",
)

# Product master hierarchy (broadest → narrowest)
PRODUCT_HIERARCHY = [
    "business",
    "category_description",
    "brand_group_description",     # = sub_category (по твоему уточнению)
    "brand_description",
    "sub_brand_description",
    "gtin",
]
DONOR_MATCH_ATTRS = ["brand_size"]   # only this for donor matching

print(f"Darkstore table:        {DARKSTORE_SALES_TABLE}")
print(f"Product master:         {PRODUCT_MASTER_TABLE}")
print(f"Media plan:             {MEDIA_PLAN_TABLE}")
print(f"Mapping:                {MEDIA_MAPPING_TABLE}")
print(f"Test placements:        {len(TEST_IDS)}")
print(f"Pooling hierarchy:      {PRODUCT_HIERARCHY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Product master sanity

# COMMAND ----------

print("=" * 70); print("SECTION 2: Product master"); print("=" * 70)

pm = spark.table(PRODUCT_MASTER_TABLE)
n_pm = pm.count()
n_gtin = pm.select("gtin").distinct().count()
n_with_dup = pm.groupBy("gtin").count().filter("count > 1").count()

check("Product master non-empty", n_pm > 0, f"{n_pm:,} rows")
check("Most gtin unique in product master",
      n_with_dup < 0.01 * n_gtin,
      f"{n_with_dup} gtin with duplicates out of {n_gtin}",
      "if dupes, decide which row wins — usually take latest by sys_id")

print("\nNull-coverage of hierarchy columns:")
null_counts = pm.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in PRODUCT_HIERARCHY + DONOR_MATCH_ATTRS
]).collect()[0].asDict()
for c, n in null_counts.items():
    pct = 100 * n / n_pm
    print(f"  {c:35s}  {n:>8}  ({pct:.1f}% null)")
    if pct > 30:
        warn(f"Level {c} has heavy nulls", f"{pct:.1f}% missing",
             f"pooling at this level will be unreliable — fallback to next-up level")

# Are brand_group and sub_category the same column or not?
print("\nbrand_group_description × sub_category_description mapping (sample):")
pm.groupBy("brand_group_description", "sub_category_description").count() \
  .orderBy(F.desc("count")).limit(20).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Hierarchy cardinality on each level

# COMMAND ----------

print("Cardinality per level (how many unique values exist):")
for lvl in PRODUCT_HIERARCHY:
    n = pm.select(lvl).distinct().count()
    print(f"  {lvl:35s}  {n:>6} unique")

print("\nbrand_size buckets (donor matcher):")
pm.select("brand_size").groupBy("brand_size").count().orderBy(F.desc("count")).limit(20).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Darkstore sales — granularity & period

# COMMAND ----------

print("=" * 70); print("SECTION 3: Darkstore sales granularity"); print("=" * 70)

ds = spark.table(DARKSTORE_SALES_TABLE)
n_total = ds.count()
print(f"Total rows: {n_total:,}")

# Unique darkstores
n_warehouses = ds.select(DS_COLS["darkstore"]).distinct().count()
print(f"Unique {DS_COLS['darkstore']}: {n_warehouses:,}")
check("Approximately 2500 darkstores (matches user spec)",
      1500 <= n_warehouses <= 3500,
      f"{n_warehouses} unique darkstores",
      "if very different, customer_id may be the actual darkstore id, swap DS_COLS['darkstore']")

# Customer vs warehouse — are they 1:1 or 1:many?
print("\nRelationship between customer_id and warehouse_guid:")
cust_per_wh = ds.groupBy("warehouse_guid").agg(F.countDistinct("customer_id").alias("n_customers"))
cust_per_wh.describe("n_customers").show()
wh_per_cust = ds.groupBy("customer_id").agg(F.countDistinct("warehouse_guid").alias("n_warehouses"))
wh_per_cust.describe("n_warehouses").show()

# Granularity test
gran = ds.groupBy(DS_COLS["week"], DS_COLS["darkstore"], DS_COLS["barcode"]).count() \
         .agg(F.max("count").alias("max_dup"), F.avg("count").alias("avg_dup")).collect()[0]
check(
    f"Unique on ({DS_COLS['week']}, {DS_COLS['darkstore']}, {DS_COLS['barcode']})",
    gran["max_dup"] == 1,
    f"max dups: {gran['max_dup']}, avg: {gran['avg_dup']:.2f}",
    "if dups, add customer_id to the key OR window-deduplicate by sys_id/latest")

# Period
span = ds.agg(F.min(DS_COLS["week"]).alias("min"), F.max(DS_COLS["week"]).alias("max"),
              F.countDistinct(DS_COLS["week"]).alias("n_weeks")).collect()[0]
print(f"\nPeriod: {span['min']} → {span['max']}   ({span['n_weeks']} unique weeks)")
check("≥30 weeks of history (needed for pre-period + test)",
      span["n_weeks"] >= 30,
      f"{span['n_weeks']} weeks",
      "if less, hierarchical pooling will dominate for many cases (still works, less precise)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. OSA / price / sales — distributions

# COMMAND ----------

print("=" * 70); print("SECTION 4: Distribution of key metrics"); print("=" * 70)

print("\nosa_fact distribution (this is your darkstore-level numeric distribution):")
ds.select(F.col(DS_COLS["osa"])).describe().show()
osa_max = float(ds.agg(F.max(DS_COLS["osa"])).collect()[0][0] or 0)
if osa_max <= 1.01:
    print("  → osa_fact is a FRACTION [0,1]")
    OSA_SCALE = 1.0
elif osa_max <= 100.5:
    print("  → osa_fact is a PERCENTAGE [0,100] — will divide by 100 in pipeline")
    OSA_SCALE = 100.0
else:
    print("  → osa_fact is something larger — investigate units")
    OSA_SCALE = None

# Share of rows with zero OSA but positive sales (impossible) — data quality flag
n_paradox = ds.filter((F.col(DS_COLS["osa"]) == 0) & (F.col(DS_COLS["units"]) > 0)).count()
check("No rows with OSA=0 but units>0", n_paradox == 0,
      f"{n_paradox} paradoxical rows",
      "ETL inconsistency — usually safe to keep, but flag for the data owner")

print("\nprice_without_promo distribution:")
ds.select(F.col(DS_COLS["price_base"])).describe().show()
print("\npromo_price distribution:")
ds.select(F.col(DS_COLS["price_promo"])).describe().show()
print("\ndiscount_percent distribution:")
ds.select(F.col(DS_COLS["discount_pct"])).describe().show()

# Nulls
print("\nNull rate per critical column:")
for k in ["barcode", "darkstore", "week", "units", "price_base", "price_promo", "osa"]:
    c = DS_COLS[k]
    n = ds.filter(F.col(c).isNull()).count()
    print(f"  {c:30s}  {n:>10,}   ({100*n/n_total:.2f}% null)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Activations: which SKUs land on darkstore panel

# COMMAND ----------

print("=" * 70); print("SECTION 5: Activations vs darkstore panel"); print("=" * 70)

mp = spark.table(MEDIA_PLAN_TABLE).drop('source_filename') \
        if 'source_filename' in spark.table(MEDIA_PLAN_TABLE).columns \
        else spark.table(MEDIA_PLAN_TABLE)
mm = spark.table(MEDIA_MAPPING_TABLE)

acts = (mp.filter(F.col('client') == 'Samokat')
          .join(mm, on='placement_id', how='inner')
          .withColumnRenamed('gtin', 'barcode')
          .withColumn('start_date', F.col('start_date').cast('date'))
          .withColumn('end_date',   F.col('end_date').cast('date'))
          .withColumn('barcode', F.col('barcode').cast('string'))
          .filter(F.col('placement_id').isin(TEST_IDS))
          .dropDuplicates(['placement_id', 'barcode']))

n_tests = acts.count()
test_bcs = set(r[0] for r in acts.select('barcode').distinct().collect())
print(f"Test activation rows: {n_tests}  |  unique barcodes: {len(test_bcs)}")

ds_bcs = set(r[0] for r in
             ds.select(F.col(DS_COLS["barcode"]).cast("string")).distinct().collect())
pm_bcs = set(r[0] for r in
             spark.table(PRODUCT_MASTER_TABLE).select("gtin").distinct().collect())

print(f"Barcodes in darkstore sales:  {len(ds_bcs):,}")
print(f"Barcodes in product master:   {len(pm_bcs):,}")

ix_ds = test_bcs & ds_bcs
ix_pm = test_bcs & pm_bcs
print(f"\nTest barcodes ∩ darkstore: {len(ix_ds)} / {len(test_bcs)} ({100*len(ix_ds)/max(1,len(test_bcs)):.1f}%)")
print(f"Test barcodes ∩ product master: {len(ix_pm)} / {len(test_bcs)} ({100*len(ix_pm)/max(1,len(test_bcs)):.1f}%)")

check("≥95% of test SKUs found in darkstore sales", len(ix_ds) >= 0.95 * len(test_bcs))
check("≥95% of test SKUs found in product master",  len(ix_pm) >= 0.95 * len(test_bcs))

if len(test_bcs - ix_ds) > 0:
    print(f"\nTest SKUs missing from darkstore (sample 10):")
    for b in list(test_bcs - ix_ds)[:10]:
        print(f"  {b}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Activation coverage — full rollout vs partial
# MAGIC
# MAGIC For each placement: in how many darkstores did the SKU actually sell during the active window?
# MAGIC If coverage ≈ 100% — no untreated darkstore control. We rely on time-FE + product-pool.
# MAGIC If coverage < 100% — untreated darkstores become the gold-standard internal control.

# COMMAND ----------

print("=" * 70); print("SECTION 6: Activation rollout coverage"); print("=" * 70)

# Total darkstores per week (universe baseline)
total_warehouses = (ds.groupBy(DS_COLS["week"])
                      .agg(F.countDistinct(DS_COLS["darkstore"]).alias("n_active_warehouses_week"))
                      .agg(F.avg("n_active_warehouses_week").alias("avg")).collect()[0]["avg"])
print(f"Average warehouses active per week: {total_warehouses:.0f}")

# For each placement (barcode, start, end) — count darkstores where the SKU sold during the window
acts_pd = acts.toPandas()
acts_pd["start_date"] = pd.to_datetime(acts_pd["start_date"])
acts_pd["end_date"]   = pd.to_datetime(acts_pd["end_date"])

# Pre-compute darkstores per (barcode, week) — needed several times below
bc_week_stores = (ds.groupBy(DS_COLS["barcode"], DS_COLS["week"])
                    .agg(F.collect_set(DS_COLS["darkstore"]).alias("darkstores"),
                         F.size(F.collect_set(DS_COLS["darkstore"])).alias("n_darkstores")))
bcws = bc_week_stores.select(
    F.col(DS_COLS["barcode"]).cast("string").alias("barcode"),
    F.col(DS_COLS["week"]).alias("week"),
    "n_darkstores").toPandas()
bcws["week"] = pd.to_datetime(bcws["week"])

coverage_rows = []
for _, r in acts_pd.iterrows():
    weeks = bcws[(bcws["barcode"] == r["barcode"]) &
                 (bcws["week"] >= r["start_date"]) &
                 (bcws["week"] <= r["end_date"])]
    if len(weeks):
        avg_active = weeks["n_darkstores"].mean()
        coverage = avg_active / total_warehouses
    else:
        avg_active, coverage = 0, 0
    coverage_rows.append({
        "placement_id": r["placement_id"],
        "tool_name": r.get("tool_name"),
        "barcode": r["barcode"],
        "avg_active_darkstores": avg_active,
        "coverage_pct": coverage,
    })
cov = pd.DataFrame(coverage_rows)

print("\nDistribution of coverage across placements:")
print(cov["coverage_pct"].describe())
print(f"\nPlacements at near-full coverage (≥95%): {(cov['coverage_pct'] >= 0.95).sum()} / {len(cov)}")
print(f"Placements at partial coverage (<80%):  {(cov['coverage_pct'] <  0.80).sum()} / {len(cov)}")

if (cov["coverage_pct"] < 0.80).sum() > 0.5 * len(cov):
    print("\n✅ Good news: many placements are partial — you have an INTERNAL CONTROL group.")
    print("  → SCM can use untreated darkstores as donors. Cleanest possible identification.")
elif (cov["coverage_pct"] >= 0.95).sum() > 0.5 * len(cov):
    print("\n⚠️  Most placements roll out across all darkstores simultaneously.")
    print("  → SCM must rely on other SKUs (donor pool from product hierarchy) — still works,")
    print("    but external sanity-check via Lavka becomes more important.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Donor pool feasibility per hierarchy level
# MAGIC
# MAGIC For each treated barcode, how many candidate donor SKUs exist at each hierarchy level?

# COMMAND ----------

print("=" * 70); print("SECTION 7: Donor pool sizes by hierarchy level"); print("=" * 70)

pm_pd = pm.select(["gtin"] + PRODUCT_HIERARCHY + DONOR_MATCH_ATTRS).toPandas()
pm_pd["gtin"] = pm_pd["gtin"].astype(str)
# Deduplicate by gtin — td_product can carry SCD-style history rows
n_before = len(pm_pd)
pm_pd = pm_pd.drop_duplicates(subset=["gtin"], keep="first").reset_index(drop=True)
print(f"Product master rows: {n_before:,} -> {len(pm_pd):,} after dedup on gtin")

# All SKUs that have any sales in darkstore panel (these can be donors)
donor_universe = ds_bcs   # set built earlier
pm_donors = pm_pd[pm_pd["gtin"].isin(donor_universe)].copy()

donor_stats = []
for b in test_bcs:
    row = pm_pd[pm_pd["gtin"] == b]
    if row.empty:
        donor_stats.append({"barcode": b, **{lvl: None for lvl in PRODUCT_HIERARCHY[:-1]}})
        continue
    r = row.iloc[0]
    counts = {}
    for lvl in PRODUCT_HIERARCHY[:-1]:    # skip 'gtin' itself
        val = r[lvl]
        if pd.isna(val):
            counts[lvl] = None
            continue
        # donor universe at this level, excluding the SKU itself
        n = ((pm_donors[lvl] == val) & (pm_donors["gtin"] != b)).sum()
        counts[lvl] = n
    donor_stats.append({"barcode": b, **counts})

donor_df = pd.DataFrame(donor_stats)
print("Donors per level (median across test SKUs):")
print(donor_df[PRODUCT_HIERARCHY[:-1]].median().to_string())
print("\nDonors per level (min across test SKUs — bottleneck):")
print(donor_df[PRODUCT_HIERARCHY[:-1]].min().to_string())
print("\nDonors per level (max):")
print(donor_df[PRODUCT_HIERARCHY[:-1]].max().to_string())

# Recommend a sensible MIN_DONORS threshold
print("\nRecommendation:")
print("  - sub_brand level: pool only if donors ≥ 3")
print("  - brand level:     pool only if donors ≥ 5")
print("  - subcat / category level: pool only if donors ≥ 10")
print("  - For each test SKU, the SCM pipeline starts at the narrowest level meeting threshold.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Pre-period cleanness
# MAGIC
# MAGIC For SCM to work, donors must NOT themselves be activated during the pre-period of the treated SKU.

# COMMAND ----------

print("=" * 70); print("SECTION 8: Pre-period cleanness"); print("=" * 70)

# Build the full activation calendar (barcode × week) for ALL placements (test or not)
all_acts = (mp.filter(F.col('client') == 'Samokat')
              .join(mm, on='placement_id', how='inner')
              .withColumnRenamed('gtin', 'barcode')
              .withColumn('start_date', F.col('start_date').cast('date'))
              .withColumn('end_date',   F.col('end_date').cast('date'))
              .withColumn('barcode', F.col('barcode').cast('string')))

calendar = (all_acts
    .withColumn("date",
        F.explode(F.sequence(F.col("start_date"), F.col("end_date"), F.expr("interval 1 day"))))
    .withColumn("week", _week("date"))
    .select("barcode", "week").distinct())
cal_pd = calendar.toPandas()
cal_pd["week"] = pd.to_datetime(cal_pd["week"])
cal_set = set(zip(cal_pd["barcode"], cal_pd["week"]))
print(f"Total (barcode, week) pairs with ANY activation: {len(cal_set):,}")

# For each test placement, count how many of its potential donors (same brand) are clean in pre-period
PRE_WEEKS = 8
sample = cov.merge(donor_df[["barcode", "brand_description"]], on="barcode", how="left").head(20)
print("\nSample 20 placements — donor cleanness in pre-period:")
print("(at brand level; '%clean' = donors whose pre-window had no activation)")

for _, r in sample.iterrows():
    if pd.isna(r.get("brand_description")): continue
    treated_row = acts_pd[(acts_pd["placement_id"] == r["placement_id"]) &
                          (acts_pd["barcode"] == r["barcode"])]
    if treated_row.empty: continue
    start = treated_row.iloc[0]["start_date"]
    pre_start = start - pd.Timedelta(weeks=PRE_WEEKS)
    pre_weeks = pd.date_range(pre_start, start - pd.Timedelta(weeks=1), freq="W-MON")
    # Brand donors
    brand_donors = pm_pd[(pm_pd["brand_description"] == r["brand_description"]) &
                         (pm_pd["gtin"] != r["barcode"]) &
                         (pm_pd["gtin"].isin(donor_universe))]["gtin"].tolist()
    if not brand_donors: continue
    clean = 0
    for d in brand_donors:
        if not any((d, w) in cal_set for w in pre_weeks):
            clean += 1
    print(f"  {r['placement_id']:>18s}  bc={r['barcode']}  brand={r['brand_description'][:20]:<20}  "
          f"donors={len(brand_donors):>3}  clean={clean} ({100*clean/len(brand_donors):.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cheat-sheet
# MAGIC
# MAGIC | # | Symptom | Section | Fix |
# MAGIC |---|---|---|---|
# MAGIC | 1 | DARKSTORE_SALES_TABLE = '...' | §1 | Fill in actual table name |
# MAGIC | 2 | Granularity not unique on (week, warehouse, gtin) | §3 | Add customer_id to key, or dedup by latest sys_id |
# MAGIC | 3 | OSA scale unclear or values >100 | §4 | Set OSA_SCALE explicitly in SCM pipeline |
# MAGIC | 4 | <80% test SKUs in darkstore sales | §5 | Check barcode format / leading zeros; lpad to 13 chars |
# MAGIC | 5 | Coverage ≈ 100% for most placements | §6 | SCM will use product-hierarchy donors only; no internal control |
# MAGIC | 6 | Coverage < 80% for most placements | §6 | **Bonus** — untreated darkstores become primary control |
# MAGIC | 7 | Median donors at brand level < 5 | §7 | Drop sub_brand level from hierarchy; start at brand |
# MAGIC | 8 | Most brand donors contaminated in pre-period | §8 | Widen pre-window OR use category-level pooling |
# MAGIC
# MAGIC When everything is ✅ — message me and I'll write `synthetic_ab_scm.py` based on the actual numbers from this diagnostic.

# COMMAND ----------

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
print("Fix issues top-down, then proceed to the SCM pipeline.")
