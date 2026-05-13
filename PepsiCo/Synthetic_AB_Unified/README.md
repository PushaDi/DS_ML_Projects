# Synthetic_AB_Unified

Консолидированный проект по оценке uplift маркетинговых инструментов в Самокате при отсутствии классического A/B.

## Что внутри

Сводит воедино **4 метода** из `PepsiCo/Synthetic_AB/Test 2.1/` и `PepsiCo/Sytnhetic_AB/New_test/`:

- **SCM** (Synthetic Control, route A/B/C + placebo)
- **GSC + IFE** (Generalized Synthetic Control)
- **MC** (Matrix Completion)
- **HBM** (Hierarchical Bayesian on log-target)

Делает meta-analysis (попарные корреляции, sign agreement), выбирает best-fit пару методов и выдаёт финальный рейтинг инструментов.

## Финальный ответ

Лучший подход — **консенсус SCM + HBM** (sign agreement 73 %, единственная пара с положительной корреляцией). GSC и MC при T=57 недель смещены и используются только как sanity-check.

См. [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md) — полный отчёт с рейтингом 8 инструментов и SOTA-ресерчем.

## Запуск

```bash
/Users/dmitry/DS_ML_Projects/PepsiCo/Sales_impact_model/.venv-hbm/bin/python src/meta_analysis.py
```

Output: `results/unified_per_case.csv`, `results/unified_tool_summary.csv`, `results/method_agreement.csv`.

## Структура

```
data/        # все CSV из существующих прогонов SCM/GSC/MC/HBM
src/         # meta_analysis.py — консолидация
results/     # выход финального прогона
reports/     # FINAL_REPORT.md
research/    # ссылки на SOTA-методы (см. отчёт § 3)
```
