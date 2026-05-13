# Prompt Analysis & Improvements

## Overview

Your notebook uses a two-stage Map-Reduce approach with two main prompts:
1. **prompt_individual** (Stage 1) - Filters individual posts for relevance
2. **prompt_final_report** (Stage 2) - Generates comprehensive monthly report

## Current Issues & Suggested Improvements

---

## 1. PROMPT_INDIVIDUAL (Stage 1 - Filtering)

### Current Version:
```
FMCG/e-com РФ 2024-25:
КАТЕГОРИИ: напитки, лимонады, кола, энергетики, снеки, чипсы, молочная продукция, детское питание, холодный чай
БРЕНДЫ: Nestle(дет), Черкизово, Мираторг, Эфко, Добрый Cola, Вкусно и точка, Stars Coffee, азиатские, Балтика
ПЛАТФОРМЫ: Ozon, WB, ЯМ, X5, Магнит + онлайн/FMCG
ТЕМЫ: замещение, парал.импорт, локализация, СТМ

ИСКЛЮЧИТЬ: B2B, банки, авто, электроника

Релевантно: суть 2-3 предл
Нет: "НЕРЕЛЕВАНТНО"

{row['text']}
```

### Issues:
1. **Too cryptic** - Uses abbreviations (парал.импорт, дет, ЯМ, WB)
2. **No role/context** - Model doesn't know who it is or why it's doing this
3. **Ambiguous instructions** - "суть 2-3 предл" is unclear
4. **No output format** - How should the summary be structured?
5. **No examples** - Edge cases not covered
6. **Model mismatch** - Uses 'gemma-3-27b-it' (should be 'gemma-2-27b-it')

### Improved Version:

```python
prompt_individual = f"""Ты - аналитик российского рынка FMCG и e-commerce. Твоя задача: определить, релевантна ли новость для ежемесячного обзора рынка и извлечь ключевые факты.

ОБЛАСТЬ АНАЛИЗА (FMCG/e-commerce РФ 2024-2025):

**Категории продуктов:**
- Напитки: лимонады, кола, энергетики, холодный чай
- Снеки: чипсы, закуски
- Молочная продукция
- Детское питание

**Бренды и компании:**
- Производители: Nestle (детское питание), Черкизово, Мираторг, Эфко, Балтика
- Новые бренды: Добрый Cola, Вкусно и точка, Stars Coffee, азиатские бренды
- Маркетплейсы: Ozon, Wildberries (WB), Яндекс Маркет (ЯМ), СберМегаМаркет
- Ритейл: X5 Group (Пятёрочка, Перекрёсток), Магнит, Лента, ВкусВилл

**Ключевые темы:**
- Импортозамещение и локализация производства
- Параллельный импорт
- СТМ (собственные торговые марки)
- Омниканальность и онлайн-торговля
- Ценообразование и промо

**ИСКЛЮЧИТЬ:**
- B2B-сервисы (не для конечных потребителей)
- Финансовые услуги (банки, кредиты, инвестиции)
- Автомобили и запчасти
- Электроника и бытовая техника
- HR и вакансии

**ИНСТРУКЦИЯ:**

Если новость релевантна:
- Извлеки ключевую информацию: КТО (компания/бренд), ЧТО (событие), ЦИФРЫ (если есть)
- Напиши краткую суть в 2-3 предложениях
- Используй конкретные названия и числа

Если новость НЕ релевантна:
- Напиши только: "НЕРЕЛЕВАНТНО"

**НОВОСТЬ:**
{row['text']}

**ТВОЙ ОТВЕТ:**"""
```

### Key Improvements:
- ✅ Clear role definition
- ✅ Expanded abbreviations
- ✅ Structured sections with headers
- ✅ Explicit output format (WHO + WHAT + NUMBERS)
- ✅ Clear branching logic (relevant vs not)
- ✅ Better context for the model

---

## 2. PROMPT_FINAL_REPORT (Stage 2 - Synthesis)

### Current Version:
```
Анализ российского ритейла и FMCG за месяц.

ЗАДАЧА: структурированный отчет с фокусом на тренды, цифры, системные изменения.

КРИТЕРИИ: ✓ повторяющиеся сигналы ✓ количественные данные ✗ единичные события

РАЗДЕЛЫ:

🔸 **1. Общее состояние**
Ключевые выводы: потребление, спрос, конкуренция (2-3 тренда)
...
🔸 **8. Supply chain**
Проблемы → **[Компания/проблема]:** суть, масштаmoб, решения
...
```

### Issues:
1. **Typo** - "масштаmoб" should be "масштаб"
2. **No audience specification** - Who will read this?
3. **No tone guidance** - Formal? Analytical? Executive summary?
4. **Missing data handling** - What if section has no relevant news?
5. **Token limit risk** - prompt_final_report is 468k characters!
6. **Emoji usage** - Inconsistent (some sections have, others don't)
7. **No prioritization** - All 10 sections treated equally

### Improved Version:

```python
# First, handle the token limit issue
MAX_SUMMARIES = 500  # Limit to ~200k characters
if len(relevant_summaries) > MAX_SUMMARIES:
    print(f"⚠️ Слишком много новостей ({len(relevant_summaries)}). Используем последние {MAX_SUMMARIES}.")
    relevant_summaries = relevant_summaries[-MAX_SUMMARIES:]

combined_summaries = "\n".join(f"- {summary}" for summary in relevant_summaries)

# Check total length
if len(combined_summaries) > 300000:
    print(f"⚠️ Текст слишком длинный ({len(combined_summaries)} символов). Сокращаем...")
    combined_summaries = combined_summaries[:300000] + "\n\n[...остальные новости обрезаны из-за ограничений API...]"

prompt_final_report = f"""Ты - старший аналитик российского рынка FMCG и e-commerce. Составь ежемесячный аналитический обзор для руководства компании.

**ЦЕЛЕВАЯ АУДИТОРИЯ:** Топ-менеджмент (CEO, CMO, Head of Sales)
**ТОН:** Деловой, лаконичный, фокус на цифрах и бизнес-импликациях
**ПЕРИОД:** {start_date.strftime('%B %Y')}
**ИСТОЧНИК ДАННЫХ:** {len(relevant_summaries)} релевантных новостей из {len(channel_usernames)} отраслевых каналов

---

**ЗАДАЧА:** Структурированный отчет с фокусом на:
- ✅ Повторяющиеся сигналы (тренды, а не единичные события)
- ✅ Количественные данные (цифры, проценты, суммы)
- ✅ Системные изменения (влияние на рынок)
- ❌ Маркетинговые анонсы без бизнес-контекста
- ❌ Единичные новости без влияния на индустрию

---

**СТРУКТУРА ОТЧЕТА (10 РАЗДЕЛОВ):**

**1. Общее состояние рынка**
Основные макротренды месяца: потребительский спрос, ценообразование, конкурентная среда (2-3 ключевых вывода)

**2. Финансовые результаты**
Компании, опубликовавшие отчетность → **[Компания]:** выручка, прибыль, GMV, доля e-commerce, динамика vs прошлый период

**3. Маркетплейсы**
Ozon, Wildberries, Яндекс Маркет, СберМегаМаркет → **[Компания]:** изменения комиссий, логистика, новые сервисы, ключевые метрики

**4. Омниканальный ритейл**
X5, Магнит, Лента, ВкусВилл → **[Компания]:** цифровая трансформация, доставка, приложения, интеграция онлайн/офлайн

**5. Агрегаторы и доставка**
Яндекс Еда, Деливери, Т-банк (Купер) → **[Компания]:** тарифы, технологические новинки, география, партнерства

**6. Производители FMCG**
Бренды и фабрики → **[Компания]:** новые продукты, ценообразование, маркетинговые кампании, проблемы поставок

**7. Новинки и инновации**
Новые запуски → **[Бренд/продукт]:** описание, целевая аудитория, потенциальное влияние на рынок

**8. Цепочки поставок**
Проблемы и решения → **[Проблема/компания]:** суть, масштаб, способы решения, влияние на цены

**9. Регулирование**
Законы, ФАС, инициативы госорганов → **[Событие]:** суть изменений, кого затрагивает, сроки внедрения, бизнес-эффект

**10. Прогноз на следующий месяц**
Ожидаемые тренды и риски (2-3 предложения)

---

**ПРАВИЛА ОФОРМЛЕНИЯ:**
- Используй **жирный шрифт** для названий компаний и брендов
- Обязательно указывай цифры и даты
- Максимум 3 предложения на каждый пункт
- Если по разделу нет значимых новостей, напиши: "*Значимых изменений не зафиксировано*"
- Избегай общих фраз типа "продолжается развитие" - конкретика и факты

---

**ИСХОДНЫЕ ДАННЫЕ (НОВОСТИ):**

{combined_summaries}

---

**АНАЛИТИЧЕСКИЙ ОТЧЕТ:**"""
```

### Key Improvements:
- ✅ Clear role and audience
- ✅ Tone and style guidance
- ✅ Context about data source
- ✅ Token limit handling (critical!)
- ✅ Instructions for missing data
- ✅ Typo fixed
- ✅ More specific output requirements
- ✅ Date context added

---

## 3. MODEL SELECTION ISSUE

### Current Code:
```python
model = genai.GenerativeModel('gemma-3-27b-it')  # ❌ Incorrect model name
```

### Issue:
- `gemma-3-27b-it` doesn't exist
- Correct name is `gemma-2-27b-it` or use Gemini models

### Recommended Fix:

```python
# For Stage 1 (filtering) - use faster model
model_filter = genai.GenerativeModel('gemini-1.5-flash')  # Fast and cheap

# For Stage 2 (synthesis) - use smarter model
model_report = genai.GenerativeModel('gemini-1.5-pro')  # Better reasoning

# Or if you prefer Gemma:
# model_filter = genai.GenerativeModel('gemma-2-9b-it')  # Smaller, faster
# model_report = genai.GenerativeModel('gemma-2-27b-it')  # Larger, better
```

**Why this matters:**
- Stage 1: Runs 2800+ times → needs to be FAST
- Stage 2: Runs once → can be SLOW but SMART

---

## 4. TOKEN LIMIT PROTECTION

### Critical Issue:
```python
len(prompt_final_report)  # = 468,441 characters (~117k tokens!)
```

**Gemini limits:**
- gemini-1.5-flash: 1M tokens input
- gemini-1.5-pro: 2M tokens input
- gemini-2.0-flash: 1M tokens input

You're at ~12% of limit, but it's risky. **Add safeguards:**

```python
def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters for Russian"""
    return len(text) // 4

def truncate_summaries(summaries, max_tokens=100000):
    """Keep only most recent summaries within token limit"""
    combined = ""
    kept_summaries = []

    for summary in reversed(summaries):  # Start from most recent
        test_combined = f"- {summary}\n" + combined
        if estimate_tokens(test_combined) > max_tokens:
            break
        combined = test_combined
        kept_summaries.insert(0, summary)

    return kept_summaries

# Usage:
if estimate_tokens(combined_summaries) > 100000:
    print(f"⚠️ Truncating summaries to fit token limit...")
    relevant_summaries = truncate_summaries(relevant_summaries, max_tokens=100000)
    combined_summaries = "\n".join(f"- {s}" for s in relevant_summaries)
```

---

## 5. ERROR HANDLING & RETRY LOGIC

### Current Code:
```python
try:
    response = model.generate_content(prompt_individual)
    if "НЕРЕЛЕВАНТНО" not in response.text:
        relevant_summaries.append(response.text.strip())
    time.sleep(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    continue
```

### Issues:
- No retry on transient failures
- No handling of rate limits
- No validation of response quality

### Improved Version:

```python
import time
from google.api_core import retry, exceptions

@retry.Retry(
    predicate=retry.if_exception_type(
        exceptions.ResourceExhausted,
        exceptions.ServiceUnavailable,
    ),
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    timeout=60.0,
)
def generate_with_retry(model, prompt):
    """Generate content with automatic retry on rate limits"""
    return model.generate_content(prompt)

# Usage:
try:
    response = generate_with_retry(model, prompt_individual)

    # Validate response
    if not response.text or len(response.text.strip()) < 10:
        print(f"  ⚠️ Пустой или слишком короткий ответ, пропускаем...")
        continue

    # Check for irrelevant marker
    if "НЕРЕЛЕВАНТНО" not in response.text.upper():
        relevant_summaries.append(response.text.strip())

    time.sleep(0.5)  # Reduced from 1s - Gemini allows 60 req/min

except exceptions.ResourceExhausted:
    print(f"  ⚠️ Rate limit достигнут, пауза 10 секунд...")
    time.sleep(10)
    continue
except Exception as e:
    print(f"  ❌ Ошибка: {e}")
    continue
```

---

## 6. ADDITIONAL RECOMMENDATIONS

### A. Add Progress Tracking

```python
from tqdm import tqdm

for index, row in tqdm(df_filtered.sort_values(by='date').iterrows(),
                       total=len(df_filtered),
                       desc="Анализ постов"):
    # Your existing code
```

### B. Save Intermediate Results

```python
import json
from datetime import datetime

# After Stage 1
checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"
with open(checkpoint_file, 'w', encoding='utf-8') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'summaries': relevant_summaries,
        'total_posts': len(df_filtered),
        'relevant_count': len(relevant_summaries)
    }, f, ensure_ascii=False, indent=2)

print(f"💾 Checkpoint сохранен: {checkpoint_file}")

# To resume from checkpoint:
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
        relevant_summaries = checkpoint['summaries']
        print(f"✅ Загружено из checkpoint: {len(relevant_summaries)} summaries")
```

### C. Add Quality Metrics

```python
# After generating final report
quality_metrics = {
    'total_posts': len(df_filtered),
    'relevant_posts': len(relevant_summaries),
    'relevance_rate': len(relevant_summaries) / len(df_filtered) * 100,
    'report_length': len(final_response.text),
    'sections_count': final_response.text.count('🔸'),
    'companies_mentioned': sum(1 for line in final_response.text.split('\n') if '**' in line),
}

print("\n📊 Метрики качества:")
for key, value in quality_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.1f}")
    else:
        print(f"  {key}: {value}")
```

---

## SUMMARY OF CHANGES

### High Priority (Implement First):
1. ✅ Fix model name: `gemma-3-27b-it` → `gemini-1.5-flash` or `gemma-2-27b-it`
2. ✅ Add token limit protection (truncate_summaries function)
3. ✅ Improve prompt_individual with clear structure and role
4. ✅ Fix typo in prompt_final_report ("масштаmoб" → "масштаб")

### Medium Priority:
5. ✅ Add retry logic for API failures
6. ✅ Improve error handling
7. ✅ Add audience and tone to prompt_final_report
8. ✅ Handle missing data in sections

### Low Priority (Nice to Have):
9. ✅ Add progress bars with tqdm
10. ✅ Save intermediate checkpoints
11. ✅ Add quality metrics
12. ✅ Optimize sleep time (1s → 0.5s)

---

## IMPLEMENTATION STEPS

1. **Create backup of current notebook**
2. **Update model names** in both stages
3. **Replace prompt_individual** with improved version
4. **Replace prompt_final_report** with improved version
5. **Add token limit protection** before Stage 2
6. **Test with small dataset** (10-20 posts) first
7. **Run full pipeline** and compare results
8. **Add checkpointing** if satisfied with results

---

## EXPECTED IMPROVEMENTS

### Before:
- ❌ Cryptic prompts hard to understand
- ❌ Risk of token limit errors
- ❌ No retry on failures
- ❌ Generic summaries without clear structure

### After:
- ✅ Clear, structured prompts
- ✅ Protected from token limits
- ✅ Resilient to API errors
- ✅ Better quality output with specific facts and numbers
- ✅ Faster processing (reduced sleep time)
- ✅ Recoverable from failures (checkpoints)

---

## Questions?

If you need help implementing these changes, I can:
1. Create a new notebook with all improvements applied
2. Create a test script to compare old vs new prompts
3. Add more advanced features (like automatic quality scoring)
