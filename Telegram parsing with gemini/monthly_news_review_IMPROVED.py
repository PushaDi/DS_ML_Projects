# %% [markdown]
# # Monthly News Review - IMPROVED VERSION
#
# **Improvements in this version:**
# - ✅ Fixed model names (gemini-1.5-flash for filtering, gemini-1.5-pro for synthesis)
# - ✅ Better structured prompts with clear roles
# - ✅ Token limit protection
# - ✅ Retry logic for API failures
# - ✅ Progress tracking with tqdm
# - ✅ Checkpoint system to recover from failures
# - ✅ Quality metrics
# - ✅ Automatic Apple Notes integration

# %% [markdown]
# ## 📦 Install Dependencies

# %%
# %pip install telethon google-generativeai python-dotenv nest_asyncio tqdm

# %% [markdown]
# ## 🔧 Configuration

# %%
import nest_asyncio
nest_asyncio.apply()

import os
import pandas as pd
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from telethon.sync import TelegramClient
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import retry, exceptions

load_dotenv()

# %%
# API Configuration
api_id = 26086057
api_hash = 'b38a2ed34078cf550fe4a24a4e70396c'
gemini_api_key = 'AIzaSyAcw-ZQLNdY7j7YdoDLqp8iNcou6lLkZt4'
session_name = 'parse_session'

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Use appropriate models for each stage
model_filter = genai.GenerativeModel('gemini-1.5-flash')  # Fast for filtering
model_report = genai.GenerativeModel('gemini-1.5-pro')    # Smart for synthesis

print("✅ Configuration loaded")
print(f"   Filter model: gemini-1.5-flash (fast)")
print(f"   Report model: gemini-1.5-pro (smart)")

# %% [markdown]
# ## 📡 Telegram Channel Configuration

# %%
channel_usernames = [
    'https://t.me/PRORETAILCOMMUNITY',
    'https://t.me/ozon_adv',
    'https://t.me/mediaresearchesanalytics',
    'https://t.me/RetailTECHNet',
    'https://t.me/torgFMCG',
    'https://t.me/svobodakassa',
    'https://t.me/fmcg_ru',
    'https://t.me/retail_fmcg',
    'https://t.me/mpgo_ru',
    'https://t.me/DataInsight',
    'https://t.me/retailtoday',
    'https://t.me/foodtechonline',
    'https://t.me/YakovPartners',
    'https://t.me/ecom_russia',
    'https://t.me/magnitnews',
    'https://t.me/foodtech_russia',
    'https://t.me/bs_marketplace',
    'https://t.me/praktikadaysonline',
    'https://t.me/headecom',
    'https://t.me/RetailTechIN',
    'https://t.me/hikollegi',
    'https://t.me/NotBorringPPT',
    'https://t.me/cmoclub',
    'https://t.me/upgradeewdn',
    'https://t.me/ecomnews',
    'https://t.me/kuper_ru',
    'https://t.me/x5dialog',
    'https://t.me/t_bank_data',
    'https://t.me/nielsenrussia',
    'https://t.me/adindex',
    'https://t.me/b_retail',
    'https://t.me/retail_ru',
    'https://t.me/retailerswift'
]

# Date range for analysis
end_date = pd.to_datetime("2025-09-30").tz_localize('UTC')
start_date = pd.to_datetime("2025-09-01").tz_localize('UTC')

print(f"📅 Period: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
print(f"📺 Channels: {len(channel_usernames)}")

# %% [markdown]
# ## 🌐 Parse Telegram Channels

# %%
all_posts = []

async def parse_channels():
    """Parse messages from all configured channels"""
    async with TelegramClient(session_name, api_id, api_hash) as client:
        for channel in tqdm(channel_usernames, desc="Parsing channels"):
            try:
                async for message in client.iter_messages(channel, offset_date=end_date):
                    if message.date < start_date:
                        break
                    if message.text:
                        all_posts.append({
                            "id": message.id,
                            "date": message.date,
                            "text": message.text,
                            "channel": channel
                        })
            except Exception as e:
                print(f"  ⚠️ Error parsing {channel}: {e}")
                continue

# Run parsing
await parse_channels()

# Create DataFrame
df = pd.DataFrame(all_posts)
df['date'] = pd.to_datetime(df['date'])
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

print(f"\n✅ Collected {len(df_filtered):,} posts from {len(channel_usernames)} channels")

# %% [markdown]
# ## 🧠 Utility Functions

# %%
def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters for Russian"""
    return len(text) // 4


def truncate_summaries(summaries, max_tokens=100000):
    """Keep only most recent summaries within token limit"""
    combined = ""
    kept_summaries = []

    for summary in reversed(summaries):
        test_combined = f"- {summary}\n" + combined
        if estimate_tokens(test_combined) > max_tokens:
            break
        combined = test_combined
        kept_summaries.insert(0, summary)

    if len(kept_summaries) < len(summaries):
        print(f"⚠️ Truncated from {len(summaries)} to {len(kept_summaries)} summaries due to token limit")

    return kept_summaries


def save_checkpoint(relevant_summaries, total_posts):
    """Save intermediate results"""
    checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"

    checkpoint_data = {
        'saved_at': datetime.now().isoformat(),
        'summaries': relevant_summaries,
        'total_posts': total_posts,
        'relevant_count': len(relevant_summaries),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    }

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint():
    """Load checkpoint if exists"""
    checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"

    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"✅ Loaded from checkpoint: {checkpoint['relevant_count']} summaries")
        return checkpoint['summaries']

    return None


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


print("✅ Utility functions loaded")

# %% [markdown]
# ## 📝 Improved Prompts

# %%
def get_improved_prompt_individual(text):
    """Improved prompt for filtering individual posts"""
    return f"""Ты - аналитик российского рынка FMCG и e-commerce. Твоя задача: определить, релевантна ли новость для ежемесячного обзора рынка и извлечь ключевые факты.

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
{text}

**ТВОЙ ОТВЕТ:**"""


def get_improved_prompt_final(summaries, start_date, num_channels, num_summaries):
    """Improved prompt for generating final report"""
    return f"""Ты - старший аналитик российского рынка FMCG и e-commerce. Составь ежемесячный аналитический обзор для руководства компании.

**ЦЕЛЕВАЯ АУДИТОРИЯ:** Топ-менеджмент (CEO, CMO, Head of Sales)
**ТОН:** Деловой, лаконичный, фокус на цифрах и бизнес-импликациях
**ПЕРИОД:** {start_date.strftime('%B %Y')}
**ИСТОЧНИК ДАННЫХ:** {num_summaries} релевантных новостей из {num_channels} отраслевых каналов

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

{summaries}

---

**АНАЛИТИЧЕСКИЙ ОТЧЕТ:**"""


print("✅ Improved prompts loaded")

# %% [markdown]
# ## 🔍 Stage 1: Filter Relevant Posts (Map)

# %%
# Try to load from checkpoint first
checkpoint_summaries = load_checkpoint()

if checkpoint_summaries is not None:
    print(f"📂 Using checkpoint data: {len(checkpoint_summaries)} summaries")
    relevant_summaries = checkpoint_summaries
else:
    print(f"\n🔍 Starting analysis of {len(df_filtered)} posts (Stage 1: Filtering)...")
    relevant_summaries = []
    error_count = 0

    for index, row in tqdm(df_filtered.sort_values(by='date').iterrows(),
                           total=len(df_filtered),
                           desc="Filtering posts"):

        prompt_individual = get_improved_prompt_individual(row['text'])

        try:
            response = generate_with_retry(model_filter, prompt_individual)

            # Validate response
            if not response.text or len(response.text.strip()) < 10:
                continue

            # Check for irrelevant marker
            if "НЕРЕЛЕВАНТНО" not in response.text.upper():
                relevant_summaries.append(response.text.strip())

            # Reduced sleep time - Gemini allows 60 req/min
            time.sleep(0.5)

        except exceptions.ResourceExhausted:
            print(f"\n  ⚠️ Rate limit reached, pausing 10 seconds...")
            time.sleep(10)
            error_count += 1
            continue

        except Exception as e:
            print(f"\n  ❌ Error processing post {row['id']}: {e}")
            error_count += 1
            continue

    print(f"\n✅ Stage 1 complete!")
    print(f"   Total posts: {len(df_filtered):,}")
    print(f"   Relevant: {len(relevant_summaries):,} ({len(relevant_summaries)/len(df_filtered)*100:.1f}%)")
    print(f"   Errors: {error_count}")

    # Save checkpoint
    if len(relevant_summaries) > 0:
        save_checkpoint(relevant_summaries, len(df_filtered))

# %% [markdown]
# ## 📊 Stage 2: Generate Final Report (Reduce)

# %%
if not relevant_summaries:
    print("❌ No relevant news found. Cannot generate report.")
else:
    # Token limit protection
    print(f"\n📏 Checking token limits...")
    combined_summaries = "\n".join(f"- {summary}" for summary in relevant_summaries)
    estimated_tokens = estimate_tokens(combined_summaries)

    print(f"   Summaries: {len(relevant_summaries):,}")
    print(f"   Estimated tokens: {estimated_tokens:,}")

    # Truncate if needed (keeping 100k token buffer)
    if estimated_tokens > 100000:
        print(f"   ⚠️ Token limit exceeded, truncating...")
        relevant_summaries = truncate_summaries(relevant_summaries, max_tokens=100000)
        combined_summaries = "\n".join(f"- {summary}" for summary in relevant_summaries)
        estimated_tokens = estimate_tokens(combined_summaries)
        print(f"   ✅ After truncation: {estimated_tokens:,} tokens")

    # Generate final report
    print(f"\n📝 Generating final report (Stage 2: Synthesis)...")

    prompt_final_report = get_improved_prompt_final(
        summaries=combined_summaries,
        start_date=start_date,
        num_channels=len(channel_usernames),
        num_summaries=len(relevant_summaries)
    )

    try:
        final_response = generate_with_retry(model_report, prompt_final_report)

        print("\n" + "="*70)
        print("          ИТОГОВЫЙ АНАЛИТИЧЕСКИЙ ОТЧЕТ")
        print("="*70 + "\n")
        print(final_response.text)
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error generating final report: {e}")
        final_response = None

# %% [markdown]
# ## 📊 Quality Metrics

# %%
if 'final_response' in locals() and final_response is not None:
    quality_metrics = {
        'total_posts': len(df_filtered),
        'relevant_posts': len(relevant_summaries),
        'relevance_rate': len(relevant_summaries) / len(df_filtered) * 100,
        'report_length_chars': len(final_response.text),
        'report_length_words': len(final_response.text.split()),
        'sections_count': final_response.text.count('🔸'),
        'companies_mentioned': sum(1 for line in final_response.text.split('\n') if '**' in line),
        'period': f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}",
        'channels': len(channel_usernames),
    }

    print("\n📊 Quality Metrics:")
    print("="*50)
    for key, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")

    # Save metrics
    metrics_file = f"metrics_{start_date.strftime('%Y%m')}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(quality_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Metrics saved: {metrics_file}")

# %% [markdown]
# ## 📝 Send Report to Apple Notes

# %%
# Import the helper module
from apple_notes_helper import send_monthly_review_to_notes

# Check if we have a report to send
if 'final_response' in locals() and final_response is not None and hasattr(final_response, 'text'):
    print("\n📤 Sending report to Apple Notes...")

    # Send to Apple Notes
    success = send_monthly_review_to_notes(
        report_text=final_response.text,
        start_date=start_date,
        end_date=end_date,
        folder="Monthly Reviews"
    )

    if success:
        print("\n✅ Report successfully saved to Apple Notes!")
        print("   You can find it in the 'Monthly Reviews' folder")
    else:
        print("\n⚠️ Failed to save to Apple Notes. Check the error messages above.")
else:
    print("\n⚠️ No report found. Please run the previous cells first to generate the report.")

# %% [markdown]
# ## 🎉 Complete!
#
# **Summary:**
# - ✅ Parsed Telegram channels
# - ✅ Filtered relevant posts with improved prompts
# - ✅ Generated comprehensive report
# - ✅ Saved metrics and checkpoints
# - ✅ Sent to Apple Notes
#
# **Files created:**
# - `checkpoint_YYYYMM.json` - Checkpoint for recovery
# - `metrics_YYYYMM.json` - Quality metrics
# - Apple Notes entry in "Monthly Reviews" folder
