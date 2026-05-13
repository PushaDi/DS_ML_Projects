"""
ULTRA-FAST filtering using concurrent API calls
Processes 2800 posts in ~5-10 minutes instead of 4+ hours!
"""

import asyncio
import time
from typing import List, Dict
import pandas as pd
from tqdm.asyncio import tqdm
import google.generativeai as genai
from google.api_core import retry, exceptions

# Configure Gemini
genai.configure(api_key='AIzaSyAcw-ZQLNdY7j7YdoDLqp8iNcou6lLkZt4')

# Use FASTEST model for filtering
model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Ultra fast!

# For final report, use smarter model
model_report = genai.GenerativeModel('gemini-2.5-flash')  # Balanced


def get_improved_prompt_individual(text):
    """Improved filtering prompt"""
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


async def process_single_post(text: str, index: int, semaphore: asyncio.Semaphore) -> Dict:
    """
    Process a single post with rate limiting
    Uses semaphore to limit concurrent requests
    """
    async with semaphore:
        try:
            prompt = get_improved_prompt_individual(text)

            # Generate content (this is sync, but we wrap it)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )

            # Check if relevant
            if response.text and "НЕРЕЛЕВАНТНО" not in response.text.upper():
                return {
                    'index': index,
                    'summary': response.text.strip(),
                    'relevant': True
                }
            else:
                return {
                    'index': index,
                    'summary': None,
                    'relevant': False
                }

        except exceptions.ResourceExhausted:
            # Rate limit hit - wait and retry
            await asyncio.sleep(5)
            return await process_single_post(text, index, semaphore)

        except Exception as e:
            print(f"\n  ❌ Error processing post {index}: {e}")
            return {
                'index': index,
                'summary': None,
                'relevant': False,
                'error': str(e)
            }


async def process_posts_concurrent(df: pd.DataFrame, max_concurrent: int = 50) -> List[str]:
    """
    Process all posts concurrently with rate limiting

    Args:
        df: DataFrame with posts
        max_concurrent: Maximum concurrent API calls (default 50)

    Returns:
        List of relevant summaries
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all posts
    tasks = []
    for index, row in df.iterrows():
        task = process_single_post(row['text'], index, semaphore)
        tasks.append(task)

    # Process with progress bar
    print(f"\n🚀 Processing {len(tasks)} posts with {max_concurrent} concurrent requests...")
    results = []

    # Use tqdm for progress tracking
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Filtering"):
        result = await coro
        results.append(result)

    # Extract relevant summaries
    relevant_summaries = [
        r['summary'] for r in sorted(results, key=lambda x: x['index'])
        if r['relevant'] and r['summary']
    ]

    # Count errors
    error_count = sum(1 for r in results if 'error' in r)

    print(f"\n✅ Complete!")
    print(f"   Total posts: {len(df):,}")
    print(f"   Relevant: {len(relevant_summaries):,} ({len(relevant_summaries)/len(df)*100:.1f}%)")
    print(f"   Errors: {error_count}")

    return relevant_summaries


def estimate_time(num_posts: int, concurrent: int) -> float:
    """
    Estimate processing time

    Assumptions:
    - Average API call: 1-2 seconds
    - With concurrency: total_time ≈ (num_posts / concurrent) * avg_call_time
    """
    avg_call_time = 1.5  # seconds
    estimated_time = (num_posts / concurrent) * avg_call_time
    return estimated_time / 60  # return in minutes


# Example usage
async def main():
    """Main execution function"""

    # Load your data
    import os
    from pathlib import Path

    data_dir = Path(__file__).parent

    # Try to find the latest parsed data
    csv_files = list(data_dir.glob("telegram_messages_*.csv"))
    if not csv_files:
        print("❌ No telegram data found. Run the parsing first.")
        return

    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading data from: {latest_file}")

    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])

    # Filter by date (adjust as needed)
    start_date = pd.to_datetime("2025-09-01").tz_localize('UTC')
    end_date = pd.to_datetime("2025-09-30").tz_localize('UTC')
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    print(f"📊 Found {len(df_filtered):,} posts to process")

    # Estimate time
    concurrent_limit = 50  # Adjust based on your API quota
    estimated_mins = estimate_time(len(df_filtered), concurrent_limit)
    print(f"⏱️  Estimated time: ~{estimated_mins:.1f} minutes (with {concurrent_limit} concurrent requests)")
    print(f"   Compare to sequential: ~{len(df_filtered)*1.5/60:.1f} minutes\n")

    # Process concurrently
    start_time = time.time()
    relevant_summaries = await process_posts_concurrent(df_filtered, max_concurrent=concurrent_limit)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Actual time: {elapsed_time/60:.1f} minutes")
    print(f"   Speed-up: {(len(df_filtered)*1.5/elapsed_time):.1f}x faster than sequential!")

    # Save results
    import json
    output_file = f"relevant_summaries_{start_date.strftime('%Y%m')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summaries': relevant_summaries,
            'total_posts': len(df_filtered),
            'relevant_count': len(relevant_summaries),
            'processing_time_seconds': elapsed_time,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved to: {output_file}")

    return relevant_summaries


if __name__ == "__main__":
    # Run the async main function
    import nest_asyncio
    nest_asyncio.apply()  # Allow nested event loops in Jupyter

    asyncio.run(main())
