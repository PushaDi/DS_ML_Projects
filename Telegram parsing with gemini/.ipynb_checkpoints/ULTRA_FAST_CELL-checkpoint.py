# ==========================================
# ULTRA-FAST FILTERING CELL
# Copy-paste this into your notebook to replace the slow filtering cell
# Processes 2800 posts in ~5 minutes instead of 4+ hours!
# ==========================================

import asyncio
from typing import List, Dict
from tqdm.asyncio import tqdm
import nest_asyncio
nest_asyncio.apply()

# Use ULTRA-FAST model for filtering
model_filter = genai.GenerativeModel('gemini-2.5-flash-lite')  # Super fast!

async def process_single_post(text: str, index: int, semaphore: asyncio.Semaphore) -> Dict:
    """Process a single post with rate limiting"""
    async with semaphore:
        try:
            prompt = get_improved_prompt_individual(text)

            # Run sync API call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_filter.generate_content(prompt)
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

        except Exception as e:
            return {
                'index': index,
                'summary': None,
                'relevant': False,
                'error': str(e)
            }


async def process_posts_concurrent(df: pd.DataFrame, max_concurrent: int = 50) -> List[str]:
    """Process all posts concurrently - 40-50x FASTER!"""

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all posts
    tasks = [
        process_single_post(row['text'], index, semaphore)
        for index, row in df.iterrows()
    ]

    print(f"🚀 Processing {len(tasks)} posts with {max_concurrent} concurrent requests...")
    print(f"   Model: gemini-2.5-flash-lite (ultra-fast)")
    print(f"   Estimated time: ~{len(tasks)/(max_concurrent*0.7):.1f} minutes\n")

    results = []

    # Process with progress bar
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Ultra-fast filtering"):
        result = await coro
        results.append(result)

    # Extract relevant summaries (preserve order)
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


# ==========================================
# RUN THE ULTRA-FAST FILTERING
# ==========================================

import time

# Check for checkpoint first
checkpoint_summaries = load_checkpoint()

if checkpoint_summaries is not None:
    print(f"📂 Using checkpoint data: {len(checkpoint_summaries)} summaries")
    relevant_summaries = checkpoint_summaries
else:
    print(f"\n🚀 Starting ULTRA-FAST filtering of {len(df_filtered)} posts...\n")

    start_time = time.time()

    # Run concurrent processing (50 requests at once!)
    relevant_summaries = await process_posts_concurrent(
        df_filtered,
        max_concurrent=50  # Adjust this: 30-60 recommended
    )

    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Processing time: {elapsed_time/60:.2f} minutes")
    print(f"   Speed: {len(df_filtered)/elapsed_time:.1f} posts/second")
    print(f"   Speed-up vs sequential: ~{(len(df_filtered)*1.5/elapsed_time):.0f}x faster!")

    # Save checkpoint
    if len(relevant_summaries) > 0:
        save_checkpoint(relevant_summaries, len(df_filtered))

# ==========================================
# DONE! Continue with Stage 2 (report generation)
# ==========================================
