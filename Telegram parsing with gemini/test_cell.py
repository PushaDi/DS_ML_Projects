# === DIAGNOSTIC TEST - Run this instead of Cell 9 ===

import asyncio
import time

print("=== STARTING DIAGNOSTIC TEST ===\n")

# 1. Check df_filtered
print(f"1. Checking df_filtered...")
print(f"   Rows: {len(df_filtered)}")
print(f"   Columns: {list(df_filtered.columns)}")
print(f"   First text length: {len(df_filtered.iloc[0]['text']) if len(df_filtered) > 0 else 0} chars")

# 2. Check model
print(f"\n2. Testing model initialization...")
try:
    test_model = genai.GenerativeModel('gemma-3-12b-it')
    print(f"   ✅ Model created: {test_model}")
except Exception as e:
    print(f"   ❌ Model error: {e}")

# 3. Test single API call
print(f"\n3. Testing single API call...")
try:
    test_prompt = "Скажи 'тест' одним словом."
    start = time.time()
    response = test_model.generate_content(test_prompt)
    elapsed = time.time() - start
    print(f"   ✅ API works! Response: {response.text[:50]}")
    print(f"   Time: {elapsed:.2f}s")
except Exception as e:
    print(f"   ❌ API error: {type(e).__name__}: {e}")

# 4. Test async function
print(f"\n4. Testing async function...")

async def test_async():
    print(f"   Inside async function...")
    semaphore = asyncio.Semaphore(1)

    async def test_single(text, index):
        async with semaphore:
            print(f"      Processing post {index}...")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: test_model.generate_content("Скажи 'ok'")
            )
            print(f"      Got response for post {index}: {response.text[:20]}")
            return {'index': index, 'text': response.text}

    # Test with 2 posts
    tasks = [test_single(df_filtered.iloc[i]['text'], i) for i in range(min(2, len(df_filtered)))]
    print(f"   Created {len(tasks)} tasks")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"   ✅ Async completed! Results: {len(results)}")
    return results

try:
    start = time.time()
    test_results = await test_async()
    elapsed = time.time() - start
    print(f"   ✅ Async test passed! Time: {elapsed:.2f}s")
    for r in test_results:
        if isinstance(r, Exception):
            print(f"      Error: {r}")
        else:
            print(f"      Result {r['index']}: {r['text'][:30]}")
except Exception as e:
    print(f"   ❌ Async failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== DIAGNOSTIC TEST COMPLETE ===")
