# Run this BEFORE Cell 9 to diagnose the issue

import os
from pathlib import Path

print("=== DIAGNOSTIC CHECK ===\n")

# Check if df_filtered exists
if 'df_filtered' in locals() or 'df_filtered' in globals():
    print(f"✅ df_filtered exists: {len(df_filtered)} posts")
else:
    print("❌ df_filtered NOT FOUND - Run Cell 6 first!")

# Check for checkpoint file
checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"
if Path(checkpoint_file).exists():
    print(f"\n⚠️  CHECKPOINT FILE EXISTS: {checkpoint_file}")
    print(f"   This will SKIP processing and load cached results!")
    print(f"\n   To force re-processing, delete it:")
    print(f"   !rm {checkpoint_file}")
else:
    print(f"\n✅ No checkpoint file found - will process all posts")

# Check if start_date exists
if 'start_date' in locals() or 'start_date' in globals():
    print(f"\n✅ start_date: {start_date.strftime('%Y-%m-%d')}")
else:
    print("\n❌ start_date NOT FOUND - Run Cell 5 first!")

# Check if functions are defined
if 'get_improved_prompt_individual' in dir():
    print(f"✅ Prompt function exists")
else:
    print(f"❌ get_improved_prompt_individual NOT FOUND - Run Cell 8 first!")

print("\n=== END DIAGNOSTIC ===")
