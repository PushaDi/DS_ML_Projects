# Apple Notes Integration

## Overview

The notebook now automatically sends the generated monthly review report to Apple Notes after completion. This makes it easy to access, review, and share your reports across your Apple devices.

## How It Works

### 1. Helper Module

The `apple_notes_helper.py` module provides functions to:
- Convert markdown-formatted text to HTML for better formatting in Apple Notes
- Create notes with proper titles and metadata
- Organize notes into specific folders

### 2. Automatic Integration

After running all notebook cells, the final cell automatically:
1. Takes the generated report from the Gemini API
2. Formats it with metadata (date range, generation time)
3. Creates a note in Apple Notes with title: `📊 Monthly News Review - [Month Year]`
4. Saves it to the "Monthly Reviews" folder

### 3. What Gets Saved

Each note includes:
- **Title:** Formatted as "📊 Monthly News Review - September 2025"
- **Metadata:** Period covered and generation timestamp
- **Full Report:** All sections from the analysis
- **Formatting:** Headers, bold text, lists, and links preserved

## Usage

### Basic Usage

Simply run all cells in the notebook. The last cell will automatically send the report to Apple Notes.

```python
# The last cell does this automatically:
from apple_notes_helper import send_monthly_review_to_notes

send_monthly_review_to_notes(
    report_text=final_response.text,
    start_date=start_date,
    end_date=end_date,
    folder="Monthly Reviews"
)
```

### Customization Options

#### Change the Folder Name

Edit the last cell to use a different folder:

```python
send_monthly_review_to_notes(
    report_text=final_response.text,
    start_date=start_date,
    end_date=end_date,
    folder="Work Reports"  # Your custom folder name
)
```

#### Send to Multiple Folders

Add additional calls to save copies:

```python
# Save to personal folder
send_monthly_review_to_notes(
    report_text=final_response.text,
    start_date=start_date,
    end_date=end_date,
    folder="Monthly Reviews"
)

# Also save to shared folder
send_monthly_review_to_notes(
    report_text=final_response.text,
    start_date=start_date,
    end_date=end_date,
    folder="Team Reports"
)
```

#### Custom Title

Use the more flexible `send_to_apple_notes()` function:

```python
from apple_notes_helper import send_to_apple_notes

custom_title = f"FMCG Report - {start_date.strftime('%B %Y')} - DRAFT"

send_to_apple_notes(
    content=final_response.text,
    title=custom_title,
    folder="Drafts"
)
```

## Advanced Usage

### Manual Sending

You can import and use the helper module in any Python script:

```python
from apple_notes_helper import send_to_apple_notes

# Simple usage
send_to_apple_notes(
    content="Your content here",
    title="Note Title",
    folder="Notes"
)
```

### Markdown Support

The helper automatically converts markdown to HTML. Supported syntax:

```markdown
# Headers (H1)
## Headers (H2)
### Headers (H3)

**Bold text**
*Italic text*

* Bullet lists
- Also bullet lists

[Links](https://example.com)

`inline code`
```

Code blocks (triple backticks) are preserved as `<pre>` blocks.

### Error Handling

The function returns `True` on success and `False` on failure:

```python
success = send_monthly_review_to_notes(
    report_text=final_response.text,
    start_date=start_date,
    end_date=end_date
)

if success:
    print("✅ Saved to Apple Notes")
else:
    print("❌ Failed - check error messages")
```

## Requirements

- **macOS only** - Uses AppleScript to communicate with Apple Notes
- **Apple Notes app** must be installed (comes with macOS)
- **Python 3.6+** with `subprocess` module (standard library)

## Troubleshooting

### "Permission Denied" Error

Grant terminal/Jupyter permission to control Apple Notes:
1. System Settings → Privacy & Security → Automation
2. Enable Apple Notes access for your terminal/Jupyter app

### Notes Not Appearing

- Check if Apple Notes app is running
- Verify the folder name matches exactly (case-sensitive)
- Try creating the folder manually first in Apple Notes

### Formatting Issues

If formatting looks wrong:
- The helper converts markdown to HTML automatically
- Complex markdown may not render perfectly
- Check `apple_notes_helper.py` to customize conversion

### Timeout Errors

If the script times out:
- Close and reopen Apple Notes
- Reduce the size of the report
- Check system resources (RAM, CPU)

## Files

- `apple_notes_helper.py` - Main helper module with all functions
- `monthly_news_review.ipynb` - Notebook with integrated auto-send
- `APPLE_NOTES_INTEGRATION.md` - This documentation file

## Example Output

After running the notebook, you'll see:

```
📤 Sending report to Apple Notes...
✅ Note successfully created in Apple Notes folder 'Monthly Reviews'
   Title: 📊 Monthly News Review - September 2025

✅ Report successfully saved to Apple Notes!
   You can find it in the 'Monthly Reviews' folder
```

Then open Apple Notes and find your report in the "Monthly Reviews" folder with nicely formatted content!

## Tips

1. **Organize by Month:** The automatic title includes the month/year for easy sorting
2. **Search Functionality:** Apple Notes' search works great for finding specific topics across all reports
3. **iCloud Sync:** If iCloud is enabled, reports sync across all your Apple devices
4. **Export Options:** From Apple Notes, you can easily export to PDF or share via email
5. **Archiving:** Create separate folders like "2025 Reviews" to organize by year

## Future Enhancements

Potential improvements:
- Add tags/labels to notes for better organization
- Create summary notes with links to detailed reports
- Integration with Calendar for scheduled reviews
- Export to PDF automatically
- Email distribution after saving to Notes

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your macOS and Python versions
3. Test with the example in `apple_notes_helper.py`
