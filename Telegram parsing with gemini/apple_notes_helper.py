"""
Helper module to send content to Apple Notes.
"""

import subprocess
import re
from datetime import datetime


def markdown_to_html(text):
    """
    Convert simple markdown to HTML for Apple Notes.
    Handles: headers, bold, lists, links, code blocks.
    """
    html = text

    # Code blocks (```...```)
    html = re.sub(r'```(.*?)```', r'<pre>\1</pre>', html, flags=re.DOTALL)

    # Inline code (`...`)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Headers (### -> h3, ## -> h2, # -> h1)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold (**text** or __text__)
    html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
    html = re.sub(r'__(.+?)__', r'<b>\1</b>', html)

    # Italic (*text* or _text_)
    html = re.sub(r'\*(.+?)\*', r'<i>\1</i>', html)
    html = re.sub(r'_(.+?)_', r'<i>\1</i>', html)

    # Links [text](url)
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)

    # Bullet lists (lines starting with * or -)
    lines = html.split('\n')
    in_list = False
    result = []

    for line in lines:
        if re.match(r'^[\*\-] ', line):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item = re.sub(r'^[\*\-] ', '', line)
            result.append(f'<li>{item}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)

    if in_list:
        result.append('</ul>')

    html = '\n'.join(result)

    # Line breaks
    html = html.replace('\n\n', '<br><br>')
    html = html.replace('\n', '<br>')

    return html


def send_to_apple_notes(content, title=None, folder="Notes"):
    """
    Send content to Apple Notes.

    Args:
        content (str): The content to add (supports markdown)
        title (str): Optional title for the note. If None, uses first line of content
        folder (str): Apple Notes folder name (default: "Notes")

    Returns:
        bool: True if successful, False otherwise
    """

    # Auto-generate title if not provided
    if title is None:
        first_line = content.split('\n')[0]
        # Remove markdown formatting from title
        title = re.sub(r'[#*_`]', '', first_line).strip()
        if len(title) > 100:
            title = title[:100] + "..."

    # Convert markdown to HTML
    html_content = markdown_to_html(content)

    # Prepare title for the note (will be the first line)
    note_title_html = f"<h1>{title}</h1><br>"
    full_html = note_title_html + html_content

    # AppleScript to create note
    applescript = f'''
    on run
        tell application "Notes"
            activate

            -- Get or create folder
            set targetFolder to missing value
            repeat with aFolder in folders
                if name of aFolder is "{folder}" then
                    set targetFolder to aFolder
                    exit repeat
                end if
            end repeat

            if targetFolder is missing value then
                set targetFolder to make new folder with properties {{name:"{folder}"}}
            end if

            -- Create note
            tell targetFolder
                make new note with properties {{body:"{escape_for_applescript(full_html)}"}}
            end tell

            return "SUCCESS"
        end tell
    end run
    '''

    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print(f"✅ Note successfully created in Apple Notes folder '{folder}'")
            print(f"   Title: {title}")
            return True
        else:
            print(f"❌ Failed to create note in Apple Notes")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Timeout while creating Apple Note")
        return False
    except Exception as e:
        print(f"❌ Error creating Apple Note: {e}")
        return False


def escape_for_applescript(text):
    """
    Escape special characters for AppleScript string.
    """
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    return text


def send_monthly_review_to_notes(report_text, start_date, end_date, folder="Monthly Reviews"):
    """
    Specialized function for sending monthly review to Apple Notes.

    Args:
        report_text (str): The full report text
        start_date (str or datetime): Start date of review period
        end_date (str or datetime): End date of review period
        folder (str): Apple Notes folder name

    Returns:
        bool: True if successful, False otherwise
    """

    # Format dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Create title
    month_year = start_date.strftime("%B %Y")
    title = f"📊 Monthly News Review - {month_year}"

    # Add metadata header
    header = f"""
**Period:** {start_date.strftime("%d.%m.%Y")} - {end_date.strftime("%d.%m.%Y")}
**Generated:** {datetime.now().strftime("%d.%m.%Y %H:%M")}

---

"""

    full_content = header + report_text

    return send_to_apple_notes(full_content, title=title, folder=folder)


if __name__ == "__main__":
    # Test example
    test_content = """
# Test Report

This is a **bold** statement and this is *italic*.

## Key Points

* First point
* Second point
* Third point

Visit [Google](https://google.com) for more info.
"""

    send_to_apple_notes(test_content, title="Test Note", folder="Tests")
