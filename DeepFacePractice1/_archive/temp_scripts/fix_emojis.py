"""Quick script to remove emojis and make files Windows-safe"""

from pathlib import Path

# Files to fix
files_to_fix = [
    "src/demos/demo_2_analysis.py",
    "src/demos/demo_3_embeddings.py",
    "quick_start.py",
]

# Emoji replacements
replacements = {
    r'🎓': '',
    r'✅': '[OK]',
    r'⚠️': '[WARN]',
    r'⚠': '[WARN]',
    r'✓': '[OK]',
    r'✗': '[FAIL]',
    r'🎉': '=',
    r'🚀': '>>',
    r'☕': '',
    r'❌': '[ERROR]',
}

for file_path in files_to_fix:
    if not Path(file_path).exists():
        print(f"Skipping {file_path} - not found")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")
    else:
        print(f"No changes needed for {file_path}")

print("\nDone!")
