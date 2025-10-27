# Archive Folder

This folder contains files that were moved during project cleanup on 2025-10-21.

All files here are **redundant or outdated** but kept for reference/recovery purposes.

## Archive Structure

### `/backups/` - Backup & Migration Files

- `demo_1_verification_original.py.backup` - Original demo backup
- `migration_mapping.json` - Old migration data
- `images_backup/` - Old image folder structure (replaced by person_XXXX format)

### `/old_reports/` - Previous Report Packages

- `face_verification_report_20251020_233001.zip` - Compressed report
- `face_verification_report_20251020_233001/` - Extracted report folder

### `/temp_scripts/` - Temporary/One-time Scripts

- `fix_emojis.py` - Emoji encoding fix (already applied)
- `fix_image_names.py` - Image renaming script (already applied)
- `migrate_images.py` - Migration script (already completed)
- `test_installation.py` - Basic installation test
- `simple_test.py` - Simple verification test

### `/duplicate_utils/` - Replaced Utility Files

- `advanced_visualizer.py` - Replaced by `image_inspection_tool.py`
- `interactive_viewer.py` - Merged into `visualizer.py`
- `optimized_viewer.py` - Merged into `visualizer.py`

### `/old_entry_points/` - Deprecated Entry Scripts

- `main.py` - Replaced by `quick_start.py`
- `run_demo.bat` - Outdated batch launcher
- `run_test.bat` - Outdated test runner
- `create_shareable_package.py` - One-time packaging script
- `generate_full_report.py` - Replaced by `run_quality_inspection.py`

### `/old_outputs/` - Old Output Files

- `interactive_viewer.html` - Old viewer version
- `viewer.html` - Old viewer version

## Recovery

If you need to recover any file:

1. Copy it from this archive folder
2. Move it back to the project root or appropriate location
3. Check for any import/path updates needed

## Safe to Delete

This entire `_archive/` folder can be safely deleted at any time without affecting the main project functionality.

---
*Archived on: 2025-10-21*
*Archive created by: Claude Code Cleanup*
