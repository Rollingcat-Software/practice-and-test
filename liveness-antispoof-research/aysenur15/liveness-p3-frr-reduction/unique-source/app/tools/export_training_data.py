# app/tools/export_training_data.py

"""
Export training data with proper nested JSON parsing.
Only exports frames where liveness detection actually ran.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def extract_features_from_json(data: Dict) -> Optional[Dict]:
    """
    Extract ML features from nested JSON structure.
    Returns None if liveness was skipped.
    """

    # Check if liveness ran
    liveness_skipped = not data.get('liveness', {}).get('checks', {}).get('face_usable', False)

    if liveness_skipped:
        return None  # Skip this frame

    # Extract from nested structure
    liveness_scores = data.get('liveness', {}).get('scores', {})

    # Get features (with fallback to 0.0)
    features = {
        'flicker': liveness_scores.get('flicker_score') or 0.0,
        'device_replay': liveness_scores.get('device_replay_score') or 0.0,
        'moire': liveness_scores.get('moire_score') or 0.0,
        'reflection': liveness_scores.get('reflection_score') or 0.0,
        'depth_flat': 0.0,  # Not in this format
        'cutout': 0.0,  # Not in this format
        'focal_blur': 0.0,  # Not in this format
        'screen_frame': liveness_scores.get('screen_frame_score') or 0.0,
        'reflect_compact': 0.0,  # Not in this format
        'rppg_score': liveness_scores.get('rppg_score') or 0.0,
    }

    # Check if we have actual data (not all zeros/nulls)
    non_zero_count = sum(1 for v in features.values() if v != 0.0)

    if non_zero_count < 3:  # Need at least 3 non-zero features
        return None

    return features


def load_json_file(json_path: Path) -> Dict:
    """Load a single JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_to_csv(input_dir: str = "data/test_frames",
                  output_file: str = "data/training_data.csv"):
    """
    Export all valid JSON files to CSV.
    Only includes frames where liveness detection ran successfully.
    """
    data_path = Path(input_dir)
    json_files = list(data_path.glob("*.json"))

    print(f"📂 Found {len(json_files)} JSON files")

    rows = []
    skipped = 0

    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            features = extract_features_from_json(data)

            if features is None:
                print(f"  ⏭️  {json_file.stem}: SKIPPED (liveness not run)")
                skipped += 1
                continue

            # Add label (from filename or from JSON)
            label = 1 if 'spoof' in json_file.stem.lower() else 0
            features['label'] = label
            features['filename'] = json_file.stem

            rows.append(features)

            print(f"  ✅ {json_file.stem}: {'SPOOF' if label == 1 else 'LIVE'} "
                  f"(flicker={features['flicker']:.2f}, device={features['device_replay']:.2f})")

        except Exception as e:
            print(f"  ❌ {json_file.stem}: ERROR - {e}")
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"📊 EXPORT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total files found: {len(json_files)}")
    print(f"Valid samples: {len(rows)}")
    print(f"Skipped (no liveness): {skipped}")

    if len(rows) == 0:
        print("\n❌ NO VALID SAMPLES! All frames were skipped.")
        print("\nPossible reasons:")
        print("  1. Face quality was too low during collection")
        print("  2. Liveness detection didn't run")
        print("  3. Need to recollect data with better face visibility")
        return None

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    feature_cols = [col for col in df.columns if col not in ['label', 'filename']]
    df = df[['filename'] + feature_cols + ['label']]

    # Save CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Exported to: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   LIVE samples: {(df['label'] == 0).sum()}")
    print(f"   SPOOF samples: {(df['label'] == 1).sum()}")

    # Show feature statistics
    print(f"\n📈 Feature Statistics:")
    for col in feature_cols:
        non_zero = (df[col] != 0.0).sum()
        print(f"  {col:20s}: non-zero={non_zero}/{len(df)}, "
              f"min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")

    print(f"\n📋 Sample data:")
    print(df.head(10))

    return df


if __name__ == "__main__":
    df = export_to_csv()

    if df is not None:
        print("\n✅ Ready for model training!")
    else:
        print("\n❌ Need to recollect data!")