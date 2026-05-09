"""Integration tests for preview-synchronized collector output.

These tests validate the labeled records produced by
`app/tools/test_data_collector.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def _resolve_test_frames_dir() -> Path:
    """Find the collector output directory, preferring the latest numbered run."""
    data_dir = Path("data")
    if not data_dir.exists():
        return data_dir / "test_frames"

    candidates = [path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("test_frames")]
    if not candidates:
        return data_dir / "test_frames"

    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("test_frames")
        if suffix.isdigit():
            return (int(suffix), path.name)
        if suffix == "":
            return (0, path.name)
        return (-1, path.name)

    return sorted(candidates, key=sort_key)[-1]


TEST_FRAMES_DIR = _resolve_test_frames_dir()
SUMMARY_FILE = TEST_FRAMES_DIR / "summary.jsonl"


def load_test_frames() -> list[dict]:
    """Load collected test frame metadata."""
    if not SUMMARY_FILE.exists():
        pytest.skip("Test data not found. Run: python -m app.tools.test_data_collector")

    frames = []
    with open(SUMMARY_FILE) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def load_preview_temporal_frames() -> list[dict]:
    """Load only records produced by the synchronized preview collector."""
    frames = load_test_frames()
    preview_frames = [frame for frame in frames if frame.get("collector_pipeline") == "preview_temporal"]
    if not preview_frames:
        pytest.skip("No preview_temporal collector frames found. Re-collect data with the updated collector.")
    return preview_frames


def _binary_prediction(frame_meta: dict) -> dict:
    """Return the collector's binary prediction payload."""
    return (frame_meta.get("binary_prediction") or {})


def test_hybrid_fusion_accuracy_on_collected_live_frames():
    """Validate collector predictions on real LIVE frames."""
    frames = load_preview_temporal_frames()
    live_frames = [f for f in frames if f["label"] == "live"]

    if not live_frames:
        pytest.skip("No LIVE frames collected")

    decided_frames = [frame for frame in live_frames if _binary_prediction(frame).get("is_live") is not None]
    if not decided_frames:
        pytest.skip("No LIVE frames with binary prediction yet. Re-collect after the collector update.")

    correct = 0
    total = len(decided_frames)

    for frame_meta in decided_frames:
        prediction = _binary_prediction(frame_meta)
        if bool(prediction.get("is_live")):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    assert accuracy >= 0.80, f"LIVE accuracy {accuracy:.1%} < 80%"


def test_hybrid_fusion_accuracy_on_collected_spoof_frames():
    """Validate collector predictions on real SPOOF frames."""
    frames = load_preview_temporal_frames()
    spoof_frames = [f for f in frames if f["label"] == "spoof"]

    if not spoof_frames:
        pytest.skip("No SPOOF frames collected")

    decided_frames = [frame for frame in spoof_frames if _binary_prediction(frame).get("is_live") is not None]
    if not decided_frames:
        pytest.skip("No SPOOF frames with binary prediction yet. Re-collect after the collector update.")

    correct = 0
    total = len(decided_frames)

    for frame_meta in decided_frames:
        prediction = _binary_prediction(frame_meta)
        if not bool(prediction.get("is_live")):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    assert accuracy >= 0.80, f"SPOOF accuracy {accuracy:.1%} < 80%"


def test_collector_preview_metadata_present():
    """Verify the updated collector stores synchronized preview metadata."""
    frames = load_preview_temporal_frames()
    if not frames:
        pytest.skip("No frames collected")

    frame_meta = frames[0]
    liveness = frame_meta.get("liveness") or {}
    checks = liveness.get("checks") or {}
    scores = liveness.get("scores") or {}
    metadata = liveness.get("metadata") or {}

    assert frame_meta.get("collector_version")
    assert frame_meta.get("collector_pipeline") == "preview_temporal"
    assert "decision_state" in (frame_meta.get("prediction") or {})
    assert "label" in (frame_meta.get("binary_prediction") or {})
    assert "rppg_available" in checks
    assert "moire_score" in scores
    assert "device_replay_score" in scores
    assert "preview_sample_count" in metadata
    assert frame_meta.get("frame_metrics") is not None
    assert frame_meta.get("aggregate_metrics") is not None


def test_summary_report():
    """Generate accuracy report from collected frames."""
    if not SUMMARY_FILE.exists():
        pytest.skip("No summary file")

    frames = load_test_frames()
    if not frames:
        pytest.skip("No frames in summary")

    live_frames = [f for f in frames if f["label"] == "live"]
    spoof_frames = [f for f in frames if f["label"] == "spoof"]

    print("\n" + "=" * 70)
    print("HYBRID FUSION TEST DATA SUMMARY")
    print("=" * 70)
    print(f"\nTotal frames collected: {len(frames)}")
    print(f"  LIVE:  {len(live_frames)}")
    print(f"  SPOOF: {len(spoof_frames)}")
    print(f"\nTest data location: {TEST_FRAMES_DIR.absolute()}")
    print("=" * 70 + "\n")

    # Provide guidance
    print("📊 Next steps:")
    print(f"1. Review collected frames in {TEST_FRAMES_DIR.as_posix()}/")
    print("2. Run accuracy tests:")
    print("   pytest tests/integration/test_hybrid_fusion_real_data.py -v")
    print("3. Check individual frame metrics:")
    print(f"   cat {SUMMARY_FILE.as_posix()} | jq '.'")
