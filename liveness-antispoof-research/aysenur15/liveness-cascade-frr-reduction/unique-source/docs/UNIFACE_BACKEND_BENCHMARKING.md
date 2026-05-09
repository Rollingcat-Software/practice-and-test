# UniFace Backend Benchmarking

Use `scripts/compare_liveness_backends.py` to compare the current `enhanced` and `uniface` backends on the same fixture set and persist the raw output for auditability.

Example:

```powershell
python scripts/compare_liveness_backends.py --limit 20 --output logs/uniface_benchmark.json
```

The saved JSON file is the benchmark artifact to keep alongside any reported summary numbers such as average scores or pass counts.
