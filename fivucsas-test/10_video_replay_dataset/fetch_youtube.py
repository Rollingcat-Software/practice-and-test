"""
Fetch YouTube face videos as video-INJECTION attack samples for PAD testing.

Downloads talking-head / vlog clips with yt-dlp, trims each to a short window
(default 15s), and writes them to ./youtube_injection/ with filenames containing
the keyword "injection" so batch_video_eval.py labels them as ATTACK.

These raw videos simulate a direct video-injection attack (virtual webcam /
DeepFaceLive style) — NOT a screen-replay attack. Feeding them straight into the
detector measures APCER-injection: how often injected video passes as LIVE.

Requirements
------------
    pip install yt-dlp
    ffmpeg on PATH (for trimming).  Windows: `winget install Gyan.FFmpeg` or
    download from https://www.gyan.dev/ffmpeg/builds/ and add bin/ to PATH.

Usage
-----
    # From a URL list (one URL per line, # for comments):
    python fetch_youtube.py --urls youtube_urls.txt --seconds 15

    # Or via search (uses yt-dlp ytsearch; results vary, prefer explicit URLs):
    python fetch_youtube.py --search "face interview talking head" --count 20

Output
------
    youtube_injection/injection_001.mp4 ...
    youtube_injection/manifest.csv   (index, source, file, status)
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

# Call yt-dlp via the current interpreter so it works even when the yt-dlp.exe
# shim is not on PATH (common with --user installs on Windows).
YTDLP = [sys.executable, "-m", "yt_dlp"]

# ffmpeg location (auto-detected). yt-dlp needs ffmpeg for trimming/recoding.
FFMPEG_DIR = None


def find_ffmpeg():
    """Return the dir containing ffmpeg, or None. Checks PATH then winget."""
    p = shutil.which("ffmpeg")
    if p:
        return str(Path(p).parent)
    import glob
    pat = str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
              / "*FFmpeg*" / "ffmpeg-*-full_build" / "bin" / "ffmpeg.exe")
    hits = glob.glob(pat)
    return str(Path(hits[0]).parent) if hits else None


def ffmpeg_args():
    return ["--ffmpeg-location", FFMPEG_DIR] if FFMPEG_DIR else []

OUT_DIR  = Path(__file__).parent / "youtube_injection"
URLS_DEF = Path(__file__).parent / "youtube_urls.txt"
MANIFEST = None  # set after OUT_DIR resolved


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def read_urls(path: Path):
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def resolve_search(query: str, n: int):
    """Expand a search query into up to n real YouTube video URLs (live, current)."""
    cmd = YTDLP + ["--flat-playlist", "--no-warnings",
                   "--print", "https://www.youtube.com/watch?v=%(id)s",
                   f"ytsearch{n}:{query}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        urls = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith("http")]
        return urls
    except Exception as e:
        print(f"  search failed for '{query}': {e}")
        return []


def download_one(source: str, out_path: Path, start: int, seconds: int, res_cap: int):
    """Download + trim one video. Returns (ok, message)."""
    section = f"*{start}-{start + seconds}"
    cmd = YTDLP + [
        "-f", f"bestvideo[height<={res_cap}][ext=mp4]+bestaudio/best[height<={res_cap}]",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--recode-video", "mp4",
        *ffmpeg_args(),
        "-o", str(out_path),
        source,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and out_path.exists():
            return True, "ok"
        # yt-dlp sometimes appends extension; check glob
        hits = list(out_path.parent.glob(out_path.stem + ".*"))
        if hits:
            return True, "ok"
        return False, (r.stderr or r.stdout or "unknown error")[-200:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", type=str, default=str(URLS_DEF),
                    help="Text file with one YouTube URL per line")
    ap.add_argument("--search", type=str, default=None,
                    help="yt-dlp ytsearch query instead of a URL file")
    ap.add_argument("--count", type=int, default=20,
                    help="How many results to fetch when using --search")
    ap.add_argument("--queries-file", type=str, default=None,
                    help="Text file with one search QUERY per line (multi-query mode)")
    ap.add_argument("--per-query", type=int, default=4,
                    help="Videos to fetch per query in --queries-file mode")
    ap.add_argument("--seconds", type=int, default=15, help="Clip length per video")
    ap.add_argument("--start", type=int, default=10, help="Start offset (skip intros)")
    ap.add_argument("--res-cap", type=int, default=720, help="Max video height")
    args = ap.parse_args()

    try:
        subprocess.run(YTDLP + ["--version"], capture_output=True, timeout=30, check=True)
    except Exception:
        print("yt-dlp not available. Install with:  pip install yt-dlp")
        sys.exit(1)
    global FFMPEG_DIR
    FFMPEG_DIR = find_ffmpeg()
    if FFMPEG_DIR:
        print(f"ffmpeg: {FFMPEG_DIR}")
    else:
        print("WARNING: ffmpeg not found — trimming/recoding will fail. "
              "Install with: winget install Gyan.FFmpeg")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.csv"

    # Build the source list
    if args.queries_file:
        qpath = Path(args.queries_file)
        if not qpath.exists():
            print(f"Queries file not found: {qpath}"); sys.exit(1)
        queries = read_urls(qpath)  # same parser (one per line, # comments)
        print(f"Resolving {len(queries)} queries x {args.per_query} videos each ...")
        sources = []
        seen = set()
        for q in queries:
            urls = resolve_search(q, args.per_query)
            for u in urls:
                if u not in seen:
                    seen.add(u); sources.append(u)
            print(f"  '{q}' -> {len(urls)} videos")
        print(f"Total unique videos resolved: {len(sources)}")
        args.search = None  # force per-URL download path below
    elif args.search:
        sources = [f"ytsearch{args.count}:{args.search}"]
        # When searching, yt-dlp expands into multiple entries; we let it write
        # numbered files via the output template below.
        print(f"Search mode: ytsearch{args.count}:{args.search}")
    else:
        urls_path = Path(args.urls)
        if not urls_path.exists():
            print(f"URL file not found: {urls_path}\n"
                  f"Create it (one URL per line) or use --search.")
            sys.exit(1)
        sources = read_urls(urls_path)
        if not sources:
            print("URL file is empty. Add YouTube URLs (one per line).")
            sys.exit(1)
        print(f"Loaded {len(sources)} URLs from {urls_path}")

    rows = []

    if args.search:
        # Single yt-dlp call that expands the search into numbered files
        tmpl = str(OUT_DIR / "injection_%(autonumber)03d.%(ext)s")
        cmd = YTDLP + [
            "-f", f"bestvideo[height<={args.res_cap}][ext=mp4]+bestaudio/best[height<={args.res_cap}]",
            "--download-sections", f"*{args.start}-{args.start + args.seconds}",
            "--force-keyframes-at-cuts", "--no-playlist",
            "--recode-video", "mp4",
            "--autonumber-start", "1",
            *ffmpeg_args(),
            "-o", tmpl,
            sources[0],
        ]
        print("Running search download (this may take a while)...")
        subprocess.run(cmd)
        files = sorted(OUT_DIR.glob("injection_*.mp4"))
        for f in files:
            rows.append({"index": f.stem, "source": args.search, "file": f.name, "status": "ok"})
    else:
        for i, url in enumerate(sources, 1):
            out_path = OUT_DIR / f"injection_{i:03d}.mp4"
            ok, msg = download_one(url, out_path, args.start, args.seconds, args.res_cap)
            print(f"  [{i}/{len(sources)}] {'OK' if ok else 'FAIL'}  {url}  {('' if ok else msg)}")
            rows.append({"index": f"injection_{i:03d}", "source": url,
                         "file": out_path.name if ok else "", "status": "ok" if ok else msg})

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "source", "file", "status"])
        w.writeheader(); w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDone: {n_ok}/{len(rows)} videos in {OUT_DIR}")
    print(f"Manifest: {manifest}")
    print(f"\nNext:\n  python batch_video_eval.py --dataset \"{OUT_DIR}\" "
          f"--attack-keyword injection --frame-stride 2 --max-frames 300")


if __name__ == "__main__":
    main()
