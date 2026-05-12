#!/usr/bin/env python3
"""
Photo indexer + quality scorer for @jeanseestheworld blog.

Usage:
  python tools/index-photos.py [--photos assets/photos/] [--force]
  python tools/index-photos.py --chunk-size 200 --workers 8  # large volumes

Pipeline:
  1. Scan files lazily with os.scandir (streaming, no full list in memory)
  2. Extract EXIF + generate thumbnails in parallel (ProcessPoolExecutor)
  3. Submit one Batch API job per chunk (default 500 photos/chunk)
  4. Write index after each chunk — safe to interrupt and resume
  5. Photos scoring < 7.0 are deleted automatically

Requires: pip install anthropic Pillow piexif
"""

import argparse
import base64
import concurrent.futures
import itertools
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Load .env from repo root if present (before any API client is instantiated)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from PIL import Image, ImageOps
    import piexif
except ImportError:
    sys.exit("Missing deps — run: pip install Pillow piexif")

try:
    import anthropic
except ImportError:
    sys.exit("Missing dep — run: pip install anthropic")


REPO_ROOT   = Path(__file__).parent.parent
PHOTOS_DIR  = REPO_ROOT / "assets" / "photos"
OUTPUT_PATH = REPO_ROOT / "_photos" / "index.json"

SUPPORTED     = {".jpg", ".jpeg", ".png", ".webp"}
SCORE_SIZE    = 512                  # px for Vision scoring thumbnail
SCORE_QUALITY = 70                   # JPEG quality for scoring thumbnail
POLL_INTERVAL = 30                   # seconds between batch status checks
CHUNK_SIZE    = 500                  # photos per batch submission
MAX_WORKERS   = os.cpu_count() or 4  # parallel thumbnail/EXIF workers


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def extract_exif(path: Path) -> dict:
    result = {"width": 0, "height": 0, "orientation": "unknown",
              "date": None, "coordinates": None}
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            result["width"], result["height"] = img.size
            result["orientation"] = "portrait" if img.height > img.width else "landscape"
    except Exception:
        return result

    try:
        exif_data = piexif.load(str(path))
        dt_raw = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        if dt_raw:
            result["date"] = dt_raw.decode().split(" ")[0].replace(":", "-")
        gps = exif_data.get("GPS", {})
        if gps:
            def dms_to_dd(dms, ref):
                d, m, s = [(n / d) for n, d in dms]
                dd = d + m / 60 + s / 3600
                return -dd if ref in (b"S", b"W") else dd
            lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
            lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
            if lat_dms and lon_dms:
                result["coordinates"] = {
                    "lat": round(dms_to_dd(lat_dms, lat_ref), 6),
                    "lng": round(dms_to_dd(lon_dms, lon_ref), 6),
                }
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Resize for Vision scoring
# ---------------------------------------------------------------------------

def thumbnail_b64(path: Path) -> str:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.thumbnail((SCORE_SIZE, SCORE_SIZE), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=SCORE_QUALITY)
        return base64.standard_b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Worker: top-level so it can be pickled by ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _process_file(args: tuple) -> tuple:
    """Extract EXIF + generate thumbnail in a worker process."""
    path_str, repo_root_str = args
    path = Path(path_str)
    repo_root = Path(repo_root_str)
    rel = str(path.relative_to(repo_root))
    exif = extract_exif(path)
    try:
        b64 = thumbnail_b64(path)
    except Exception:
        b64 = None
    return rel, exif, b64


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

SCORING_SYSTEM = (
    "You are a professional travel photography judge. "
    "Score the photo on exactly 4 criteria, each 0.0–2.5 (one decimal). "
    "Return ONLY valid JSON, no explanation."
)

SCORING_PROMPT = """Score this travel photo on 4 criteria (each 0.0–2.5):
- brightness: exposure quality (not too dark, not blown out)
- sharpness: focus and clarity
- composition: framing, rule of thirds, visual balance
- visual_interest: subject appeal, storytelling, scroll-stop power

Also set:
- instagram_safe: true if portrait/square format AND score >= 7.0
- hero_candidate: true if total score >= 8.5

Return JSON only:
{"brightness":X,"sharpness":X,"composition":X,"visual_interest":X,"instagram_safe":bool,"hero_candidate":bool}"""


def build_batch_requests(items: list[tuple]) -> tuple[list[dict], dict[str, str]]:
    """Build Batch API request list from [(rel_path, b64_thumbnail)] tuples."""
    requests = []
    id_map = {}
    for i, (rel, b64) in enumerate(items):
        if b64 is None:
            continue
        custom_id = f"photo_{i:05d}"
        id_map[custom_id] = rel
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "system": SCORING_SYSTEM,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": SCORING_PROMPT},
                    ],
                }],
            },
        })
    return requests, id_map


def submit_and_poll(
    client: anthropic.Anthropic,
    requests: list[dict],
    id_map: dict[str, str],
) -> dict[str, dict]:
    print(f"\nSubmitting batch of {len(requests)} photos to Haiku Vision...")
    for attempt in range(1, 6):
        try:
            batch = client.messages.batches.create(requests=requests)
            break
        except Exception as e:
            if attempt == 5:
                raise
            wait = 30 * attempt
            print(f"  [!] Batch submit failed (attempt {attempt}/5): {e} — retrying in {wait}s...")
            time.sleep(wait)
    batch_id = batch.id
    print(f"Batch ID: {batch_id} — polling every {POLL_INTERVAL}s...")

    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        print(
            f"  processing={counts.processing}  "
            f"succeeded={counts.succeeded}  "
            f"errored={counts.errored}  "
            f"expired={counts.expired}"
        )
        if status.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL)

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            try:
                text = result.result.message.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                file_path = id_map.get(result.custom_id, result.custom_id)
                results[file_path] = json.loads(text)
            except Exception as e:
                print(f"  [!] Parse error for {result.custom_id}: {e}")
        else:
            print(f"  [!] Failed: {id_map.get(result.custom_id, result.custom_id)} — {result.result.type}")

    return results


# ---------------------------------------------------------------------------
# Fallback: single-photo scoring (--no-batch)
# ---------------------------------------------------------------------------

def score_single(client: anthropic.Anthropic, photo_path: Path) -> dict:
    img_b64 = thumbnail_b64(photo_path)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=SCORING_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": SCORING_PROMPT},
            ],
        }],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Index writer
# ---------------------------------------------------------------------------

def _write_index(photos: list) -> None:
    usable_count = sum(1 for p in photos if p.get("quality", {}).get("usable"))
    hero_count   = sum(1 for p in photos if p.get("quality", {}).get("hero_candidate"))
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total":         len(photos),
        "usable":        usable_count,
        "hero_candidates": hero_count,
        "photos":        photos,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Index and score travel photos")
    parser.add_argument("--photos",     type=Path, default=PHOTOS_DIR,  help="Photos directory")
    parser.add_argument("--force",      action="store_true",             help="Re-score already indexed photos")
    parser.add_argument("--no-batch",   action="store_true",             help="Score one by one (no async batch)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,   help=f"Photos per batch job (default {CHUNK_SIZE})")
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS,  help=f"Parallel thumbnail workers (default {MAX_WORKERS})")
    args = parser.parse_args()

    if not args.photos.exists():
        sys.exit(f"Photos directory not found: {args.photos}")

    client = anthropic.Anthropic()

    # Load existing index so already-scored photos are skipped
    all_photos: list[dict] = []
    existing:   dict[str, dict] = {}
    if OUTPUT_PATH.exists() and not args.force:
        data     = json.loads(OUTPUT_PATH.read_text())
        all_photos = data.get("photos", [])
        existing   = {p["file"]: p for p in all_photos}
        print(f"Existing index: {len(existing)} photos already scored")

    # Generator — lazily yield paths of unseen files (no full list in memory)
    def iter_new_files():
        for entry in os.scandir(args.photos):
            if Path(entry.name).suffix.lower() in SUPPORTED:
                rel = str(Path(entry.path).relative_to(REPO_ROOT))
                if rel not in existing or args.force:
                    yield entry.path

    chunk_num = total_new = total_del = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        file_gen = iter_new_files()

        while True:
            batch_paths = list(itertools.islice(file_gen, args.chunk_size))
            if not batch_paths:
                break

            chunk_num += 1
            print(f"\nChunk {chunk_num} — {len(batch_paths)} photos"
                  f" — EXIF + thumbnails ({args.workers} workers)...")

            # Parallel EXIF extraction + thumbnail generation
            worker_args = [(p, str(REPO_ROOT)) for p in batch_paths]
            processed   = list(executor.map(_process_file, worker_args))

            # Build per-photo metadata map and flat (rel, b64) list for batch
            meta_map:   dict[str, dict]  = {}
            batch_items: list[tuple]     = []
            for rel, exif, b64 in processed:
                meta_map[rel] = {
                    "file":           rel,
                    "width":          exif["width"],
                    "height":         exif["height"],
                    "orientation":    exif["orientation"],
                    "date":           exif["date"],
                    "coordinates":    exif["coordinates"],
                    "destination":    None,
                    "country":        None,
                    "city":           None,
                    "itinerary_match": None,
                    "tags":           [],
                }
                batch_items.append((rel, b64))

            # Score
            if args.no_batch:
                scores: dict[str, dict] = {}
                for i, (rel, _) in enumerate(batch_items, 1):
                    print(f"  [{i}/{len(batch_items)}] {rel}")
                    try:
                        scores[rel] = score_single(client, REPO_ROOT / rel)
                    except Exception as e:
                        print(f"    Error: {e}")
            else:
                batch_requests, id_map = build_batch_requests(batch_items)
                scores = submit_and_poll(client, batch_requests, id_map)

            # Merge scores — keep usable, delete the rest
            chunk_new = chunk_del = 0
            for rel, raw in scores.items():
                meta  = meta_map[rel]
                total = round(
                    raw.get("brightness",     0) +
                    raw.get("sharpness",      0) +
                    raw.get("composition",    0) +
                    raw.get("visual_interest", 0),
                    1,
                )
                meta["quality"] = {
                    "score":          total,
                    "brightness":     raw.get("brightness"),
                    "sharpness":      raw.get("sharpness"),
                    "composition":    raw.get("composition"),
                    "visual_interest": raw.get("visual_interest"),
                    "instagram_safe": raw.get("instagram_safe", False),
                    "hero_candidate": raw.get("hero_candidate", False),
                    "usable":         total >= 7.0,
                }
                if total >= 7.0:
                    all_photos.append(meta)
                    chunk_new += 1
                else:
                    file_path = REPO_ROOT / rel
                    if file_path.exists():
                        file_path.unlink()
                    chunk_del += 1

            total_new += chunk_new
            total_del += chunk_del

            if chunk_del:
                print(f"  Deleted {chunk_del} photo(s) with score < 7.0")

            # Write index after every chunk — safe to interrupt
            all_photos.sort(
                key=lambda p: p.get("quality", {}).get("score", 0),
                reverse=True,
            )
            _write_index(all_photos)
            print(f"  ✓ Chunk {chunk_num} done — +{chunk_new} added, index = {len(all_photos)} photos")

    if chunk_num == 0:
        print("Nothing new to score.")
        return

    usable = sum(1 for p in all_photos if p.get("quality", {}).get("usable"))
    heroes = sum(1 for p in all_photos if p.get("quality", {}).get("hero_candidate"))
    print(f"\n✓ Finished — {total_new} added, {total_del} deleted")
    print(f"  Total: {len(all_photos)}  |  Usable (≥7.0): {usable}  |  Hero candidates (≥8.5): {heroes}")


if __name__ == "__main__":
    main()
