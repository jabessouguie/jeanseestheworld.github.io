#!/usr/bin/env python3
"""
Photo resizer — converts originals (JPEG/HEIC) to web-ready WebP at 3 sizes.

Usage:
  python tools/resize.py --input /path/to/originals --output assets/photos/

Output naming:  [stem]-hero.webp | [stem]-content.webp | [stem]-thumbnail.webp
  - hero      : 1920×1080, quality 85  (only for photos with quality_score ≥ 8.5 if index exists)
  - content   : 1200×800,  quality 80  (all photos)
  - thumbnail : 400×300,   quality 75  (all photos)

Requires: pip install Pillow pillow-heif
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow not found — run: pip install Pillow pillow-heif")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    print("Warning: pillow-heif not installed — HEIC files will be skipped")

SIZES = {
    "hero":      (1920, 1080, 85),
    "content":   (1200, 800,  80),
    "thumbnail": (400,  300,  75),
}

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
if HEIF_SUPPORT:
    SUPPORTED_EXT.update({".heic", ".heif"})


def load_quality_index(repo_root: Path) -> dict[str, float]:
    """Return {file_stem: quality_score} from _photos/index.json if it exists."""
    index_path = repo_root / "_photos" / "index.json"
    if not index_path.exists():
        return {}
    data = json.loads(index_path.read_text())
    scores = {}
    for photo in data.get("photos", []):
        stem = Path(photo["file"]).stem
        scores[stem] = photo.get("quality", {}).get("score", 0)
    return scores


def resize_photo(src: Path, out_dir: Path, quality_score: float) -> dict[str, str]:
    """Resize one photo to all applicable sizes. Returns {size_name: output_path}."""
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        print(f"  [!] Cannot open {src.name}: {e}")
        return {}

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    written = {}

    for size_name, (max_w, max_h, quality) in SIZES.items():
        if size_name == "hero" and quality_score < 8.5:
            continue

        resized = img.copy()
        resized.thumbnail((max_w, max_h), Image.LANCZOS)

        out_path = out_dir / f"{stem}-{size_name}.webp"
        resized.save(str(out_path), "WEBP", quality=quality)
        written[size_name] = str(out_path)

    return written


def main():
    parser = argparse.ArgumentParser(description="Resize travel photos to web-ready WebP")
    parser.add_argument("--input",  required=True, type=Path, help="Source folder (originals)")
    parser.add_argument("--output", required=True, type=Path, help="Destination folder")
    parser.add_argument("--force",  action="store_true", help="Re-generate even if output exists")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input folder not found: {args.input}")

    # Load quality scores if index exists
    repo_root = Path(__file__).parent.parent
    quality_scores = load_quality_index(repo_root)

    sources = sorted(
        f for f in args.input.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    )

    if not sources:
        print(f"No supported image files found in {args.input}")
        return

    print(f"Found {len(sources)} photos — resizing to WebP...")
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    for src in sources:
        stem = src.stem
        # Check if content version already exists (proxy for "already processed")
        content_path = args.output / f"{stem}-content.webp"
        if content_path.exists() and not args.force:
            stats["skipped"] += 1
            continue

        quality_score = quality_scores.get(stem, 0.0)
        result = resize_photo(src, args.output, quality_score)

        if result:
            sizes_written = ", ".join(result.keys())
            score_label = f"score={quality_score:.1f}" if quality_score else "unscored"
            print(f"  ✓ {src.name} [{score_label}] → {sizes_written}")
            stats["processed"] += 1
        else:
            stats["errors"] += 1

    print(
        f"\nDone — {stats['processed']} processed, "
        f"{stats['skipped']} skipped (already exist), "
        f"{stats['errors']} errors"
    )
    print(f"Output: {args.output.resolve()}")

    # Storage estimate
    total_kb = sum(
        f.stat().st_size // 1024
        for f in args.output.glob("*.webp")
    )
    print(f"Total output size: ~{total_kb // 1024} MB")


if __name__ == "__main__":
    main()
