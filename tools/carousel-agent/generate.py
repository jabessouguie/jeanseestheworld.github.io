#!/usr/bin/env python3
"""
Carousel Instagram generator for @jeanseestheworld
Usage: python generate.py --notes notes/trip.txt --photos assets/photos/slug/ --handle @jeanseestheworld --output output/carousel/
All input options combinable: --notes, --html, --claude-export
Priority: notes > user messages (claude export) > assistant messages > html
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not found — run: pip install -r requirements.txt")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow not found — run: pip install -r requirements.txt")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("beautifulsoup4 not found — run: pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
REPO_ROOT    = SCRIPT_DIR.parent.parent
PHOTOS_INDEX = REPO_ROOT / "_photos" / "index.json"
BRAND_JSON   = REPO_ROOT / "_config" / "brand.json"
STYLE_JSON   = SCRIPT_DIR / "templates" / "slide_style.json"
FONTS_DIR    = SCRIPT_DIR / "fonts"
PLANNER_TXT  = SCRIPT_DIR / "prompts" / "planner.txt"
CAPTION_TXT  = SCRIPT_DIR / "prompts" / "caption.txt"

CANVAS_W = 1080
CANVAS_H = 1350

# Token budget caps — prevents unnecessary API spend
MAX_TOKENS = {
    "haiku_extract":  800,   # structured outline from raw notes
    "carousel_plan": 1500,   # JSON plan (10 slides)
    "caption":        500,   # Instagram caption
}
MAX_NOTES_CHARS   = 12000   # ~3 000 tokens
MAX_HTML_CHARS    = 20000   # ~5 000 tokens
MAX_EXPORT_MSGS   = 20      # keep most-recent N messages from Claude export
MAX_PHOTOS_PASSED = 30      # cap photos sent in API context


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate Instagram carousel slides")
    p.add_argument("--notes",        type=Path, help="Plain text trip notes file")
    p.add_argument("--html",         type=Path, help="HTML page to parse (blog post / article)")
    p.add_argument("--claude-export",type=Path, help="Claude conversation export JSON")
    p.add_argument("--photos",       type=Path, help="Directory of resized trip photos")
    p.add_argument("--handle",       default="@jeanseestheworld", help="Instagram handle")
    p.add_argument("--output",       type=Path, default=SCRIPT_DIR / "output", help="Output directory")
    p.add_argument("--dry-run",      action="store_true", help="Plan only, do not render slides")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source collection & merging
# ---------------------------------------------------------------------------
def collect_sources(args) -> str:
    """Collect all text sources and merge by priority order."""
    parts = []

    if args.notes and args.notes.exists():
        text = args.notes.read_text(encoding="utf-8")
        if len(text) > MAX_NOTES_CHARS:
            text = text[:MAX_NOTES_CHARS]
            print(f"  [~] Notes truncated to {MAX_NOTES_CHARS} chars (~3 000 tokens)")
        parts.append(("notes", text))
        print(f"  [+] Notes loaded: {args.notes}")

    if args.claude_export and args.claude_export.exists():
        user_msgs, asst_msgs = parse_claude_export(args.claude_export)
        # Keep only the most-recent N messages (most relevant + cheapest)
        user_msgs  = user_msgs[-MAX_EXPORT_MSGS:]
        asst_msgs  = asst_msgs[-MAX_EXPORT_MSGS:]
        if user_msgs:
            parts.append(("claude_user", "\n\n".join(user_msgs)))
            print(f"  [+] Claude export (user, {len(user_msgs)} messages)")
        if asst_msgs:
            parts.append(("claude_assistant", "\n\n".join(asst_msgs)))
            print(f"  [+] Claude export (assistant, {len(asst_msgs)} messages)")

    if args.html and args.html.exists():
        html_text = parse_html(args.html)
        if len(html_text) > MAX_HTML_CHARS:
            html_text = html_text[:MAX_HTML_CHARS]
            print(f"  [~] HTML truncated to {MAX_HTML_CHARS} chars (~5 000 tokens)")
        parts.append(("html", html_text))
        print(f"  [+] HTML parsed: {args.html}")

    if not parts:
        sys.exit("Error: no input sources provided. Use --notes, --html, or --claude-export.")

    # Merge in priority order (notes first, then user, then assistant, then html)
    order = {"notes": 0, "claude_user": 1, "claude_assistant": 2, "html": 3}
    parts.sort(key=lambda x: order.get(x[0], 99))

    merged = "\n\n---\n\n".join(f"[SOURCE: {label}]\n{text}" for label, text in parts)
    return merged


def parse_html(path: Path) -> str:
    """Extract readable text from HTML, stripping scripts/styles."""
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_claude_export(path: Path):
    """Parse Claude conversation export JSON. Returns (user_messages, assistant_messages)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    user_msgs = []
    asst_msgs = []

    # Handle both array-of-messages and {messages: [...]} formats
    messages = data if isinstance(data, list) else data.get("messages", [])

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-part content
            text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            text = str(content)
        text = text.strip()
        if not text:
            continue
        if role == "user":
            user_msgs.append(text)
        elif role == "assistant":
            asst_msgs.append(text)

    return user_msgs, asst_msgs


# ---------------------------------------------------------------------------
# Photo index loading & filtering
# ---------------------------------------------------------------------------
def load_photos(photos_dir: Path | None, min_score: float = 7.0) -> list[dict]:
    """Load photos from _photos/index.json, filtered by itinerary_match + quality.

    Filters to the relevant itinerary slug first (cheap), then by quality score.
    Caps at MAX_PHOTOS_PASSED to avoid ballooning API context.
    """
    if not PHOTOS_INDEX.exists():
        print("  [!] _photos/index.json not found — no photos available")
        return []

    index = json.loads(PHOTOS_INDEX.read_text(encoding="utf-8"))
    all_photos = index.get("photos", [])

    # Step 1 — filter by itinerary slug (itinerary_match field, not full path scan)
    if photos_dir:
        slug = photos_dir.name
        matched = [
            p for p in all_photos
            if p.get("itinerary_match") == slug
            or slug in p.get("file", "")
        ]
        # Fall back to all photos if slug produces nothing (e.g. unindexed directory)
        if not matched:
            matched = all_photos
            print(f"  [~] No itinerary_match='{slug}' in index — using all photos")
    else:
        matched = all_photos

    # Step 2 — quality filter
    usable = [p for p in matched if p.get("quality", {}).get("score", 0) >= min_score]

    # Step 3 — sort by score desc, cap to avoid token waste
    usable.sort(key=lambda p: p.get("quality", {}).get("score", 0), reverse=True)
    if len(usable) > MAX_PHOTOS_PASSED:
        print(f"  [~] Capping photos at {MAX_PHOTOS_PASSED} (had {len(usable)})")
        usable = usable[:MAX_PHOTOS_PASSED]

    print(f"  [+] Photos available: {len(usable)} (score ≥ {min_score}, slug='{photos_dir.name if photos_dir else 'all'}')")
    return usable


# ---------------------------------------------------------------------------
# Haiku: extract structured data from raw notes
# ---------------------------------------------------------------------------
def extract_structured_data(client: anthropic.Anthropic, merged_notes: str) -> dict:
    """Use Haiku 4.5 to extract key trip data cheaply."""
    print("  [→] Haiku: extracting structured data…")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=MAX_TOKENS["haiku_extract"],
        system="You are a travel data extractor. Extract structured trip information from raw notes. Return only valid JSON.",
        messages=[{
            "role": "user",
            "content": f"""Extract key information from these trip notes and return JSON:

{{
  "destination": "main destination or route name",
  "countries": ["list of countries"],
  "cities": ["list of cities in order"],
  "duration_days": 0,
  "budget_per_day_eur": 0,
  "transport": ["main transport types used"],
  "highlights": ["3-5 key moments or places"],
  "honest_observations": ["1-2 honest negatives or caveats"],
  "interrail_pass_used": "pass name or null",
  "best_for": ["interest tags: culture, nature, food, etc."]
}}

Notes:
{merged_notes[:4000]}"""
        }]
    )
    try:
        raw = response.content[0].text
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  [!] Haiku parse warning: {e} — using fallback")
        return {"destination": "Europe", "cities": [], "highlights": [], "honest_observations": []}


# ---------------------------------------------------------------------------
# Sonnet: generate carousel plan
# ---------------------------------------------------------------------------
def generate_carousel_plan(
    client: anthropic.Anthropic,
    structured_data: dict,
    merged_notes: str,
    photos: list[dict],
    handle: str,
    brand: dict,
    style: dict,
    planner_prompt: str,
) -> dict:
    """Use Sonnet 4.6 to generate the full carousel plan JSON with prompt caching."""
    print("  [→] Sonnet: generating carousel plan…")

    photo_summary = json.dumps(
        [{"filename": p["filename"], "city": p.get("city"), "quality_score": p.get("quality_score"),
          "tags": p.get("tags", []), "mood": p.get("mood"), "subjects": p.get("subjects", [])}
         for p in photos[:30]],
        indent=2
    )

    user_message = f"""Handle: {handle}

Structured trip data:
{json.dumps(structured_data, indent=2)}

Available photos (top {min(30, len(photos))} by quality score):
{photo_summary}

Raw trip notes (excerpt):
{merged_notes[:3000]}

Generate a 7-10 slide carousel plan. Return only valid JSON matching the schema in your instructions."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=MAX_TOKENS["carousel_plan"],
        system=[
            {
                "type": "text",
                "text": planner_prompt,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": f"Brand config:\n{json.dumps(brand, indent=2)}\n\nSlide style:\n{json.dumps(style, indent=2)}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_message}],
        betas=["prompt-caching-2024-07-31"],
    )

    raw = response.content[0].text
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"Error parsing Sonnet plan JSON: {e}\nRaw output:\n{raw[:500]}")

    cache_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
    print(f"  [✓] Plan: {len(plan.get('slides', []))} slides | cache_read={cache_tokens}")
    return plan


# ---------------------------------------------------------------------------
# Photo selection algorithm (4 passes)
# ---------------------------------------------------------------------------
def select_photos_for_slides(slides: list[dict], photos: list[dict], photos_dir: Path | None) -> dict[int, Path | None]:
    """
    4-pass photo selection algorithm.
    Pass A: exact city match + highest quality
    Pass B: tag/subject overlap
    Pass C: any photo from same itinerary above threshold
    Pass D: best available (fallback)
    Returns: dict {slide_number: photo_path or None}
    """
    selected: dict[int, Path | None] = {}
    used_paths: set[str] = set()

    def find_photo(hint: str, city: str | None, exclude: set, min_score: float = 7.0) -> dict | None:
        hint_words = set((hint or "").lower().split())
        city_norm = (city or "").lower().replace(" ", "-")

        # Pass A: city match
        if city_norm:
            candidates = [p for p in photos
                          if city_norm in p.get("city", "").lower()
                          and p.get("quality_score", 0) >= min_score
                          and p["filename"] not in exclude]
            if candidates:
                return candidates[0]

        # Pass B: tag overlap with hint
        if hint_words:
            def tag_score(p):
                tags = set(t.lower() for t in p.get("tags", []) + p.get("subjects", []))
                return len(hint_words & tags)

            candidates = [p for p in photos
                          if p.get("quality_score", 0) >= min_score
                          and p["filename"] not in exclude
                          and tag_score(p) > 0]
            candidates.sort(key=lambda p: (-tag_score(p), -p.get("quality_score", 0)))
            if candidates:
                return candidates[0]

        # Pass C: any photo above threshold not yet used
        candidates = [p for p in photos
                      if p.get("quality_score", 0) >= min_score
                      and p["filename"] not in exclude]
        if candidates:
            return candidates[0]

        # Pass D: any photo above minimum
        candidates = [p for p in photos if p["filename"] not in exclude]
        return candidates[0] if candidates else None

    for slide in slides:
        num = slide.get("slide_number", 0)
        layout = slide.get("layout", "content")

        if layout == "cta":
            selected[num] = None
            continue

        min_score = 8.5 if layout == "hook" else 7.0
        hint = slide.get("photo_hint", "")
        city = None
        if hint:
            # Try to infer city from hint or tag
            city = slide.get("tag", "").lower()

        photo = find_photo(hint, city, used_paths, min_score)
        if photo:
            path = REPO_ROOT / photo.get("path", "")
            if not path.exists() and photos_dir:
                # Try photos_dir directly
                path = photos_dir / photo["filename"]
            selected[num] = path if path.exists() else None
            used_paths.add(photo["filename"])
            score = photo.get("quality_score", 0)
            flag = "" if score >= min_score else " ⚠ suboptimal"
            print(f"    Slide {num}: {photo['filename']} (score={score}){flag}")
        else:
            selected[num] = None
            print(f"    Slide {num}: no photo found")

    return selected


# ---------------------------------------------------------------------------
# Font loading helpers
# ---------------------------------------------------------------------------
def load_font(name_hint: str, size: int) -> ImageFont.FreeTypeFont:
    """Try to load a font from FONTS_DIR, fall back to default."""
    name_lower = name_hint.lower()

    candidates = []
    if FONTS_DIR.exists():
        for f in FONTS_DIR.glob("*.ttf"):
            fname = f.stem.lower()
            if "playfair" in fname and "playfair" in name_lower:
                candidates.append(f)
            elif "roboto" in fname and "roboto" in name_lower:
                candidates.append(f)
            elif "montserrat" in fname and "montserrat" in name_lower:
                candidates.append(f)

    # Bold preference
    if "bold" in name_lower:
        bold_candidates = [c for c in candidates if "bold" in c.stem.lower()]
        if bold_candidates:
            candidates = bold_candidates

    if candidates:
        try:
            return ImageFont.truetype(str(candidates[0]), size)
        except OSError:
            pass

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def parse_color(color_str: str) -> tuple[int, int, int, int]:
    """Parse hex or rgba(...) to RGBA tuple."""
    color_str = color_str.strip()
    if color_str.startswith("#"):
        r, g, b = hex_to_rgb(color_str)
        return (r, g, b, 255)
    if color_str.startswith("rgba"):
        vals = re.findall(r"[\d.]+", color_str)
        r, g, b = int(vals[0]), int(vals[1]), int(vals[2])
        a = int(float(vals[3]) * 255) if len(vals) > 3 else 255
        return (r, g, b, a)
    return (255, 255, 255, 255)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------
def apply_gradient_overlay(img: Image.Image, opacity: float = 0.35) -> Image.Image:
    """Apply dark-to-transparent gradient from bottom."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h = img.size[1]
    for y in range(h):
        alpha = int(opacity * 255 * (y / h))
        draw.line([(0, y), (img.size[0], y)], fill=(0, 0, 0, alpha))
    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result


def draw_text_wrapped(draw, text: str, font, x: int, y: int, max_width: int,
                      fill=(255, 255, 255, 255), line_height: float = 1.3) -> int:
    """Draw wrapped text. Returns y position after last line."""
    words = text.split()
    lines = []
    current = []

    for word in words:
        test = " ".join(current + [word])
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    font_size = font.size if hasattr(font, "size") else 30
    line_h = int(font_size * line_height)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h

    return y


def draw_pill_tag(draw, text: str, x: int, y: int, font, padding_h: int = 18, padding_v: int = 8):
    """Draw a white-bordered pill label."""
    bbox = font.getbbox(text.upper())
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    rx = padding_h
    ry = padding_v
    rect = [x, y, x + tw + rx * 2, y + th + ry * 2]
    draw.rounded_rectangle(rect, radius=20, outline=(255, 255, 255, 255), width=2)
    draw.text((x + rx, y + ry), text.upper(), font=font, fill=(255, 255, 255, 255))
    return rect


def draw_circle_counter(img: Image.Image, number: int, style: dict):
    """Draw slide counter circle in top-right corner."""
    diameter = style.get("diameter", 72)
    margin = style.get("margin", 28)
    bg_color = parse_color(style.get("background", "#FFFFFF"))
    text_color = parse_color(style.get("color", "#1a3a35"))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x1 = img.width - diameter - margin
    y1 = margin
    x2 = x1 + diameter
    y2 = y1 + diameter
    draw.ellipse([x1, y1, x2, y2], fill=bg_color)

    font = load_font("Roboto Bold", style.get("size", 28))
    text = str(number)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = x1 + (diameter - tw) // 2
    cy = y1 + (diameter - th) // 2
    draw.text((cx, cy), text, font=font, fill=text_color)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Slide renderers
# ---------------------------------------------------------------------------
def render_hook_slide(slide: dict, photo_path: Path | None, style: dict, handle: str) -> Image.Image:
    """Render the hook/first slide: full photo with overlay, tag top-left, big title bottom."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (26, 58, 53))

    if photo_path and photo_path.exists():
        photo = Image.open(photo_path).convert("RGB")
        photo = photo.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        photo_rgba = apply_gradient_overlay(photo, opacity=0.55)
        canvas = photo_rgba.convert("RGB")

    draw = ImageDraw.Draw(canvas)

    # Tag (top-left)
    tag_text = slide.get("tag", "")
    if tag_text:
        tag_font = load_font("Roboto Bold", style["tag"]["size"])
        draw_pill_tag(draw, tag_text, x=40, y=40, font=tag_font)

    # Title (bottom)
    title_font = load_font("Playfair Display Bold", style["hook_slide"]["title_size"])
    title_text = slide.get("title", "")
    if title_text:
        draw_text_wrapped(draw, title_text, title_font,
                          x=52, y=CANVAS_H - 280,
                          max_width=CANVAS_W - 104,
                          fill=(255, 255, 255, 255),
                          line_height=style["title"]["line_height"])

    # Handle (bottom-left)
    handle_font = load_font("Montserrat Regular", style["handle"]["size"])
    draw.text((52, CANVAS_H - style["handle"]["margin_bottom"] - 30), handle,
              font=handle_font, fill=parse_color(style["handle"]["color"]))

    return canvas


def render_content_slide(slide: dict, photo_path: Path | None, style: dict,
                         slide_number: int, handle: str) -> Image.Image:
    """Render a split layout content slide."""
    photo_h = int(CANVAS_H * 0.50)
    text_h  = CANVAS_H - photo_h

    # Photo zone
    if photo_path and photo_path.exists():
        photo = Image.open(photo_path).convert("RGB")
        photo = photo.resize((CANVAS_W, photo_h), Image.LANCZOS)
        photo_rgba = apply_gradient_overlay(photo, opacity=style["photo_zone"]["overlay_opacity"])
        photo_zone = photo_rgba.convert("RGB")
    else:
        photo_zone = Image.new("RGB", (CANVAS_W, photo_h), (40, 80, 60))

    # Text zone
    text_bg_color = hex_to_rgb(style["text_zone"]["background"])
    text_zone = Image.new("RGB", (CANVAS_W, text_h), text_bg_color)

    # Combine
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
    canvas.paste(photo_zone, (0, 0))
    canvas.paste(text_zone, (0, photo_h))

    draw = ImageDraw.Draw(canvas)
    pad_h = style["text_zone"]["padding_h"]
    pad_v = style["text_zone"]["padding_v"]
    text_y = photo_h + pad_v

    # Tag
    tag_text = slide.get("tag", "")
    if tag_text:
        tag_font = load_font("Roboto Bold", style["tag"]["size"])
        tag_rect = draw_pill_tag(draw, tag_text, x=pad_h, y=text_y, font=tag_font)
        text_y = tag_rect[3] + style["title"]["margin_top"]

    # Title
    title_text = slide.get("title", "")
    if title_text:
        title_font = load_font("Playfair Display Bold", style["title"]["size"])
        text_y = draw_text_wrapped(draw, title_text, title_font,
                                   x=pad_h, y=text_y,
                                   max_width=CANVAS_W - pad_h * 2,
                                   fill=parse_color(style["title"]["color"]),
                                   line_height=style["title"]["line_height"])

    # Separator
    sep = style["separator"]
    sep_y = text_y + 18
    draw.rectangle([pad_h, sep_y, pad_h + sep["width"], sep_y + sep["height"]],
                   fill=parse_color(sep["color"]))
    text_y = sep_y + sep["height"] + 18

    # Body bullets
    tip_box_data = slide.get("tip_box")
    body = slide.get("body", [])

    if tip_box_data and isinstance(tip_box_data, dict):
        # Tip box style
        tb_bg = parse_color("rgba(255,255,255,0.08)")
        tb_x, tb_y = pad_h, text_y
        tb_w = CANVAS_W - pad_h * 2
        tb_h = 120

        tip_canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        tip_draw = ImageDraw.Draw(tip_canvas)
        tip_draw.rectangle([tb_x, tb_y, tb_x + tb_w, tb_y + tb_h], fill=tb_bg)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), tip_canvas).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        # Left border
        draw.rectangle([tb_x, tb_y, tb_x + 4, tb_y + tb_h], fill=(255, 255, 255, 255))

        label_font = load_font("Roboto Bold", style["tip_box"]["label_size"])
        draw.text((tb_x + 20, tb_y + 12), tip_box_data.get("label", "PRO TIP"),
                  font=label_font, fill=(255, 255, 255, 255))

        body_font = load_font("Montserrat Regular", style["body"]["size"] - 4)
        draw_text_wrapped(draw, tip_box_data.get("text", ""),
                          body_font, tb_x + 20, tb_y + 44,
                          max_width=tb_w - 40,
                          fill=parse_color(style["body"]["color"]),
                          line_height=1.4)
    elif body:
        body_font = load_font("Montserrat Regular", style["body"]["size"])
        bullet_lh = int(style["body"]["size"] * style["body"]["line_height"])
        for bullet in body[:5]:
            line = f"• {bullet}" if not bullet.startswith("•") else bullet
            text_y = draw_text_wrapped(draw, line, body_font,
                                       x=pad_h, y=text_y,
                                       max_width=CANVAS_W - pad_h * 2,
                                       fill=parse_color(style["body"]["color"]),
                                       line_height=style["body"]["line_height"])
            text_y += 6

    # Handle (bottom-left)
    handle_font = load_font("Montserrat Regular", style["handle"]["size"])
    handle_y = CANVAS_H - style["handle"]["margin_bottom"] - style["handle"]["size"]
    draw.text((pad_h, handle_y), handle, font=handle_font,
              fill=parse_color(style["handle"]["color"]))

    # Swipe indicator (bottom-right)
    if slide.get("show_swipe", True):
        swipe_font = load_font("Roboto Bold", style["swipe_indicator"]["size"])
        swipe_text = style["swipe_indicator"]["text"]
        swipe_bbox = swipe_font.getbbox(swipe_text)
        swipe_w = swipe_bbox[2] - swipe_bbox[0]
        swipe_x = CANVAS_W - pad_h - swipe_w
        swipe_y = CANVAS_H - style["swipe_indicator"]["margin_bottom"] - style["swipe_indicator"]["size"]
        draw.text((swipe_x, swipe_y), swipe_text, font=swipe_font,
                  fill=parse_color(style["swipe_indicator"]["color"]))

    # Slide counter
    if slide.get("show_counter", True):
        canvas = draw_circle_counter(canvas, slide_number, style["slide_counter"])

    return canvas


def render_cta_slide(slide: dict, style: dict, handle: str) -> Image.Image:
    """Render the final CTA slide: dark bg, centered handle, no photo."""
    cta = style["cta_slide"]
    bg_color = parse_color(cta.get("background", "rgba(20,45,40,0.92)"))
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), bg_color[:3])
    draw = ImageDraw.Draw(canvas)

    # Large centered handle
    handle_font = load_font("Playfair Display Bold", cta.get("handle_size", 52))
    handle_text = handle
    bbox = handle_font.getbbox(handle_text)
    tw = bbox[2] - bbox[0]
    cx = (CANVAS_W - tw) // 2
    cy = CANVAS_H // 2 - 60
    draw.text((cx, cy), handle_text, font=handle_font, fill=parse_color(cta.get("handle_color", "#FFFFFF")))

    # Body lines
    body = slide.get("body", [])
    body_font = load_font("Montserrat Regular", 32)
    body_y = cy + 80
    for line in body:
        bbox = body_font.getbbox(line)
        tw = bbox[2] - bbox[0]
        bx = (CANVAS_W - tw) // 2
        draw.text((bx, body_y), line, font=body_font, fill=(255, 255, 255, 200))
        body_y += 50

    return canvas


# ---------------------------------------------------------------------------
# Resize helper for scoring
# ---------------------------------------------------------------------------
def resize_for_scoring(photo_path: Path, out_dir: Path, max_size: int = 512) -> Path:
    """Resize a photo to max 512px for API vision scoring. Returns path to resized file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scoring_{photo_path.stem}.jpg"
    if out_path.exists():
        return out_path
    img = Image.open(photo_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img.save(out_path, "JPEG", quality=70)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("\n=== Carousel Generator — @jeanseestheworld ===\n")

    # Load configs
    brand = json.loads(BRAND_JSON.read_text()) if BRAND_JSON.exists() else {}
    style = json.loads(STYLE_JSON.read_text()) if STYLE_JSON.exists() else {}
    planner_prompt = PLANNER_TXT.read_text() if PLANNER_TXT.exists() else ""

    # Collect sources
    print("[1/5] Collecting sources…")
    merged_notes = collect_sources(args)

    # Load photos
    print("\n[2/5] Loading photos…")
    photos = load_photos(args.photos)

    # Haiku: extract structured data
    print("\n[3/5] Extracting structured data (Haiku)…")
    client = anthropic.Anthropic()
    structured = extract_structured_data(client, merged_notes)
    print(f"  Destination: {structured.get('destination')}")
    print(f"  Cities: {', '.join(structured.get('cities', []))}")

    # Sonnet: generate plan
    print("\n[4/5] Generating carousel plan (Sonnet)…")
    plan = generate_carousel_plan(
        client, structured, merged_notes, photos,
        args.handle, brand, style, planner_prompt
    )

    slides = plan.get("slides", [])
    print(f"\n  Carousel: \"{plan.get('carousel_title', '')}\"")
    print(f"  Slides: {len(slides)}")
    print(f"\n  Caption preview:\n  {plan.get('caption', '')[:200]}…\n")

    # Save plan
    plan_path = args.output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"  Plan saved → {plan_path}")

    if args.dry_run:
        print("\n[dry-run] Skipping slide rendering.")
        return

    # Photo selection
    print("\n[5/5] Selecting photos & rendering slides…")
    photo_map = select_photos_for_slides(slides, photos, args.photos)

    # Render slides
    print()
    for slide in slides:
        num = slide.get("slide_number", 0)
        layout = slide.get("layout", "content")
        photo_path = photo_map.get(num)

        if layout == "hook":
            img = render_hook_slide(slide, photo_path, style, args.handle)
        elif layout == "cta":
            img = render_cta_slide(slide, style, args.handle)
        else:
            img = render_content_slide(slide, photo_path, style, num, args.handle)

        out_name = f"slide_{num:02d}.jpg"
        out_path = args.output / out_name
        img.convert("RGB").save(str(out_path), "JPEG", quality=style.get("canvas", {}).get("quality", 95))
        print(f"  → {out_name} ({layout})")

    # Save caption separately
    caption_path = args.output / "caption.txt"
    caption_path.write_text(plan.get("caption", "") + "\n\n" + " ".join(plan.get("hashtags", [])))
    print(f"\n  Caption → {caption_path}")

    # Summary
    print(f"\n=== Done: {len(slides)} slides in {args.output} ===")
    print(f"  Location tag: {plan.get('location_tag', 'n/a')}")
    print(f"  Hashtags: {' '.join(plan.get('hashtags', []))}")

    # Warnings for suboptimal photos
    warnings = []
    for slide in slides:
        num = slide.get("slide_number", 0)
        layout = slide.get("layout", "content")
        if layout == "cta":
            continue
        path = photo_map.get(num)
        if path is None:
            warnings.append(f"Slide {num}: no photo — rendered with solid background")
        elif not path.exists():
            warnings.append(f"Slide {num}: photo path not found — {path}")

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")


if __name__ == "__main__":
    main()
