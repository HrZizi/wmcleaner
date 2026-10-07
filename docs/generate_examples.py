"""Generate the synthetic documentation images used by the public README.

The images are produced from artwork drawn in this script plus wmcleaner's own
small detector template. No third-party comic/manhua pages are involved, so the
generated files are safe to publish.

Run from the repository root:

    python docs/generate_examples.py
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wmcleaner.cleaning import CleanSettings, Cleaner
from wmcleaner.detector import combine_masks, detect_watermarks

OUT = Path(__file__).resolve().parent / "images"

WIDTH, HEIGHT = 850, 1100


def _synthetic_artwork() -> Image.Image:
    """A simple, original comic-like page drawn entirely with vector shapes."""
    page = Image.new("RGB", (WIDTH, HEIGHT), (247, 247, 244))
    draw = ImageDraw.Draw(page)

    # Comic pages have plain white margins; keeping this area flat also gives
    # the detector a clean, unambiguous surface to register against.

    # Two comic panels with borders. The top strip is deliberately left as
    # plain paper: that is where the watermark is composited, so detection and
    # registration run against a realistic but unambiguous background.
    panel = (60, 175, WIDTH - 60, 600)
    draw.rectangle(panel, fill=(232, 238, 246), outline=(25, 25, 30), width=5)
    lower = (60, 640, WIDTH - 60, 1040)
    draw.rectangle(lower, fill=(244, 234, 226), outline=(25, 25, 30), width=5)

    # Character: head, hair, eyes, and a speech bubble.
    cx, cy = 300, 400
    draw.ellipse((cx - 120, cy - 130, cx + 120, cy + 110), fill=(250, 224, 198), outline=(20, 20, 25), width=5)
    draw.chord((cx - 140, cy - 160, cx + 140, cy + 60), 180, 360, fill=(58, 44, 76), outline=(20, 20, 25), width=5)
    draw.ellipse((cx - 65, cy - 20, cx - 35, cy + 20), fill=(255, 255, 255), outline=(20, 20, 25), width=3)
    draw.ellipse((cx + 35, cy - 20, cx + 65, cy + 20), fill=(255, 255, 255), outline=(20, 20, 25), width=3)
    draw.ellipse((cx - 55, cy - 8, cx - 45, cy + 6), fill=(30, 30, 40))
    draw.ellipse((cx + 45, cy - 8, cx + 55, cy + 6), fill=(30, 30, 40))
    draw.arc((cx - 40, cy + 20, cx + 40, cy + 80), 20, 160, fill=(30, 30, 40), width=4)

    bubble = (560, 250, 790, 450)
    draw.ellipse(bubble, fill=(255, 255, 255), outline=(25, 25, 30), width=5)
    draw.polygon([(610, 430), (600, 510), (680, 440)], fill=(255, 255, 255), outline=(25, 25, 30))
    for row in range(3):
        y = 305 + row * 45
        draw.line([(600, y), (750 - row * 20, y)], fill=(25, 25, 30), width=6)

    # Lower panel: simple landscape and caption lines.
    draw.ellipse((120, 740, 320, 940), fill=(196, 214, 232), outline=(25, 25, 30), width=4)
    draw.rectangle((420, 740, 760, 940), fill=(214, 226, 210), outline=(25, 25, 30), width=4)
    for row in range(3):
        y = 975 + row * 22
        draw.line([(110, y), (740 - row * 90, y)], fill=(25, 25, 30), width=5)

    return page


def _stamp_logo(page: Image.Image, *, x: int, edge: str) -> Image.Image:
    """Composite wmcleaner's own bundled modern-logo template onto the page."""
    page = page.copy()
    anchor = np.array(
        Image.open(
            resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right_anchor.png")
        ).convert("L"),
        dtype=np.uint8,
    )
    ys, xs = np.where(anchor > 0)
    ay0, ay1 = int(ys.min()), int(ys.max() + 1)
    ax0, ax1 = int(xs.min()), int(xs.max() + 1)

    with resources.as_file(
        resources.files("wmcleaner").joinpath("templates", "baozimh_bottom_right.png")
    ) as asset:
        source = np.array(Image.open(asset).convert("RGB"), dtype=np.uint8)
    icon = source[ay0:ay1, ax0:ax1]

    array = np.array(page, dtype=np.uint8)
    y = 34 if edge == "top" else array.shape[0] - 34 - icon.shape[0]
    array[y:y + icon.shape[0], x:x + icon.shape[1]] = icon
    return Image.fromarray(array, "RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    artwork = _synthetic_artwork()
    marked = _stamp_logo(artwork, x=520, edge="top")

    detections = detect_watermarks(marked)
    mask = combine_masks(detections, (marked.height, marked.width))

    cleaner = Cleaner(CleanSettings(backend="opencv"))
    result = cleaner.clean(marked)

    marked.save(OUT / "synthetic_before.png")
    Image.fromarray(mask).save(OUT / "synthetic_mask.png")
    result.image.save(OUT / "synthetic_after.png")

    labels = ", ".join(f"{d.name}:{d.confidence:.2f}" for d in detections) or "none"
    print(f"detections: {labels}")
    print(f"cleaned: {result.cleaned} backend: {result.backend}")
    print(f"wrote 3 images to {OUT}")


if __name__ == "__main__":
    main()
