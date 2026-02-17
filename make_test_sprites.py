"""make_test_sprites.py — one-time helper to generate simple labeled test sprites."""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "test_sprites"
OUT.mkdir(exist_ok=True)
W, H = 128, 128

def sprite(name, draws):
    img = Image.new("RGBA", (W, H), "white")
    d = ImageDraw.Draw(img)
    for fn, coords, kw in draws:
        getattr(d, fn)(coords, **kw)
    img.save(OUT / name)
    print(f"  {name}")

print("Creating test sprites in", OUT)

# sword_1 — silver blade
sprite("sword_1.png", [
    ("rectangle", [8, 55, 120, 73],  {"fill": (192, 192, 192)}),
    ("rectangle", [108, 48, 120, 80], {"fill": (255, 215, 0)}),
])

# sword_2 — brown hilt
sprite("sword_2.png", [
    ("rectangle", [54, 72, 74, 120], {"fill": (101, 67, 33)}),
    ("ellipse",   [46, 112, 82, 128], {"fill": (80, 50, 20)}),
])

# closet_1 — doors closed
sprite("closet_1.png", [
    ("rectangle", [10, 10, 118, 118], {"fill": (101, 67, 33)}),
    ("ellipse",   [54, 58, 62, 66],   {"fill": (255, 215, 0)}),
    ("ellipse",   [66, 58, 74, 66],   {"fill": (255, 215, 0)}),
])

# closet_2 — doors open
sprite("closet_2.png", [
    ("rectangle", [10, 10, 118, 118], {"fill": (180, 130, 80)}),
    ("rectangle", [10, 10, 35, 118],  {"fill": (101, 67, 33)}),
    ("rectangle", [93, 10, 118, 118], {"fill": (101, 67, 33)}),
])

# tree — green canopy + brown trunk
sprite("tree.png", [
    ("rectangle", [54, 80, 74, 118], {"fill": (101, 67, 33)}),
    ("ellipse",   [20, 15, 108, 90], {"fill": (34, 139, 34)}),
])

# chest — gold box
sprite("chest.png", [
    ("rectangle", [10, 45, 118, 118], {"fill": (218, 165, 32)}),
    ("rectangle", [10, 45, 118, 65],  {"fill": (139, 100, 20)}),
    ("ellipse",   [56, 55, 72, 71],   {"fill": (255, 215, 0)}),
])

# coin — yellow circle
sprite("coin.png", [
    ("ellipse", [20, 20, 108, 108], {"fill": (255, 215, 0)}),
    ("ellipse", [30, 30, 98, 98],   {"fill": (255, 235, 50)}),
])

# shield — blue oval
sprite("shield.png", [
    ("ellipse", [15, 10, 113, 118], {"fill": (30, 70, 180)}),
    ("ellipse", [50, 45, 78, 83],   {"fill": (200, 200, 255)}),
])

print(f"\nDone — {len(list(OUT.glob('*.png')))} sprites in {OUT}")
