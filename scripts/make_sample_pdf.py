"""Generate a sample PDF whose content is pure pixels- no text layer at all."""


from networkx.algorithms.centrality import current_flow_betweenness_subset
from asyncio import base_events
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "pdfs" / "sales_report.pdf"
W, H = 1240, 1754 # ~150 DPI A4

def _font(size: int):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def make_chart_page() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 70), "Acme India- Quarterly Revenue (Rs Crore)", font=_font(46), fill="black")
    quarters = [("Q1", 120), ("Q2", 150), ("Q3", 210), ("Q4", 180)]
    base_y, max_h, bar_w, gap, left = 1300, 900, 180, 110, 20
    d.line((left - 40, base_y, W - 120, base_y), fill="black", width=4)
    d.line((left - 40, base_y, left - 40, base_y - max_h - 40), fill="black", width=4)
    for i, (label, value) in enumerate(quarters):
        x0 = left + i * (bar_w + gap)
        bar_h = int(max_h * value / 250)
        d.rectangle((x0, base_y - bar_h, x0 + bar_w, base_y), fill="#4f46e5"    )
        d.text((x0 + 55, base_y - bar_h - 55), str(value), font=_font(40), fill="black")
        d.text((x0 + 55, base_y + 20), label, font=_font(40), fill="black")
    return img

def make_table_page() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 70), "Regional Sales- FY2025 (Rs Crore)", font=_font(46), fill="black")
    rows = [("Region", "Revenue", "Growth %"), ("North", "240", "12"),
            ("South", "310", "18"), ("East", "150", "7"), ("West", "260", "15")]
    x0, y0, col_w, row_h = 120, 220, 320, 110
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = x0 + c * col_w, y0 + r * row_h
            d.rectangle((x, y, x + col_w, y + row_h), outline="black", width=3)
            d.text((x + 30, y + 30), cell, font=_font(38), fill="black")
    return img

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chart = make_chart_page().convert("P")
    table = make_table_page().convert("P")
    chart.save(OUT, save_all=True, append_images=[table])
    print(f"Wrote{OUT}")

if __name__ == "__main__":
    main()