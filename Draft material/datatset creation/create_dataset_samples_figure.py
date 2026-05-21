"""
Create Figure 2: Dataset samples collage (HR ground truth + 4 LR degraded variants).

Layout:
  [HR image (large)] --PhysDeg--> [2x2 grid of LR variants]

Usage:
  python create_dataset_samples_figure.py
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).parent / "dataset samples"
OUTPUT_DIR = Path(__file__).parent / "author-kit-CVPR2026-v1-latex-" / "figures"

HR_FILE = "natural_landscapes_0003-hr.png"
LR_FILES = [
    ("natural_landscapes_0003_v00.png", "LR Variant 1", "(Mild degradation)"),
    ("natural_landscapes_0003_v01.png", "LR Variant 2", "(Color shift + blur)"),
    ("natural_landscapes_0003_v02.png", "LR Variant 3", "(Noise + compression)"),
    ("natural_landscapes_0003_v03.png", "LR Variant 4", "(Noise + low-light)"),
]

BG = (255, 255, 255)
BORDER_W = 3
HR_BORDER_CLR = (46, 125, 50)
LR_BORDER_CLR = (198, 40, 40)
PAD = 14
LABEL_H = 16
SUBLABEL_H = 14
ARROW_W = 60


def get_font(sz):
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def bordered(img, w, c):
    iw, ih = img.size
    out = Image.new("RGB", (iw + 2 * w, ih + 2 * w), c)
    out.paste(img, (w, w))
    return out


def create_figure():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f_label = get_font(13)
    f_sub = get_font(11)
    f_title = get_font(11)
    f_arrow = get_font(12)

    hr_img = Image.open(SAMPLES_DIR / HR_FILE).convert("RGB")
    lr_imgs = []
    for fname, label, sub in LR_FILES:
        lr_imgs.append((Image.open(SAMPLES_DIR / fname).convert("RGB"), label, sub))

    hr_w, hr_h = hr_img.size
    target_hr_h = 400
    hr_scale = target_hr_h / hr_h
    hr_disp_w = int(hr_w * hr_scale)
    hr_disp_h = target_hr_h
    hr_resized = hr_img.resize((hr_disp_w, hr_disp_h), Image.LANCZOS)
    hr_bordered = bordered(hr_resized, BORDER_W, HR_BORDER_CLR)

    lr_cell_size = (hr_disp_h - PAD) // 2
    lr_cells = []
    for img, label, sub in lr_imgs:
        resized = img.resize((lr_cell_size, lr_cell_size), Image.LANCZOS)
        lr_cells.append((bordered(resized, BORDER_W, LR_BORDER_CLR), label, sub))

    lr_cell_w = lr_cells[0][0].size[0]
    lr_cell_h = lr_cells[0][0].size[1]

    title_area = 20
    hr_label_area = LABEL_H + 4
    lr_label_area = LABEL_H + SUBLABEL_H + 4

    grid_w = lr_cell_w * 2 + PAD
    grid_h = (lr_cell_h + lr_label_area) * 2 + PAD

    canvas_w = PAD + hr_bordered.size[0] + ARROW_W + grid_w + PAD
    canvas_h = PAD + title_area + max(hr_bordered.size[1] + hr_label_area, grid_h) + PAD

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    title_text = "Dataset Sample: One HR Source \u2192 Multiple Degraded LR Variants via PhysDeg Pipeline"
    draw.text((PAD, PAD), title_text, fill=(60, 60, 60), font=f_title)

    hr_x = PAD
    hr_y = PAD + title_area
    canvas.paste(hr_bordered, (hr_x, hr_y))

    hr_label = "HR Ground Truth"
    hr_res_text = f"({hr_w} \u00d7 {hr_h} px)"
    bb1 = draw.textbbox((0, 0), hr_label, font=f_label)
    bb2 = draw.textbbox((0, 0), hr_res_text, font=f_sub)
    hr_center_x = hr_x + hr_bordered.size[0] // 2
    draw.text((hr_center_x - (bb1[2] - bb1[0]) // 2, hr_y + hr_bordered.size[1] + 4),
              hr_label, fill=(20, 20, 20), font=f_label)
    draw.text((hr_center_x - (bb2[2] - bb2[0]) // 2, hr_y + hr_bordered.size[1] + 4 + LABEL_H),
              hr_res_text, fill=(100, 100, 100), font=f_sub)

    arrow_x = hr_x + hr_bordered.size[0] + 4
    arrow_y = hr_y + hr_bordered.size[1] // 2
    arrow_end_x = arrow_x + ARROW_W - 8
    draw.line([(arrow_x, arrow_y), (arrow_end_x, arrow_y)], fill=(100, 100, 100), width=2)
    draw.polygon([(arrow_end_x, arrow_y - 6), (arrow_end_x + 10, arrow_y),
                  (arrow_end_x, arrow_y + 6)], fill=(100, 100, 100))
    arrow_label = "PhysDeg"
    ab = draw.textbbox((0, 0), arrow_label, font=f_arrow)
    draw.text((arrow_x + (ARROW_W - (ab[2] - ab[0])) // 2, arrow_y - 20),
              arrow_label, fill=(46, 125, 50), font=f_arrow)

    grid_x = arrow_x + ARROW_W
    grid_y = hr_y + (hr_bordered.size[1] - grid_h) // 2

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for idx, (col, row) in enumerate(positions):
        cell_img, label, sub = lr_cells[idx]
        cx = grid_x + col * (lr_cell_w + PAD)
        cy = grid_y + row * (lr_cell_h + lr_label_area + PAD)
        canvas.paste(cell_img, (cx, cy))

        bb = draw.textbbox((0, 0), label, font=f_label)
        draw.text((cx + (lr_cell_w - (bb[2] - bb[0])) // 2, cy + lr_cell_h + 2),
                  label, fill=(20, 20, 20), font=f_label)
        bb2 = draw.textbbox((0, 0), sub, font=f_sub)
        draw.text((cx + (lr_cell_w - (bb2[2] - bb2[0])) // 2, cy + lr_cell_h + 2 + LABEL_H),
                  sub, fill=(120, 120, 120), font=f_sub)

    out = OUTPUT_DIR / "fig_dataset_samples.png"
    canvas.save(out, "PNG", dpi=(300, 300))
    print(f"Saved: {out}  ({canvas.size[0]}x{canvas.size[1]} px)")


if __name__ == "__main__":
    create_figure()
