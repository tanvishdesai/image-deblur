"""
Create Figure 3: Qualitative comparison grid for the research paper.

Layout:
  Panel A (2 rows): Non-blind failures (DSCF-SR, SeemoRe-B)
  Panel B (1 row):  Blind SR failure (Real-ESRGAN)

Each source worst-sample image: [SR output (left) | HR GT (right)]

Usage:
  python create_comparison_figure.py
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RESULTS_BASE = Path(__file__).parent / "results of testing sota" / "results"
OUTPUT_DIR = Path(__file__).parent / "author-kit-CVPR2026-v1-latex-" / "figures"

MODEL_DIRS = {
    "DSCF-SR": RESULTS_BASE / "DSCF-SR" / "worst_samples",
    "SeemoRe-B": RESULTS_BASE / "SeemoRe_B_X4" / "worst_samples",
    "Real-ESRGAN": RESULTS_BASE / "RealESRGAN_x4" / "worst_samples",
}

# ============================================================================
# SCENE DEFINITIONS
# ============================================================================

PANELS = [
    {
        "title": "Non-blind SR Models (Bicubic-trained)",
        "col_labels": ["Full Image", "DSCF-SR", "SeemoRe-B", "HR Ground Truth"],
        "scenes": [
            {
                "tag": "(a)",
                "label": "Portrait — Illumination degradation",
                "files": {
                    "DSCF-SR": "worst_01_psnr_8.78.png",
                    "SeemoRe-B": "worst_01_psnr_8.79.png",
                },
                "psnrs": {"DSCF-SR": "8.78", "SeemoRe-B": "8.79"},
                "crop_frac": (0.2, 0.15, 0.8, 0.7),
            },
            {
                "tag": "(b)",
                "label": "Macro — Texture ringing artifacts",
                "files": {
                    "DSCF-SR": "worst_08_psnr_9.86.png",
                    "SeemoRe-B": "worst_08_psnr_9.86.png",
                },
                "psnrs": {"DSCF-SR": "9.86", "SeemoRe-B": "9.86"},
                "crop_frac": (0.05, 0.1, 0.55, 0.85),
            },
        ],
    },
    {
        "title": "Blind SR Model (Degradation-trained)",
        "col_labels": ["Full Image", "Real-ESRGAN", "HR Ground Truth"],
        "scenes": [
            {
                "tag": "(c)",
                "label": "Macro — Color/exposure shift",
                "files": {"Real-ESRGAN": "worst_04_psnr_9.17.png"},
                "psnrs": {"Real-ESRGAN": "9.17"},
                "crop_frac": (0.1, 0.1, 0.7, 0.9),
            },
            {
                "tag": "(d)",
                "label": "Landscape — Tone distortion",
                "files": {"Real-ESRGAN": "worst_10_psnr_10.27.png"},
                "psnrs": {"Real-ESRGAN": "10.27"},
                "crop_frac": (0.15, 0.2, 0.7, 0.8),
            },
        ],
    },
]

# Visual settings
CROP_SIZE = 240
THUMB_H = 240
BORDER = 2
PAD = 10
BG = (255, 255, 255)
CROP_BOX_CLR = (220, 40, 40)
GT_BORDER = (220, 40, 40)
SR_BORDER = (180, 180, 180)
HEADER_H = 26
SCENE_H = 20
PSNR_H = 18
PANEL_SEP = 16


def split_lr_hr(img):
    w, h = img.size
    return img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))


def frac_crop(img, f):
    w, h = img.size
    return img.crop((int(f[0]*w), int(f[1]*h), int(f[2]*w), int(f[3]*h)))


def sq(img, s):
    return img.resize((s, s), Image.LANCZOS)


def thumb_with_box(hr, frac, h):
    w0, h0 = hr.size
    tw = int(h * w0 / h0)
    t = hr.resize((tw, h), Image.LANCZOS)
    d = ImageDraw.Draw(t)
    b = (int(frac[0]*tw), int(frac[1]*h), int(frac[2]*tw), int(frac[3]*h))
    for i in range(3):
        d.rectangle([b[0]-i, b[1]-i, b[2]+i, b[3]+i], outline=CROP_BOX_CLR)
    return t


def bordered(img, w, c):
    bw, bh = img.size
    out = Image.new("RGB", (bw+2*w, bh+2*w), c)
    out.paste(img, (w, w))
    return out


def get_font(sz):
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
              "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def create_figure():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f_hdr = get_font(16)
    f_scn = get_font(12)
    f_psnr = get_font(11)
    f_panel = get_font(14)

    # Pre-build all rows
    built_panels = []

    for panel in PANELS:
        panel_rows = []
        model_names = [l for l in panel["col_labels"] if l not in ("Full Image", "HR Ground Truth")]

        for scene in panel["scenes"]:
            srs = {}
            hr_ref = None
            for mn in model_names:
                fname = scene["files"].get(mn)
                if not fname:
                    continue
                fp = MODEL_DIRS[mn] / fname
                if not fp.exists():
                    print(f"WARN: {fp} missing")
                    continue
                comp = Image.open(fp).convert("RGB")
                sr, hr = split_lr_hr(comp)
                srs[mn] = sr
                if hr_ref is None:
                    hr_ref = hr

            if hr_ref is None:
                continue

            cf = scene["crop_frac"]
            patches = []

            # Thumbnail
            tb = thumb_with_box(hr_ref, cf, THUMB_H)
            tw, th = tb.size
            if th < CROP_SIZE:
                p = Image.new("RGB", (tw, CROP_SIZE), BG)
                p.paste(tb, (0, (CROP_SIZE - th) // 2))
                tb = p
            patches.append(("Full Image", bordered(tb, BORDER, SR_BORDER), ""))

            # SR crops
            for mn in model_names:
                if mn in srs:
                    c = frac_crop(srs[mn], cf)
                    c = sq(c, CROP_SIZE)
                    psnr_label = scene["psnrs"].get(mn, "")
                    patches.append((mn, bordered(c, BORDER, SR_BORDER),
                                    f"PSNR: {psnr_label} dB" if psnr_label else ""))
                else:
                    ph = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (200, 200, 200))
                    patches.append((mn, bordered(ph, BORDER, SR_BORDER), "N/A"))

            # HR GT crop
            hc = frac_crop(hr_ref, cf)
            hc = sq(hc, CROP_SIZE)
            patches.append(("HR Ground Truth", bordered(hc, BORDER, GT_BORDER), ""))

            panel_rows.append((scene["tag"], scene["label"], patches))

        built_panels.append((panel["title"], panel["col_labels"], panel_rows))

    # ---- COMPUTE CANVAS SIZE ----
    max_w = 0
    total_h = PAD

    for pidx, (ptitle, col_labels, rows) in enumerate(built_panels):
        if pidx > 0:
            total_h += PANEL_SEP + 4  # separator + line
        total_h += HEADER_H + 6  # col labels

        for tag, label, patches in rows:
            total_h += SCENE_H
            rh = max(p.size[1] for _, p, _ in patches) + PSNR_H
            rw = sum(p.size[0] for _, p, _ in patches) + PAD * (len(patches) - 1)
            max_w = max(max_w, rw)
            total_h += rh + PAD

    canvas_w = max_w + 2 * PAD
    canvas_h = total_h + PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    y = PAD

    for pidx, (ptitle, col_labels, rows) in enumerate(built_panels):
        if pidx > 0:
            y += PANEL_SEP // 2
            draw.line([(PAD, y), (canvas_w - PAD, y)], fill=(160, 160, 160), width=1)
            y += PANEL_SEP // 2 + 4

        # Panel title (bold-ish)
        # draw.text((PAD, y), ptitle, fill=(60, 60, 60), font=f_panel)
        # y += 20  # skip panel title to save space

        # Column headers (only once per panel)
        if rows:
            _, _, first_patches = rows[0]
            x = PAD
            for i, (col_name, patch, _) in enumerate(first_patches):
                pw = patch.size[0]
                lbl = col_labels[i] if i < len(col_labels) else col_name
                bb = draw.textbbox((0, 0), lbl, font=f_hdr)
                tw = bb[2] - bb[0]
                draw.text((x + (pw - tw) // 2, y), lbl, fill=(20, 20, 20), font=f_hdr)
                x += pw + PAD
            y += HEADER_H

        # Rows
        for tag, label, patches in rows:
            # Scene label
            draw.text((PAD, y), f"{tag} {label}", fill=(80, 80, 80), font=f_scn)
            y += SCENE_H

            # Place patches
            rh = max(p.size[1] for _, p, _ in patches)
            x = PAD
            for col_name, patch, psnr_lbl in patches:
                pw, ph = patch.size
                yo = (rh - ph) // 2
                canvas.paste(patch, (x, y + yo))

                # PSNR label below crop
                if psnr_lbl:
                    bb = draw.textbbox((0, 0), psnr_lbl, font=f_psnr)
                    tw = bb[2] - bb[0]
                    draw.text((x + (pw - tw) // 2, y + rh + 2),
                              psnr_lbl, fill=(150, 50, 50), font=f_psnr)
                x += pw + PAD

            y += rh + PSNR_H + PAD

    # Trim trailing whitespace without exceeding canvas bounds
    from PIL import ImageChops
    bg_img = Image.new("RGB", canvas.size, BG)
    diff = ImageChops.difference(canvas, bg_img)
    bbox = diff.getbbox()
    if bbox:
        canvas = canvas.crop((0, 0,
                              min(bbox[2] + PAD, canvas.size[0]),
                              min(bbox[3] + PAD, canvas.size[1])))

    out = OUTPUT_DIR / "fig_qualitative_comparison.png"
    canvas.save(out, "PNG", dpi=(300, 300))
    print(f"Saved: {out}  ({canvas.size[0]}x{canvas.size[1]} px)")


if __name__ == "__main__":
    create_figure()
