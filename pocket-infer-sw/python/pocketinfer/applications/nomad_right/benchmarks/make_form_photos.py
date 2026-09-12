#!/usr/bin/env python3
"""
Turns the rendered form PDFs (~/benchmark_data/forms/render/*-1.png, 150 dpi) into
what the kiosk webcam would deliver: 1280x960-ish frames with perspective, tilt,
blur, uneven brightness, JPEG noise; partial views (title only); low resolution;
plus negatives (a desk scene, a form table, a blank sheet, a text page without a
title). Writes ~/benchmark_data/forms/photos/*.jpg and manifest.json
(expected form_id or null per image).
"""
import glob, json, os, random
import cv2, numpy as np

SRC = os.path.expanduser("~/benchmark_data/forms/render")
DST = os.path.expanduser("~/benchmark_data/forms/photos")
EXPECT = {"pmsby_en": "FORM_PMSBY_CONSENT", "pmsby_hi": "FORM_PMSBY_CONSENT", "pmsby_ta": "FORM_PMSBY_CONSENT",
          "pmjjby_en": "FORM_PMJJBY_CONSENT", "pmjjby_hi": "FORM_PMJJBY_CONSENT", "pmjjby_ta": "FORM_PMJJBY_CONSENT", "pmjjby_enrol": "FORM_PMJJBY_CONSENT",
          "apy_en": "FORM_APY_REGISTRATION", "apy_hi": "FORM_APY_REGISTRATION", "apy_ta": "FORM_APY_REGISTRATION",
          "pmuy_en": "FORM_PMUY_KYC", "pmuy_kyc": "FORM_PMUY_KYC", "svanidhi": "FORM_PMSVANIDHI_LAF", "pmsvanidhi_lor_laf": "FORM_PMSVANIDHI_LAF",
          "pmkisan": "FORM_PMKISAN_SELF_DECLARATION", "mgnrega_registration": "FORM_MGNREGA_REGISTRATION", "ssa1_centralbank": "FORM_SSA1_ACCOUNT_OPENING"}
LANG_OF = {"_hi": "hi", "_ta": "ta"}
rng = random.Random(7)


def warp(img, deg, persp):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(200, 200, 200))
    d = int(persp * w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d // 2], [w - d // 3, 0], [w, h - d // 2], [d // 2, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(200, 200, 200))


def camera_like(img, blur=1.0, bright=1.0, contrast=1.0, noise=4, vignette=0.25, q=70, width=1280):
    h, w = img.shape[:2]
    s = width / w
    img = cv2.resize(img, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=(bright - 1.0) * 120)
    hh, ww = img.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    v = 1.0 - vignette * (((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2) / 2
    img = np.clip(img.astype(np.float32) * v[..., None], 0, 255).astype(np.uint8)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = np.clip(img.astype(np.int16) + rng.randint(-noise, noise) + np.random.default_rng(1).integers(-noise, noise, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    os.makedirs(DST, exist_ok=True)
    manifest = []
    for png in sorted(glob.glob(f"{SRC}/*-1.png")):
        name = os.path.basename(png)[:-6]
        if name not in EXPECT or name.endswith("300"):
            continue
        img = cv2.imread(png)
        lang = next((v for k, v in LANG_OF.items() if k in name), "en")
        variants = {
            "clean": camera_like(img, blur=0.0, noise=0, vignette=0.0, q=90),
            "photo1": camera_like(warp(img, 2.5, 0.03), blur=0.9, bright=0.95, contrast=0.95),
            "photo2": camera_like(warp(img, -4.0, 0.05), blur=1.4, bright=0.85, contrast=0.85, noise=6),
            "dark": camera_like(warp(img, 1.0, 0.02), blur=1.0, bright=0.65, contrast=0.8, vignette=0.45),
            "partial": camera_like(img[: int(img.shape[0] * 0.55)], blur=0.8),
            "lowres": camera_like(warp(img, 1.5, 0.02), blur=1.2, width=800),
        }
        for v, out in variants.items():
            fn = f"{name}__{v}.jpg"; cv2.imwrite(f"{DST}/{fn}", out)
            manifest.append({"file": fn, "expected": EXPECT[name], "lang": lang, "variant": v})
    # negatives
    vis = os.path.expanduser("~/benchmark_data/vision")
    for src, label in (("camera_frame.jpg", "desk scene"), ("form_table.jpg", "form table"), ("ration_card.jpg", "ration card"), ("eshram_card.jpg", "eshram card")):
        p = f"{vis}/{src}"
        if os.path.exists(p):
            fn = f"neg__{src}"; cv2.imwrite(f"{DST}/{fn}", camera_like(cv2.imread(p), blur=0.5))
            manifest.append({"file": fn, "expected": None, "lang": "en", "variant": label})
    blank = np.full((960, 1280, 3), 235, np.uint8)
    cv2.imwrite(f"{DST}/neg__blank.jpg", camera_like(blank, blur=0.5)); manifest.append({"file": "neg__blank.jpg", "expected": None, "lang": "en", "variant": "blank sheet"})
    # a text page without a title: page 2 of the PMSBY form
    src2 = os.path.expanduser("~/Retro/NOMADRIGHT/pocket-infer-sw/nomadright/forms/sources/pmsby_en.pdf")
    os.system(f"pdftoppm -r 150 -png -f 2 -l 2 {src2} {SRC}/pmsby_p2 2>/dev/null")
    p2 = glob.glob(f"{SRC}/pmsby_p2-*.png")
    if p2:
        cv2.imwrite(f"{DST}/neg__pmsby_page2.jpg", camera_like(cv2.imread(p2[0]), blur=0.8)); manifest.append({"file": "neg__pmsby_page2.jpg", "expected": None, "lang": "en", "variant": "inner page, no title"})
    json.dump(manifest, open(f"{DST}/manifest.json", "w"), indent=1)
    print(f"{len(manifest)} images -> {DST}")


if __name__ == "__main__":
    main()
