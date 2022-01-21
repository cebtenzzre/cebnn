#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import sys
from math import sqrt

from PIL import Image

MAX_SAMPLE_DIM = 32 * 1024
OUTPUT_SIZE = 300
CROP_FACTOR = .25  # Limits crop of longer dimension
MAX_ASPECT = 2  # Limits aspect ratio


def load_and_scale(image_path: str) -> Image.Image:
    try:
        with Image.open(image_path) as img:
            if getattr(img, 'is_animated', False):
                img.seek(img.n_frames - 1)  # Last frame is a safe choice  # pytype: disable=attribute-error
            if img is not None and (img.width > MAX_SAMPLE_DIM or img.height > MAX_SAMPLE_DIM):
                print(
                    'Image dimensions too large! {}x{} > {msd}x{msd}, file: {}'.format(
                        img.width, img.height, image_path, msd=MAX_SAMPLE_DIM),
                    file=sys.stderr,
                )
                img = None
            else:
                img = img.convert('RGB')
    except (OSError, SyntaxError, Image.DecompressionBombError, struct.error) as e:
        print('Caught error loading {}: {}'.format(image_path, e), file=sys.stderr)
        img = None

    if img is None:
        print('Generating blank sample image due to unusable file', file=sys.stderr)
        return Image.new('RGB', (OUTPUT_SIZE, OUTPUT_SIZE))  # Black replacement image

    # Crop down longer dimension
    def lim(l: int, s: int) -> int:
        return max(round(l - CROP_FACTOR * sqrt(l * s)), s)
    w, h = img.size
    w_crop, h_crop = (lim(w, h), h) if w >= h else (w, lim(h, w))
    h_inset, v_inset = round((w - w_crop) / 2), round((h - h_crop) / 2)
    img = img.crop((h_inset, v_inset, w - h_inset, h - v_inset))

    # Nearest-neighbor aspect adjust
    def limar(l: int, s: int) -> int:
        return round(min(l, MAX_ASPECT * s))
    w, h = img.size
    w_sc, h_sc = (limar(w, h), h) if w >= h else (w, limar(h, w))
    img = img.resize((w_sc, h_sc), Image.NEAREST)

    # Bilinear proportional scale
    (w, h), l = img.size, max(*img.size)
    w_sc, h_sc = round(OUTPUT_SIZE * w / l), round(OUTPUT_SIZE * h / l)
    img = img.resize((w_sc, h_sc), Image.BILINEAR)

    # Paste onto background
    bg = Image.new(img.mode, (OUTPUT_SIZE, OUTPUT_SIZE), 'black')  # pytype: disable=wrong-arg-types
    (w, h), l = img.size, max(*img.size)
    xoff, yoff = round((l - w) / 2), round((l - h) / 2)
    bg.paste(img, (xoff, yoff))
    return bg


if __name__ == '__main__':
    image_path, save_path = sys.argv[1:]
    img = load_and_scale(image_path)
    img.save(save_path, compress_level=1, exif=None)
