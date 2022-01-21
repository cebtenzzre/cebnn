#!/usr/bin/env bash
set -euo pipefail

# Prints a normalized extension for a valid image (according to 'file', 'identify', and 'feh').

(( $# == 1 )) || exit 1
m_fulltype=$(file -Lb --mime-type -- "$1")
m_type=${m_fulltype%/*}
m_subtype=${m_fulltype##*/}

[[ $m_type == 'image' ]] || exit 0
case $m_subtype in
            x-ms-bmp) newext=bmp  ;;
               x-exr) newext=exr  ;;
                jpeg) newext=jpg  ;;
   x-portable-bitmap) newext=pbm  ;;
  x-portable-greymap) newext=pgm  ;;
                 png) newext=png  ;;
   x-portable-pixmap) newext=ppm  ;;
                tiff) newext=tiff ;;
                webp) newext=webp ;;
                   *) newext=''   ;;
esac

[[ -n $newext ]] || exit 0

case $newext in
  jpg|png) feh -U -- "$1" >/dev/null || exit 0 ;;
        *) identify -format '' -- "$1" >/dev/null || exit 0 ;;
esac

if [[ $newext == 'jpg' ]]; then
  # Detect broken/unreadable multi-image JPEGs
  python - "$1" || exit 0 <<'EOF'
import sys; from PIL import Image
with Image.open(sys.argv[1]) as img:
  if getattr(img, 'is_animated', False):
    img.seek(img.n_frames - 1)
EOF
fi

printf '%s\n' "$newext"
