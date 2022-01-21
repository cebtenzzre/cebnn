#!/usr/bin/env bash
set -euo pipefail

readonly all_subdir='orig'
readonly class_subdir='class'
readonly weight_subdir='weight'
readonly test_subdir='test'
readonly test_only_subdir='test_only'

follow_links=0

while [[ $# -gt 0 ]]; do
  opt=$1
  case "$opt" in
    --) shift; break ;;
    -*)
      for ((i=1; i<${#opt}; i++)); do
        subopt=${opt:$i:1}
        case "$subopt" in
          # Help
          h)
            echo >&2 "Usage: $(basename -- "${BASH_SOURCE[0]}") [-L] [--] dest_dir [find depth]:[class]:source_dir ...";
            exit
            ;;
          L) follow_links=1 ;;
          *) echo >&2 "Unrecognized option: -$subopt"; exit 1 ;;
        esac
      done ;;
    *) break ;;
    esac
  shift
done

dest_dir=$1; shift
sources=( "$@" )
set --

[[ -d $dest_dir ]] || mkdir -- "$dest_dir"

# This is important because of the way we find conflicts
if [[ -e "${dest_dir}/${all_subdir}" ]]; then
  echo >&2 'Error: Output directory not empty.'
  exit 1
fi

find_opts=()
(( follow_links )) && find_opts=( '-L' )

# Limit searched files to those that seem like media
find_args=( '(' )
for ext in 'bmp' 'exr' 'gif' 'gifv' 'jpg' 'jpeg' 'pbm' 'pgm' 'png' 'ppm' 'tif' \
           'tiff' 'webp'; do
  find_args+=( -iname "*.${ext}" -o )
done
find_args[${#find_args[@]}-1]=')'  # Replace last -o with closing parenthesis

escape_findpath() {
  local path=$1
  local escaped_path
  case "$path" in
    # Absolute paths are OK
     /*) escaped_path=$path ;;
    # Explicitly relative paths are OK
      .) ;&
    ./*) escaped_path=$path ;;
    # Implicitly relative paths are made explicit, as recommended by man find(1)
     -*) [[ -z ${escape_warned+x} ]] && { echo >&2 'Warning: Found a path starting with a dash, escaping it.'; escape_warned=1; } ;&
      *) escaped_path="./${path}" ;;
  esac
  printf '%s\n' "$escaped_path"
}

escape_findpattern() {
  <<<"$1" sed 's|[][\*?]|\\&|g'
}

dolink() {
  local dest=$1 srcf=$2 dest_fn=$3
  local dstf="$dest"/"$dest_fn"

  if ! [[ -e $dest ]]; then
    mkdir -p -- "$dest"
  elif [[ -e $dstf ]]; then
    return 0  # Link not needed (TOCTOU, but we have set -e... *shrug*)
  fi

  ln -sT -- "$srcf" "$dest"/"$dest_fn"
}

declare -A orig_files

for src in "${sources[@]}"; do
  src_find_depth=${src%%:*}; src=${src#*:}
  src_find_args=()
  [[ -n $src_find_depth ]] && src_find_args=( -maxdepth "$src_find_depth" )
  src_class=${src%%:*}; src=${src#*:}
  escaped_path=$(escape_findpath "${src#*:}")

  # Parse classes
  src_classes=() src_weight=1 src_test=0 src_test_only=0
  if [[ -n $src_class ]]; then
    while IFS= read -r -d ',' c; do
      if [[ $c =~ ^'weight='[0-9.]+$ ]]; then
        src_weight=${c/#weight=/}
      elif [[ $c == 'test' ]]; then
        src_test=1
      elif [[ $c == 'test_only' ]]; then
        src_test_only=1
      else
        src_classes+=( "$c" )
      fi
    done < <(printf '%s,' "$src_class")
    unset c
  fi
  unset src_class

  printf 'Reading %s...\n' "$escaped_path"

  # shellcheck disable=SC2086
  while IFS= read -r -d '' f; do
    IFS= read -r -d '' norm_ext
    fname=${f##*/}
    dest_fname="${fname%.*}.${norm_ext}"
    src_file=$(realpath -es -- "$f")

    dest="$dest_dir"/"$all_subdir"
    dupe_found=0  # We need this in case a file is found again for a different class

    # Find an available name
    num=1
    while true; do
      if [[ -z ${orig_files["${dest_fname%.*}"]+x} ]]; then
        break  # Found one!
      fi
      conflict_ext=${orig_files["${dest_fname%.*}"]}
      conflict_fname="${dest_fname%.*}.${conflict_ext}"
      if cmp -s -- "$src_file" "${dest}/${conflict_fname}"; then
        dest_fname=$conflict_fname
        dupe_found=1
        break
      fi

      # Legitimate conflict, rename it
      dest_fname="${fname%.*} (${num}).${norm_ext}"
      (( ++num ))
    done

    # Link it!
    if ! (( dupe_found )); then
      orig_files["${dest_fname%.*}"]=${dest_fname##*.}
      dolink "$dest_dir"/"$all_subdir" "$(realpath -- "$src_file")" "$dest_fname"
    fi
    for c in "${src_classes[@]}"; do
      dolink "$dest_dir"/"$class_subdir"/"$c" ../../"$all_subdir"/"$dest_fname" "$dest_fname"
    done
    dolink "$dest_dir"/"$weight_subdir"/"$src_weight" ../../"$all_subdir"/"$dest_fname" "$dest_fname"
    if (( src_test )); then
      dolink "$dest_dir"/"$test_subdir" ../"$all_subdir"/"$dest_fname" "$dest_fname"
    fi
    if (( src_test_only )); then
      dolink "$dest_dir"/"$test_only_subdir" ../"$all_subdir"/"$dest_fname" "$dest_fname"
    fi
  done < <(find "${find_opts[@]}" -- "$escaped_path" "${src_find_args[@]}" \! \( -name '.*' -prune \) -xtype f "${find_args[@]}" -print0 | parallel -0r -n1 'ext=$(./get_image_ext.sh {}); if [[ -n $ext ]]; then printf "%s\0%s\0" {} "$ext"; fi')
done

# Clean up with rmlint (NB: requires patch: "Make follow_symlinks do what I would expect")
export all_subdir
export ALL_DIR="$dest_dir"/"$all_subdir"
export CLASS_DIR="$dest_dir"/"$class_subdir"
export WEIGHT_DIR="$dest_dir"/"$weight_subdir"
export TEST_DIR="$dest_dir"/"$test_subdir"
export TEST_ONLY_DIR="$dest_dir"/"$test_only_subdir"
export -f escape_findpattern
rmlint -o sh:stdout -VVV -fx -T df -S 'X< \([0-9]+\)\.[^ ]+$>ma' -c sh:cmd='bad_fn=${1##*/}; good_fn=${2##*/}; dirs=( "$CLASS_DIR" ); [[ -d $WEIGHT_DIR ]] && dirs+=( "$WEIGHT_DIR" ); [[ -d $TEST_DIR ]] && dirs+=( "$TEST_DIR" ); [[ -d $TEST_ONLY_DIR ]] && dirs+=( "$TEST_ONLY_DIR" ); find "${dirs[@]}" -lname "*/$(escape_findpattern "$all_subdir/$bad_fn")" -delete -exec bash -c '\''dest=$(dirname -- "$1"); ln -rst "$dest" -- "$ALL_DIR/$2"'\'' _ {} "$good_fn" \; ; rm -- "$1"' "$dest_dir"/"$all_subdir" | bash -s -- -dxpq
find "$(escape_findpath "$dest_dir"/"$class_subdir")" -xtype l -printf 'Found dead link: %p\n'
