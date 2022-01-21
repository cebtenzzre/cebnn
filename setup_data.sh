#!/usr/bin/env bash
set -euo pipefail

script_dir=$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")

(( $# )) || { printf 'Error: Expected at least one argument, got %s\n' $#; exit 1; }
data_dir=$1 scaled_dir=$(realpath -e -- "$2")
set --

cd -- "$data_dir"

find class -mindepth 1 -maxdepth 1 -type d -printf '%P,' | sed 's/,$/\n/' >classes.txt

# Split and tag
echo 'Splitting and tagging...'
"${script_dir}/tag_all.py"

# Scale
echo 'Scaling images...'
ddname=${data_dir#data/}
my_scaled_dir="${scaled_dir}/${ddname}"
mkdir "$my_scaled_dir"
ln -sT "$my_scaled_dir" images
find orig -xtype f -print0 | SCRIPT_DIR=$script_dir nice parallel -0r -n1 'bn={/} && "${SCRIPT_DIR}/scale.py" {} images/"${bn%.*}".png'
