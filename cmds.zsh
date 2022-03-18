_script_dir=$(cd "$(dirname -- "${(%):-%x}")";  pwd)
source "${_script_dir}/.venv/bin/activate"

model=rexnet_200
dataset=my_dataset
batch_size=32
weight_decay=1e-2
l2reg=1e-2

_getbase() {
  if [[ -z ${base+x} ]]; then
    local base=( --base-model="${model}" )
  fi
  printf '%s\n' ${(pj: :)${(q+)base}}  # Quote the array
}

train() {
  (( $# < 4 )) && return 1
  local n=$1 epochs=$2 lr=$3; shift 3
  ./cebnn_main.py --data-dir="data/${dataset}" --task=train --save="nets/${dataset}/${model}_${n}.torch" "${(@Q)${(z)$(_getbase)}}" --batch-size=${batch_size} --epochs=${epochs} --post-resample=none --wd=${weight_decay} --lr=${lr} --l2reg=${l2reg} --tta=mean "$@"
}

find_lr() {
  ./cebnn_main.py --data-dir=data/${dataset} --task=find_lr --quick-findlr "${(@Q)${(z)$(_getbase)}}" --batch-size=${batch_size} --optimizer=sgdw --wd=${weight_decay} --l2reg=${l2reg} "$@"
}

find_aug() {
  (( ! $# )) && return 1
  local lr=$1; shift 1
  ./cebnn_main.py --data-dir=data/${dataset} --task=find_aug --quick-findlr "${(@Q)${(z)$(_getbase)}}" --batch-size=${batch_size} --optimizer=sgdw --wd=${weight_decay} --lr=${lr} --l2reg=${l2reg} "$@"
}

train_more() {
  (( $# < 3 )) && return 1
  local old_n=$1 n=$2 epochs=$3; shift 3
  ./cebnn_main.py --data-dir="data/${dataset}" --task=train --load="nets/${dataset}/${model}_${old_n}.torch" --save="nets/${dataset}/${model}_${n}.torch" --epochs=${epochs} --l2reg=${l2reg} "$@"
}

train_xgb() {
  (( ! $# )) && return 1
  local n=$1; shift 1
  ./cebnn_main.py --data-dir="data/${dataset}" --task=train_xgb --save="nets/${dataset}/xgb_${n}.torch" --batch-size=64 --cvfolds=5 --post-oversample=0 "$@"
}

plot_roc() (
  set -euo pipefail
  (( ! $# )) && return 1
  for cp in "$@"; do printf '%s\n' "$cp"; ./cebnn_main.py --data-dir="data/${dataset}" --task=roc --fig-dir="nets/${dataset}/figs" --load="$cp"; done
)

get_correct() (
  set -euo pipefail
  (( $# < 2 )) && return 1
  thresh_opt_metric=$1; shift 1
  for cp in "$@"; do printf '%s\n' "$cp"; ./cebnn_main.py --data-dir="data/${dataset}" --task=get_correct --correct-dir="nets/${dataset}/correct${thresh_opt_metric}" --load="$cp"; done
)

get_correct_test() (
  set -euo pipefail
  (( $# < 2 )) && return 1
  thresh_opt_metric=$1; shift 1
  for cp in "$@"; do printf '%s\n' "$cp"; ./cebnn_main.py --data-dir="data/${dataset}" --task=get_correct --correct-dir="nets/${dataset}/correct${thresh_opt_metric}_test" --load="$cp" --test-with-cpickle-thr="nets/${dataset}/correct${thresh_opt_metric}/${cp##*/}_correct.pkl"; done
)

eval_test() (
  set -euo pipefail
  (( ! $# )) && return 1
  for cp in "$@"; do printf '%s\n' "$cp"; ./cebnn_main.py --data-dir="data/${dataset}" --task=eval_test --eval-dir="nets/${dataset}/eval" --load="$cp"; done
)

sorted_metric() (
  set -euo pipefail
  (( $# < 2 )) && return 1
  metric=$1; shift 1
  find "$@" -maxdepth 0 -xtype f -print0 | sed -z 's|^\./||' | {
    i=0
    while IFS= read -rd '' file; do
      # Progress bar
      python - $i $# <<'EOF'
import collections, io, os, sys, types, tqdm
if os.isatty(sys.stderr.fileno()):
  wr = lambda _, s: realwr(s.replace('\n', '\r'))
  sys.stderr.write, realwr = types.MethodType(wr, sys.stderr), sys.stderr.write
  collections.deque(tqdm.tqdm(range(int(sys.argv[1])), total=int(sys.argv[2]), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'), maxlen=0)
EOF

      mf="metrics/${file#*/}.txt"
      if [[ -e $mf ]] && (( $(stat -c %Y -- "$mf") >= $(stat -Lc %Y -- "$file") )); then
        printf '%s\n' "$file"
        cat -- "$mf"
      elif [[ ${METRIC_CACHEDONLY-} -eq 1 ]]; then
        continue
      else
        printf '%s\n' "$file"
        mkdir -p -- "$(dirname -- "$mf")"
        trap 'command rm -f -- "$mf"' INT TERM EXIT
        ./cebnn_main.py --data-dir=data/${dataset} --task=metrics --load="$file" 2>|stderr.log | command tee "$mf"
        trap - INT TERM EXIT
      fi
      (( ++i ))
    done
    echo -n >&2 '\x1b[2K'
  } | sed -n -e '/^nets\//p' -e "/${metric}/{n;s/^\\s*//p}" \
    | xargs -rd '\n' -n2 python -c "$(<<'EOF'
import ast, sys
net, valstr = sys.argv[1:]
vals = tuple(float(v) for v in ast.literal_eval(valstr).values())
print("{}@{:#.4}@{}".format(net, sum(vals) / len(vals), valstr))
EOF
  )" | column -ts '@' | sort -gk2
)

alias mcc='sorted_metric "MCC"'
alias roc_auc='sorted_metric "ROC AUC"'

printnet() {
  python3.10 - "$1" <<'EOF' | jq
import json, sys, torch
from torchvision import transforms
def default(o):
  return o.transforms if isinstance(o, transforms.Compose) else repr(o)
pkl = {k: v for k, v in torch.load(sys.argv[1]).items() if not (k in ("modules", "dataset_indices", "params_trained", "random_state", "funky_random_state", "history") or k.endswith("_state_dict"))}
json.dump(pkl, sys.stdout, default=default)
EOF
}

printhist() {
  python3.10 - "$1" <<'EOF'
import sys, torch
from collections import namedtuple
from skorch.callbacks import PrintLog
FakeNet = namedtuple('FakeNet', ('verbose', 'history'))
pl = PrintLog().initialize()
hist = torch.load(sys.argv[1])['history']
for i, _ in enumerate(hist):
    pl.on_epoch_end(FakeNet(1, hist[:i+1]))
EOF
}

printnet_full() {
  python3.10 - "$1" <<'EOF' | jq
import dis, json, sys, torch, types
from torchvision import transforms
def default(o):
  if isinstance(o, transforms.Compose):
    return o.transforms
  elif type(o) is types.FunctionType:
    return (dis.Bytecode(o).dis(), getattr(o, '__dict__', None))
  if callable(o) and hasattr(type(o), '__init__') and callable(getattr(type(o), '__call__', None)) and type(type(o).__call__).__name__ != 'wrapper_descriptor':
    return (dis.Bytecode(type(o).__call__).dis(), getattr(o, '__dict__', None))
  return (repr(o), getattr(o, '__dict__', None))
pkl = {k: v for k, v in torch.load(sys.argv[1]).items() if not (k in ("modules", "random_state", "funky_random_state", "history") or k.endswith("_state_dict"))}
json.dump(pkl, sys.stdout, default=default)
EOF
}

LINT_FILES=( algorithm.py best_majvote_eval.py best_majvote.py cpickledir_checksame.py datamunge.py dataset.py dedup_listing.py eval_print_majvote.py eval_print.py eval_print_sorted.py eval.py eval_stream.py cebnn_common.py cebnn_main.py losses.py merge_results.py scale.py subtract_listings.py tag_all.py util.py )

_lint() {
  # Make sure we can import the main file
  if (( ${@[(Ie)cebnn_main.py]} )); then
    python -c 'import cebnn_main' || return 1
  fi

  # Modules with stubs have to be checked separately
  mypy --python-executable="$(which python)" util.py || return 1
  mypy --python-executable="$(which python)" || return 1

  (( $# )) || return
  pytype -j auto --keep-going --strict_namedtuple_checks --precise-return "$@"
  flake8 --max-line-length=120 --select=F,U100,E501,W291 --ignore=F811 "$@"
}

lint() {
  setopt local_options
  set -o pipefail

  local touched_files
  touched_files=( ${(@0)"$(git status -z --porcelain=v1 "${LINT_FILES[@]}" | sed -z 's/^.. //')"} ) || return 1

  _lint "${touched_files[@]}"
}

lint_full() { _lint "${LINT_FILES[@]}"; }
