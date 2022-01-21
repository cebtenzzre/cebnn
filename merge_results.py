#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Used for merging results from different eval.py runs.
# Example: ./merge_results.py \
#   eval/clazz/content_4_results_clazz_2.1_1.pkl \
#   eval/clazz/content_5minus4_results_clazz_2.1_1.pkl \
#   eval/clazz/content_5.txt \
#   eval/clazz/content_5_results_clazz_2.1_1_test2.pkl

from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Any, Dict


def getval(key: str, *sources: Dict[str, Any]) -> Any:
    for source in sources:
        if (val := source.get(key)) is not None:
            return val


if __name__ == '__main__':
    apickle, diffpickle, keyfile, destpickle = sys.argv[1:]

    with open(apickle, 'rb') as pf:
        adata = dict(pickle.load(pf))
    with open(diffpickle, 'rb') as pf:
        diffdata = dict(pickle.load(pf))
    with open(keyfile) as kf:
        line_count = sum(1 for _ in kf)
        kf.seek(0)
        lines = (l.rstrip('\n') for l in tqdm(kf, total=line_count, leave=False))
        sumdata = tuple((k, getval(k, diffdata, adata)) for k in lines)

    with open(destpickle, 'wb') as pf:
        pickle.dump(sumdata, pf)

    print('{} + {} = {} -> {}'.format(len(adata), len(diffdata), len(adata) + len(diffdata), len(sumdata)))
