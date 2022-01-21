#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import pickle
import sys
from typing import TYPE_CHECKING

from util import zip_strict

if TYPE_CHECKING:
    from typing import Any, Dict, List, Union
    from util import Array
    StrPath = Union[str, 'os.PathLike[str]']


def getfiles(dir_: str) -> List[os.DirEntry[str]]:
    files = (e for e in os.scandir(dir_) if e.is_file())
    return sorted(files, key=lambda e: e.name)


def load(path: StrPath) -> Dict[str, Any]:
    with open(path, 'rb') as pf:
        return pickle.load(pf)


def wrong_pct(dataa: Array, datab: Array) -> float:
    return 100 * sum(1 for a, b in zip_strict(dataa, datab) if a != b) / len(dataa)


if __name__ == '__main__':
    dira, dirb = sys.argv[1:]
    filesa, filesb = getfiles(dira), getfiles(dirb)
    assert filesa and len(filesa) == len(filesb)

    fail = False
    for filea, fileb in zip_strict(filesa, filesb):
        assert filea.name == fileb.name
        pkla, pklb = load(filea), load(fileb)
        if pkla['y_pred'] != pklb['y_pred']:
            print(
                'Found y_pred discrepcancy: {} != {} ({:.1f}% wrong)'.format(
                    filea.name, fileb.name, wrong_pct(pkla['y_pred'], pklb['y_pred'])),
                file=sys.stderr,
            )
            fail = True
        if pkla['y_true'] != pklb['y_true']:
            print(
                'Found y_true discrepcancy: {} != {} ({:.1f}% wrong)'.format(
                    filea.name, fileb.name, wrong_pct(pkla['y_true'], pklb['y_true'])),
                file=sys.stderr,
            )
            fail = True

    if fail:
        print('Verification failed.')
    else:
        print('All files verified successfully.')
