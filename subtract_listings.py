#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import sys

READ_SIZE = 128 * 1024


def gethash(path: str) -> bytes:
    hsh = hashlib.blake2b()
    with open(path, 'rb') as ef:
        while chunk := ef.read(READ_SIZE):
            hsh.update(chunk)
    return hsh.digest()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Wrong number of arguments')

    minuend, subtrahend = sys.argv[1:]
    excluded_hashes = set()

    with open(subtrahend, 'r') as sf:
        for path in sf:
            excluded_hashes.add(gethash(path.rstrip('\n')))

    with open(minuend, 'r') as mf:
        for path in mf:
            path = path.rstrip('\n')
            digest = gethash(path)
            if digest in excluded_hashes:
                continue  # Dupe or subtraction!
            excluded_hashes.add(digest)
            print(path, flush=True)
