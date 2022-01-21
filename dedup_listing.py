#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import sys

READ_SIZE = 128 * 1024

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise ValueError('Wrong number of arguments')

    hashes = set()
    for entry in sys.stdin:
        entry = entry.rstrip('\r\n')
        hsh = hashlib.blake2b()
        with open(entry, 'rb') as ef:
            while chunk := ef.read(READ_SIZE):
                hsh.update(chunk)
        digest = hsh.digest()
        if digest in hashes:
            continue  # Dupe!
        hashes.add(digest)
        print(entry, flush=True)
