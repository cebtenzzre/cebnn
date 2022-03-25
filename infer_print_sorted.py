#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import pickle
import sys

if __name__ == '__main__':
    if not 2 <= len(sys.argv) < 4:
        raise ValueError('Wrong number of arguments')

    # predictions file generated by infer.py
    predictions_file, *rest = sys.argv[1:]
    class_index_str, = rest if rest else ('0',)
    class_index = int(class_index_str)

    with open(predictions_file, 'rb') as pf:
        pred_data = pickle.load(pf)

    bypred = sorted(pred_data, key=lambda fp: fp[1][class_index], reverse=True)
    for fname, pred in bypred:
        print(' '.join('{:4f}'.format(p) for p in pred), fname, sep='\t')