#!/usr/bin/env python
# - *- coding: utf- 8 - *-
########################################################################
# 
# Copyright (c) 2021 root All Rights Reserved
# 
########################################################################
 
'''
 
Author: root
Date: 2021/05/24 08:17:10
'''


sep = {"[punc]", "[SEP]"}

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 4
    group_f = sys.argv[1]
    input_f = sys.argv[2]
    save_f = sys.argv[3]
    pairs = []

    with open(group_f, 'r') as rf:
        for l in rf:
            pairs.append([len(x.split()) for x in l.strip().split("|")])

    wf = open(save_f, 'w')

    with open(input_f, 'r') as rf:
        for k, l in enumerate(rf):
            ws = []
            ids = 0
            for w in l.strip().split():
                ws += [w]
                if w in sep:
                    continue
                if ids < len(pairs[k]) and pairs[k][ids] > 1:
                    ws += [f"_{w}" for i in range(pairs[k][ids]-1)]
                ids += 1
            wf.write(" ".join(ws) + "\n")
    wf.close()
