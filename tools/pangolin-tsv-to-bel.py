#! /usr/bin/env python

'''convert a graph challenge tsv ascii edge list to binary format (.bel)
the binary format is for each edge
* 64-bit integer dst
* 64-bit integer src
* 64-bit integer weight
all numbers are stored little endian (least significant byte first)

the number of edges is the byte-length of the file divided by 24

you can view the produced file with 
xxd -c 24 <file> to see one edge per line
'''

from __future__ import print_function
import sys
import struct
import os

tsv_path = sys.argv[1]
if len(sys.argv) > 2:
    bel_path = sys.argv[2]
else:
    assert tsv_path.endswith(".tsv")
    bel_path = tsv_path[:-4] + ".bel"

# tsv path and bel path should not be the same
assert tsv_path != bel_path

if os.path.isfile(bel_path):
    print("{} alread exists".format(bel_path))
    sys.exit(1)

with open(tsv_path, 'rb') as inf, open(bel_path, 'wb') as outf:
    for line in inf:
        dst, src, weight = line.split()
        try:
            dstBytes = struct.pack("<Q", int(dst))
            srcBytes = struct.pack("<Q", int(src))
            weightBytes = struct.pack("<Q", int(weight))
        except ValueError as e:
            print("error while converting {} to {}: {}".format(tsv_bath, bel_path, e))
            sys.exit(1)
        outf.write(dstBytes + srcBytes + weightBytes)

# the bel file should be some multiple of 24 bytes
assert os.path.getsize(bel_path) % 24 == 0