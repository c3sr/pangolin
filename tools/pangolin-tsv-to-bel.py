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
with open(sys.argv[1], 'rb') as inf:
    with open(sys.argv[2], 'wb') as outf:
        for line in inf:
            dst, src, weight = line.split()
            dstBytes = struct.pack("<Q", int(dst))
            srcBytes = struct.pack("<Q", int(src))
            weightBytes = struct.pack("<Q", int(weight))
            outf.write(dstBytes + srcBytes + weightBytes)
