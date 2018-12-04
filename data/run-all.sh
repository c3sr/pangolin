#! /bin/bash

OUT=tri32_um_0.csv
ERR=tri32_um_0.log

for tsv in tsv/*.tsv; do
  (../build/tri32 -m um -g 0 $tsv | tee -a $OUT) 2>&1 | tee -a $ERR;
done