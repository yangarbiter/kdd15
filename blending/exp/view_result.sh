#!/bin/bash
for i in `seq 0 $1`;
do
    grep 'deleted' ./$i/output
    #tail -n 1 ./$i/blending.model
    grep 'auc val' ./$i/output
done
