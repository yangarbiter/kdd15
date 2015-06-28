#!/bin/bash
for (( i=0; i<$1; i++ ))
do
    #grep 'exclude' ./$i/output
    #tail -n 1 ./$i/blending.model
    grep 'auc val' ./$i/output
done
