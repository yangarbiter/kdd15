#!/bin/bash

export OMP_NUM_THREADS=1
for (( i=0; i<$1; i+=10 ))
do
    for (( j=$i; j<$i+10; j++ ))
    do
        if (($j < $1))
        then
            echo $j
            mkdir -p $j
            cd $j
            python $2 $j > output &
            cd ..
        fi
    done

    for (( j=0; j<10; j++ ))
    do
        wait
    done
done
