#!/bin/bash

for (( i=0; i<$1; i+=4 ))
do
    for (( j=$i; j<$i+4; j++ ))
    do
        if (($j < $1))
        then
            echo $j
            mkdir -p $j
            cd $j
            python ../../blender.py $j > output &
            cd ..
        fi
    done

    wait
    wait
    wait
    wait
done
