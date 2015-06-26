#!/bin/bash

DIRS="/tmp2/kddblending/team1/*
/tmp2/kddblending/team2/*
/tmp2/b01902066/KDD/kdd15/blending/allpreds/mypreds/*
/tmp2/kddblending/team3/*
/tmp2/kddblending/team4/*
/tmp2/kddblending/team5/*"

mkdir -p $1
for dir in $DIRS
do
    cp $dir $1
done
