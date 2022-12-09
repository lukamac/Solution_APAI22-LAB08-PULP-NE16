#!/bin/sh
STATES="\bLOAD\b\|\bMATRIXVEC\b\|NORMQUANT_MULT\|NORMQUANT_BIAS\|\bSTREAMOUT\b"
if [ $# -eq 0 ];
then
    grep $STATES | sed 's/.*State \(.*$\)/\1/' | sort | uniq -c
else
    grep $STATES $1 | sed 's/.*State \(.*$\)/\1/' | sort | uniq -c
fi

