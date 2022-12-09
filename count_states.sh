#!/bin/sh
if [ $# -eq 0 ];
then
    grep "State" | sed 's/.*\(State .*$\)/\1/' | sort | uniq -c
else
    grep "State" $1 | sed 's/.*\(State .*$\)/\1/' | sort | uniq -c
fi

