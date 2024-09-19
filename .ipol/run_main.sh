#!/bin/bash
# Arguments: freq distance min_z max_z

freq=$1
distance=$2
min_z=$3
max_z=$4

if [ -f input_2.txt -a -f input_3.json ]
then
    ref="-i3 input_2.txt -i4 input_3.json"
fi

python $bin/main.py -i0 input_0.txt -i1 $input_1 -i2 $input_2 $ref -freq $freq -distance $distance -min_z $min_z -max_z $max_z
