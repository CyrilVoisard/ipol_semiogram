import argparse
import os
import subprocess
import numpy as np

ref = ""
compare = 0
if os.path.isfile('input_2.txt') and os.path.isfile('input_3.json'):
    ref = "-i2 $input_2 -i3 $input_3"
    compare = 1

command = f"python $bin/main.py -i0 $input_0 -i1 $input_1 {ref} -freq $freq -distance $distance -min_z $min_z -max_z $max_z"
f = open('info.txt','w')
f.write("ref = " + str(compare))
f.close()

subprocess.run(command, shell=True)
