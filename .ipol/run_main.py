import argparse
import os
import subprocess

ref = ""
compare = False
if os.path.isfile('input_2.txt') and os.path.isfile('input_3.json'):
    ref = "-i2 $input_2 -i3 $input_3"

command = f"python $bin/main.py -i0 $input_0 -i1 $input_1 {ref} -freq $freq -distance $distance -min_z $min_z -max_z $max_z"
command = "python $bin/main.py -i0 $input_0 -i1 $input_1 -freq $freq -distance $distance -min_z $min_z -max_z $max_z"

subprocess.run(command, shell=True)
