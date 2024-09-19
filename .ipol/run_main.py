import argparse
import os
import subprocess

# Create the parser
parser = argparse.ArgumentParser(description="Process para.")

# Add the arguments
parser.add_argument('freq', type=str, help='The frequency')
parser.add_argument('distance', type=str, help='The distance')
parser.add_argument('min_z', type=str, help='The minimum z')
parser.add_argument('max_z', type=str, help='The maximum z')

# Parse the arguments
args = parser.parse_args()

ref = ""
if os.path.isfile('input_2.txt') and os.path.isfile('input_3.json'):
    ref = "-i2 $input_2 -i3 $input_3"

command = f"python $bin/main.py -i0 $input_0 -i1 $input_1 {ref} -freq $freq -distance $distance -min_z $min_z -max_z $max_z"
#command = "python $bin/main.py -i0 $input_0 -i1 $input_1 -i2 $input_2 -i3 $input_3 -freq $freq -distance $distance -min_z $min_z -max_z $max_z"

subprocess.run(command, shell=True)
