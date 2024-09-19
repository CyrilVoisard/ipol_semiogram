import os
import subprocess

# Arguments: freq, distance, min_z, max_z
freq = input("Enter freq: ")
distance = input("Enter distance: ")
min_z = input("Enter min_z: ")
max_z = input("Enter max_z: ")

ref = ""
if os.path.isfile('input_2.txt') and os.path.isfile('input_3.json'):
    ref = "-i3 input_2.txt -i4 input_3.json"

command = f"python $bin/main.py -i0 input_0.txt -i1 $input_1 -i2 $input_2 {ref} -freq {freq} -distance {distance} -min_z {min_z} -max_z {max_z}"

subprocess.run(command, shell=True)
