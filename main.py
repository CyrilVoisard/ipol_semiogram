#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import sys

from package import import_data, compute_semio_val, radar_design

# if you need to access a file next to the source code, use the variable ROOT
ROOT = os.path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)

def print_semio_parameters(parameters_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

    display_dict = {'Subject': "Subject: {Subject}".format(**parameters_dict),
                    'Trial': "Trial: {Trial}".format(**parameters_dict),
                    'Age': "Age (year): {Age}".format(**parameters_dict),
                    'Gender': "Gender: {Gender}".format(**parameters_dict),
                    'Height': "Height (m): {Height}".format(**parameters_dict),
                    'Weight': "Weight (kg): {Weight}".format(**parameters_dict)
                    }
    info_msg = """
    {Subject:^30}|{Trial:^30}
    ------------------------------+------------------------------
    {Age:<30}| {WalkingSpeed:<30}
    {Height:<30}| Number of footsteps:
    {Weight:<30}| {LeftGaitCycles:<30}
    {UTurnDuration:<30}| {RightGaitCycles:<30}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_parameters.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        #for name, value in metadata_dict.items():
         #   f.write(f"{name} = {value}\n")

def print_semio_criteria(criteria_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

    print("criteria_dict", criteria_dict)

    display_dict = {'Average Speed': "Average Speed: {Average Speed}".format(**criteria_dict),
                    'Springiness': "Springiness: {Springiness}".format(**criteria_dict),
                    'Sturdiness': "Sturdiness: {Sturdiness}".format(**criteria_dict),
                    'Smoothness': "Smoothness: {Smoothness}".format(**criteria_dict),
                    'Steadiness': "Steadiness: {Steadiness}".format(**criteria_dict),
                    'Stability': "Stability: {Stability}".format(**criteria_dict),
                    'Symmetry': "Symmetry: {Symmetry}".format(**criteria_dict),
                    'Synchronisation': "Synchronisation: {Synchronisation}".format(**criteria_dict)
                    }
    info_msg = """
    Z-Scores
    -------------------------------------------+-------------------------------------------
    {Average Speed:<50}| {Steadiness:<50}
    {Springiness:<50}| {Stability:<50}
    {Sturdiness:<50}| {Symmetry:<50}
    {Smoothness:<50}| {Synchronisation:<50}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_criteria.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        #for name, value in metadata_dict.items():
         #   f.write(f"{name} = {value}\n")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Return a semiogram for a given trial.')
    parser.add_argument('-i0', metavar='data_lb',
                        help='Time series for the lower back sensor.')
    parser.add_argument('-i1', metavar='gait_events',
                        help='Metadata with gait events.')
    parser.add_argument('-freq', metavar='freq',
                        help='Acquistion frequency.')
    parser.add_argument('-age', metavar='age', type=int,
                        help='Age of the subject.')
    parser.add_argument('-min_z', metavar='min_z', type=int,
                        help='Minimum for Z-score.')
    parser.add_argument('-max_z', metavar='max_z', type=int,
                        help='Maximum for Z-score.')
    args = parser.parse_args()

    freq = int(args.freq)
    #age = args.age
    age = None 
    
    # load data (only lower back in this demo)
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0))
    seg_lim = import_data.get_seg(os.path.join(data_WD, args.i1))
    steps_lim = import_data.get_steps(os.path.join(data_WD, args.i1), seg_lim)
    
    # compute semio values (dictionnaries)
    criteria_names, criteria, parameters = compute_semio_val.compute_semio_val(age, steps_lim, seg_lim, data_lb, freq)

    # print semiogram values
    #print_semio_parameters(parameters)

    criteria_dict = dict(zip(criteria_names, criteria))
    print_semio_criteria(criteria_dict)

    # semiogram design
    radar_design.new_radar_superpose({"unique": criteria}, None,  min_r=int(args.min_z), max_r=int(args.max_z), output=data_WD)
    print("ok charge")
    sys.exit(0)
