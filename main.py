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

    display_dict = {'V': "V: {V}".format(**parameters_dict),
                    'StrT': "StrT: {StrT}".format(**parameters_dict),
                    'UTurn': "UTurn: {UTurn}".format(**parameters_dict),
                    'SteL': "SteL: {SteL}".format(**parameters_dict),
                    'SPARC_gyr': "SteL: {SteL}".format(**parameters_dict),
                    'LDLJAcc': "LDLJAcc: {LDLJAcc}".format(**parameters_dict),
                    'CVStrT': "CVStrT: {CVStrT}".format(**parameters_dict),
                    'CVdsT': "CVdsT: {CVdsT}".format(**parameters_dict),
                    'P1_aCC': "P1_aCC: {P1_aCC}".format(**parameters_dict),
                    'P2_aCC': "P2_aCC: {P2_aCC}".format(**parameters_dict),
                    'ML_RMS': "ML_RMS: {ML_RMS}".format(**parameters_dict),
                    'P1P2': "P1P2: {P1P2}".format(**parameters_dict),
                    'MSwTR': "MSwTR: {MSwTR}".format(**parameters_dict),
                    'AP_iHR': "AP_iHR: {AP_iHR}".format(**parameters_dict),
                    'ML_iHR': "ML_iHR: {ML_iHR}".format(**parameters_dict),
                    'CC_iHR': "CC_iHR: {CC_iHR}".format(**parameters_dict),
                    'dstT': "dstT: {dstT}".format(**parameters_dict), 
                    'sd_V': "{sd_V}".format(**parameters_dict),
                    'sd_StrT': "{sd_StrT}".format(**parameters_dict),
                    'sd_UTurn': "{sd_UTurn}".format(**parameters_dict),
                    'sd_SteL': "{sd_SteL}".format(**parameters_dict),
                    'sd_SPARC_gyr': "{sd_SteL}".format(**parameters_dict),
                    'sd_LDLJAcc': "{sd_LDLJAcc}".format(**parameters_dict),
                    'sd_CVStrT': "{sd_CVStrT}".format(**parameters_dict),
                    'sd_CVdsT': "{sd_CVdsT}".format(**parameters_dict),
                    'sd_P1_aCC': "{sd_P1_aCC}".format(**parameters_dict),
                    'sd_P2_aCC': "{sd_P2_aCC}".format(**parameters_dict),
                    'sd_ML_RMS': "{sd_ML_RMS}".format(**parameters_dict),
                    'sd_P1P2': "{sd_P1P2}".format(**parameters_dict),
                    'sd_MSwTR': "{sd_MSwTR}".format(**parameters_dict),
                    'sd_AP_iHR': "{sd_AP_iHR}".format(**parameters_dict),
                    'sd_ML_iHR': "{sd_ML_iHR}".format(**parameters_dict),
                    'sd_CC_iHR': "{sd_CC_iHR}".format(**parameters_dict),
                    'sd_dstT': "{sd_dstT}".format(**parameters_dict)
                    
                    }
    info_msg = """
    Values                        | Z-Scores
    ------------------------------+------------------------------
    {V:<30}| {sd_V:<30}
    {StrT:<30}| {sd_StrT:<30}
    {UTurn:<30}| {sd_UTurn:<30}
    {SteL:<30}| {sd_SteL:<30}
    {SPARC_gyr:<30}| {sd_SPARC_gyr:<30}
    {LDLJAcc:<30}| {sd_LDLJAcc:<30}
    {CVStrT:<30}| {sd_CVStrT:<30}
    {CVdsT:<30}| {sd_CVdsT:<30}
    {P1_aCC:<30}| {sd_P1_aCC:<30}
    {P2_aCC:<30}| {sd_P2_aCC:<30}
    {ML_RMS:<30}| {sd_ML_RMS:<30}
    {P1P2:<30}| {sd_P1P2:<30}
    {MSwTR:<30}| {sd_MSwTR:<30}
    {AP_iHR:<30}| {sd_AP_iHR:<30}
    {ML_iHR:<30}| {sd_ML_iHR:<30}
    {CC_iHR:<30}| {sd_CC_iHR:<30}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_parameters.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        

def print_semio_criteria(criteria_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

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
    --------------------------------------------------+--------------------------------------------------
    {Average Speed:<50}| {Steadiness:<50}
    {Springiness:<50}| {Stability:<50}
    {Sturdiness:<50}| {Symmetry:<50}
    {Smoothness:<50}| {Synchronisation:<50}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_criteria.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Return a semiogram for a given trial.')
    parser.add_argument('-i0', metavar='data_lb',
                        help='Time series for the lower back sensor.')
    parser.add_argument('-i1', metavar='gait_events',
                        help='Metadata with gait events.')
    # comparaison
    parser.add_argument('-i2', metavar='ref_data_lb',
                            help='Reference set - Time series for the lower back sensor.')#, required=False)
    parser.add_argument('-i3', metavar='ref_gait_events',
                            help='Reference set - Metadata with gait events.')#, required=False)
    compare = True
        
    parser.add_argument('-freq', metavar='freq',
                        help='Acquistion frequency.')
    parser.add_argument('-age', metavar='age', type=int,
                        help='Age of the subject.')
    parser.add_argument('-distance', metavar='distance', type=int,
                        help='Walked distance (m).')
    parser.add_argument('-min_z', metavar='min_z', type=int,
                        help='Minimum for Z-score.')
    parser.add_argument('-max_z', metavar='max_z', type=int,
                        help='Maximum for Z-score.')
    args = parser.parse_args()

    freq = int(args.freq)
    distance = int(args.distance)
    #age = args.age
    age = None 
    
    # load data (only lower back in this demo)
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0), freq)
    seg_lim = import_data.get_seg(os.path.join(data_WD, args.i1))
    steps_lim = import_data.get_steps(os.path.join(data_WD, args.i1), seg_lim)
    if compare :
        ref_data_lb = import_data.import_XSens(os.path.join(data_WD, args.i2), freq)
        ref_seg_lim = import_data.get_seg(os.path.join(data_WD, args.i3))
        ref_steps_lim = import_data.get_steps(os.path.join(data_WD, args.i3), seg_lim)
    
    # compute semio values (dictionnaries)
    criteria_names, criteria, parameters = compute_semio_val.compute_semio_val(age, distance, steps_lim, seg_lim, data_lb, freq)
    if compare:
        ref_criteria_names, ref_criteria, ref_parameters = compute_semio_val.compute_semio_val(age, distance, ref_steps_lim, ref_seg_lim, ref_data_lb, freq)

    # print semiogram values
    parameters_names = ["StrT", "sd_StrT", "UTurn", "sd_UTurn", "SteL", "sd_SteL",
              "SPARC_gyr", "sd_SPARC_gyr", "LDLJAcc", "sd_LDLJAcc", "CVStrT", "sd_CVStrT", "CVdsT", "sd_CVdsT",
              "P1_aCC", "sd_P1_aCC", "P2_aCC", "sd_P2_aCC", "ML_RMS", "sd_ML_RMS", "P1P2", "sd_P1P2",
              "MSwTR", "sd_MSwTR", "AP_iHR", "sd_AP_iHR", "ML_iHR", "sd_ML_iHR", "CC_iHR", "sd_CC_iHR", "dstT", "sd_dstT",
              "V", "sd_V"]
    parameters_dict = dict(zip(parameters_names, parameters))
    print_semio_parameters(parameters_dict)

    criteria_dict = dict(zip(criteria_names, criteria))
    print_semio_criteria(criteria_dict)

    # semiogram design
    radar_design.new_radar_superpose({"unique": criteria}, min_r=int(args.min_z), max_r=int(args.max_z), output=data_WD, name="semio")
    if compare : 
        radar_design.new_radar_superpose({"unique": ref_criteria}, min_r=int(args.min_z), max_r=int(args.max_z), output=data_WD, name="semio_ref")
        radar_design.new_radar_superpose({"ref": ref_criteria, "new": criteria}, min_r=int(args.min_z), max_r=int(args.max_z), output=data_WD, name="semio_sup")
    print("ok charge")
    sys.exit(0)
