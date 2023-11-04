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

from package import import_data, compute_semio_val

# if you need to access a file next to the source code, use the variable ROOT
ROOT = os.path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)

def load_metadata(subject, trial):
    """Return the metadata dict for the given subject-trial.

    Arguments:
        subject {int} -- Subject number
        trial {int} -- Trial number

    Returns
    -------
    dict
        Metadata
    """

    code = str(subject) + "-" + str(trial)
    fname = os.path.join(FOLDER, code)
    with open(fname + ".json") as metadata_file:
        metadata_dict = json.load(metadata_file)
    return metadata_dict


def print_trial_info(metadata_dict):
    """Dump the trial information in a text file (trial_info.txt)

    Parameters
    ----------
    metadata_dict : dict
        Metadata of the trial.
    """

    display_dict = {'Subject': "Subject: {Subject}".format(**metadata_dict),
                    'Trial': "Trial: {Trial}".format(**metadata_dict),
                    'Age': "Age (year): {Age}".format(**metadata_dict),
                    'Gender': "Gender: {Gender}".format(**metadata_dict),
                    'Height': "Height (m): {Height}".format(**metadata_dict),
                    'Weight': "Weight (kg): {Weight}".format(**metadata_dict),
                    'WalkingSpeed': "WalkingSpeed (m/s): {}".format(round(2000/(metadata_dict['TrialBoundaries'][1]-metadata_dict['TrialBoundaries'][0]), 3)),
                    'UTurnDuration': "U-Turn Duration (s): {}".format((metadata_dict['UTurnBoundaries'][1]-metadata_dict['UTurnBoundaries'][0])/100),
                    'LeftGaitCycles': '    - Left foot: {}'.format(len(metadata_dict['LeftFootEvents'])),
                    'RightGaitCycles': '    - Right foot: {}'.format(len(metadata_dict['RightFootEvents']))
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

    with open("trial_info.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        #for name, value in metadata_dict.items():
         #   f.write(f"{name} = {value}\n")



def load_signal(subject, trial):
    """Return the signal associated with the subject-trial pair.

    Parameters
    ----------
    subject : int
        Subject number
    trial : int
        Trial number

    Returns
    -------
    numpy array
        Signal

    """
    code = str(subject) + "-" + str(trial)
    fname = os.path.join(FOLDER, code)
    signal_lb = import_XSens(fname+"_lb.txt")
    signal_lf = import_XSens(fname+"_lf.txt")
    signal_rf = import_XSens(fname+"_rf.txt")
    
    t_max = min(len(signal_lb), len(signal_rf), len(signal_lf))
    signal_lb = signal_lb[0:t_max]
    signal_lf = signal_lf[0:t_max]
    signal_rf = signal_rf[0:t_max]

    # Pour TOX, calcul plus complexe
    gyr_x = signal_lb['Gyr_X']
    angle_x_full = np.cumsum(gyr_x)/100
    a = np.median(angle_x_full[0:len(angle_x_full) // 2])  # Tout début du signal
    z = np.median(angle_x_full[len(angle_x_full) // 2:len(angle_x_full)])  # Fin du signal, en enlevant la toute fin qui posait
    angle_x_full = np.sign(z)*(angle_x_full - a)*180/abs(z)
    
    sig = {'Time': signal_lb["PacketCounter"], 'TOX': angle_x_full, 'TAX': signal_lb["Acc_X"], 'TAY': signal_lb["Acc_Y"], 
           'RAV': np.sqrt(signal_rf["FreeAcc_X"]**2 + signal_rf["FreeAcc_Y"]**2 + signal_rf["FreeAcc_Z"]**2), 
           'RAZ': signal_rf["FreeAcc_Z"], 'RRY': signal_rf["Gyr_Y"], 
           'LAV': np.sqrt(signal_lf["FreeAcc_X"]**2 + signal_lf["FreeAcc_Y"]**2 + signal_lf["FreeAcc_Z"]**2), 
           'LAZ': signal_lf["FreeAcc_Z"], 'LRY': signal_lf["Gyr_Y"]}
    
    signal = pd.DataFrame(sig)
    
    return signal


def dump_plot(signal, metadata_dict, to_plot=['TOX', 'TAX', 'TAY', 'RAV', 'RAZ', 'RRY', 'LAV', 'LAZ', 'LRY']):

    n_samples, _ = signal.shape
    tt = np.arange(n_samples) / 100

    # get limits
    acc_tronc = np.take(signal, indices=[COLUMN_NAMES[dim_name]
                                   for dim_name in to_plot if dim_name[0:2] == "TA"], axis=1)
    if acc_tronc.size > 0:
        acc_tronc_ylim = [acc_tronc.min()-0.1, acc_tronc.max()+0.1]
    
    acc = np.take(signal, indices=[COLUMN_NAMES[dim_name]
                                   for dim_name in to_plot if dim_name[1] == "A"], axis=1)
    if acc.size > 0:
        acc_ylim = [acc.min()-0.1, acc.max()+0.1]
        
    rot = np.take(signal, indices=[COLUMN_NAMES[dim_name]
                                   for dim_name in to_plot if dim_name[1] == "R"], axis=1)
    if rot.size > 0:
        rot_ylim = [rot.min()-20, rot.max()+20]

    for dim_name in to_plot:
        #print(dim_name, COLUMN_NAMES[dim_name])
        fig, ax = plt.subplots(figsize=(10, 4))
        # xlim
        ax.set_xlim(0, n_samples/100)
        # plot
        dim = COLUMN_NAMES[dim_name]
        ax.plot(tt, signal.iloc[:, dim])
        # ylim
        #if dim_name[0] in ["R", "L"]:
         #   if dim_name[1] == "A":
          #      ax.set_ylim(acc_ylim)
           # elif dim_name[1] == "R":
            #    ax.set_ylim(rot_ylim)
        #elif dim_name[0:2]== "TA":
         #   ax.set_ylim(acc_tronc_ylim)
        
        # number of yticks
        plt.locator_params(axis='y', nbins=6)
        # ylabel
        ylabel = "m/s²" if dim_name[1] == "A" else "deg/s"
        ax.set_ylabel(ylabel, fontdict={"size": 15})
        for z in ax.get_yticklabels() + ax.get_xticklabels():
            z.set_fontsize(12)
        
        ymin, ymax = ax.get_ylim()
        
        # seg annotations
        u_start, u_end = metadata_dict["UTurnBoundaries"]
        ax.vlines([u_start/100, u_end/100], ymin, ymax, color='red', linestyles="--", lw=1, label = 'U-turn Boundaries')
        ax.fill_between([u_start/100, u_end/100], ymin, ymax,
                        facecolor="red", alpha=0.2, label = "U-Turn Phase")
        # step annotations
        if dim_name[0] in ["R", "L"]:
            if dim_name[0] == "R":
                steps = metadata_dict["RightFootEvents"]
            elif dim_name[0] == "L":
                steps = metadata_dict["LeftFootEvents"]
                
            label_added =False
            for start, end in steps:
                if (end < u_start) | (start > u_end):
                    if not label_added:
                        ax.vlines([start/100, end/100], ymin, ymax, linestyles="--", lw=1, label = "Gait Events")
                        r = ax.fill_between([start/100, end/100], ymin, ymax,
                                        facecolor="green", alpha=0.3, label = "Swing Phases")
                        label_added =True
                    else:
                        ax.vlines([start/100, end/100], ymin, ymax, linestyles="--", lw=1)
                        r = ax.fill_between([start/100, end/100], ymin, ymax,
                                        facecolor="green", alpha=0.3)
        fig.tight_layout()
        fig.legend(bbox_to_anchor=(1, 0.72, 0, 0.5))
        plt.savefig(dim_name + ".svg", dpi=300,
                    transparent=True, bbox_inches='tight')


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

    freq = args.freq
    age = args.age

    # load data (only lower back in this demo)
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0))
    seg_lim = import_data.get_seg(os.path.join(data_WD, args.i1))
    steps_lim = import_data.get_steps(os.path.join(data_WD, args.i1), seg_lim)
    
    # compute semio values (dictionnaries)
    criteria_names, criteria, parameters = compute_semio_val.compute_semio_val(age, steps_lim, seg_lim, data_lb, freq)
    print("ok charge")
    sys.exit(0)

    # print semiogram values
    print_semio_parameters(parameters)

    print_semio_criteria(criteria)

    # semiogram design
    radar_design.new_radar_superpose({"unique": semio_val}, None, id_exp, output, age)
