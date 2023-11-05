import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import sys


# XSens data 

def load_XSens(filename):
    """Load the data from a file.

    Arguments:
        filename {str} -- File path

    Returns
    -------
    Pandas dataframe
        signal
    """
    
    signal = pd.read_csv(filename, delimiter="\t", skiprows=1, header=0)
    t = signal["PacketCounter"]
    t_0 = t[0]
    t_fin = t[len(t) - 1]

    time = [i for i in range(int(t_0), int(t_fin) + 1)]
    time_init_0 = [i / 100 for i in range(len(time))]
    d = {'PacketCounter': time_init_0}

    colonnes = signal.columns

    for colonne in colonnes[1:]:
        val = signal[colonne]
        f = interpolate.interp1d(t, val)
        y = f(time)
        d[colonne] = y.tolist()

    signal = pd.DataFrame(data=d)

    return signal


def import_XSens(path, start=0, end=200, order=8, fc=14):
    """Import and pre-process the data from a file.

    Arguments:
        filename {str} -- file path
        start {int} -- start of the calibration period
        end {int} -- end of the calibration period
        order {int} -- order of the Butterworth low-pass filter
        fc {int} -- cut-off frequency of the Butterworth low-pass filter

    Returns
    -------
    Pandas dataframe
        data
    """
    
    data = load_XSens(path)
    
    data["FreeAcc_X"] = data["Acc_X"] - np.mean(data["Acc_X"][start:end])
    data["FreeAcc_Y"] = data["Acc_Y"] - np.mean(data["Acc_Y"][start:end])
    data["FreeAcc_Z"] = data["Acc_Z"] - np.mean(data["Acc_Z"][start:end])

    data = filter_sig(data, "Acc", order, fc)
    data = filter_sig(data, "FreeAcc", order, fc)
    data = filter_sig(data, "Gyr", order, fc)

    return data


def filter_sig(data, type_sig, order, fc):
    """Application of Butterworth low-pass filter to a Dataframe

    Arguments:
        data {dataframe} -- pandas dataframe
        type_sig {str} -- "Acc", "Gyr" or "Mag"
        order {int} -- order of the Butterworth low-pass filter
        fc {int} -- cut-off frequency of the Butterworth low-pass filter

    Returns
    -------
    Pandas dataframe
        data
    """
    data[type_sig + "_X"] = low_pass_filter(data[type_sig + "_X"], order, fc)
    data[type_sig + "_Y"] = low_pass_filter(data[type_sig + "_Y"], order, fc)
    data[type_sig + "_Z"] = low_pass_filter(data[type_sig + "_Z"], order, fc)

    return data


def low_pass_filter(sig, order=8, fc=14, fe=100):
    """Definition of a Butterworth low-pass filter

    Arguments:
        sig {dataframe} -- pandas dataframe
        order {int} -- order of the Butterworth low-pass filter
        fc {int} -- cut-off frequency of the Butterworth low-pass filter
        fe {int} -- acquisition frequency for the data
    Returns
    -------
    ndarray
        filter
    """
    
    f_nyq = fe / 2.  # Hz

    # definition of the Butterworth low-pass filter
    (b, a) = butter(N=order, Wn=(fc / f_nyq), btype='low', analog=False)

    # application
    return filtfilt(b, a, sig)


# steps
def get_steps(filename, seg_lim): 
    """Gets the gait events segmentation from the metadata JSON file. 

    Arguments:
        filename {str} -- File path
        
    Returns
    -------
    Pandas dataframe
        steps_lim
    """
    with open(filename) as steps_file:
        steps_dict = json.load(steps_file)
    steps_left = steps_dict["LeftFootEvents"]
    steps_right = steps_dict["RightFootEvents"]

    foot = []
    to = []
    hs = []

    for i in range(len(steps_right)):
        if inside(steps_right[i][0], steps_right[i][1], seg_lim):
            foot.append(1)
            to.append(steps_right[i][0])
            hs.append(steps_right[i][1])

    for i in range(len(steps_left)):
        if inside(steps_left[i][0], steps_left[i][1], seg_lim):
            foot.append(0)
            to.append(steps_left[i][0])
            hs.append(steps_left[i][1])

    _dict = {'Foot': foot, 'TO': to, 'HS': hs}
    type_dict = {'Foot': int, 'HS': int, 'TO': int}
    steps_lim = pd.DataFrame(_dict)
    steps_lim = steps_lim.astype(type_dict)
    
    return steps_lim
    

# phases
def get_seg(filename): 
    """Gets the gait events segmentation from the metadata JSON file. 

    Arguments:
        filename {str} -- File path
        
    Returns
    -------
    Pandas dataframe
        seg_lim
    """
    with open(filename) as seg_file:
        seg_dict = json.load(seg_file)
    [start_u, end_u] = seg_dict["UTurnBoundaries"]
    start = min(np.min(seg_dict["LeftFootEvents"]), np.min(seg_dict["RightFootEvents"]))
    end = max(np.max(seg_dict["LeftFootEvents"]), np.max(seg_dict["RightFootEvents"]))
    return pd.DataFrame([start, start_u, end_u, end])
    

def inside(to, hs, seg_lim):
    """Check whether the considered step is valid, meaning it is not included in the half-turn.

    Arguments:
        to {int} -- Toe-Off time
        hs {int} -- Heel-Strike time
        seg_lim {panda DataFrame} -- boundaries of the trial
        
    Returns
    -------
    Boolean
        in_
    """
    seg_lim = pd.DataFrame(seg_lim)
    if (hs <= seg_lim.iloc[1, 0]) & (to <= seg_lim.iloc[1, 0]):
            in_ = True
    else:
        if (hs >= seg_lim.iloc[2, 0]) & (to >= seg_lim.iloc[2, 0]):
            in_ = True
        else:
            in_ = False
    return in_
