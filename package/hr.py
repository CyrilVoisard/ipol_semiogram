
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy import stats

from package import features as ft


def ihr_avg(seg_lim, steps_lim, s, ml=False):
    """Compute the mean of the iHR for each of the detected steps during the trial.
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        s {vector} -- time series from the lower back sensor depending on the iHR
        ml {bool} -- the ratio is inverted if it is about the mediolateral signal

    Returns
    -------
    double
        average iHR (float)
        standard deviation iHR (float)
    """
    
    ihr_list = []
    for j in range(1, len(steps_lim) - 2):  # not the first and the last steps
        if steps_lim["Foot"].iloc[j] - steps_lim["Foot"].iloc[j + 2] == 0:  # check it is the same foot
            if ft.inside([steps_lim["HS"].iloc[j], steps_lim["HS"].iloc[j + 2]], seg_lim):
                det = det_max(s, steps_lim["HS"].iloc[j], steps_lim["HS"].iloc[j+2], ml=ml)  # neighborhood maximum 
                if det != 0:
                    ihr_list.append(det)

    ihr_list = ft.rmoutliers(ihr_list)

    return np.mean(ihr_list), np.std(ihr_list)


def det_max(s, start, end, ml=False):
    """Determine the maximum iHR in the neighborhood of the detected stride.
    
    Arguments:
        s {vector} -- time series from the lower back sensor depending on the iHR
        start {int} -- sample starting the stride
        end {int} -- sample ending the stride
        ml {bool} -- the ratio is inverted if it is about the mediolateral signal

    Returns
    -------
    float
        maximum iHR
    """
    
    det_list = []
    for k in range(30):
        for kk in range(5):
            s_step = s[max(start - 15 + k, 0):end - 15 + k - 2 + kk]
            calcul = ihr(s_step, ml)
            if calcul != 0:
                det_list.append(calcul)
            
    if len(det_list) != 0:
        return max(det_list)
    else:
        return 0
        

def ihr(sig, ml):
    """Compute the iHR of a signal
    
    Arguments:
        sig {vector} -- time series
        ml {bool} -- False if anteroposterior/craniocaudal acceleration, True if acceleration mediolateral

    Returns
    -------
    float
        iHR
    """
    
    peak_list = DFT(sig)

    peak_pair_sum = 0
    for peak in peak_list[2:21:2]:
        peak_pair_sum = peak_pair_sum + peak * peak

    peak_impair_sum = 0
    for peak in peak_list[1:21:2]:
        peak_impair_sum = peak_impair_sum + peak * peak

    if (peak_impair_sum == 0) | (peak_impair_sum == 0):
        return 0
    else:
        if ml:
            return 100 * peak_impair_sum / (peak_impair_sum + peak_pair_sum)
        else:
            return 100 * peak_pair_sum / (peak_impair_sum + peak_pair_sum)
        

def DFT(x, fs=100):
    """Compute the discrete Fourier Transform coefficient from the fundamental frequency.
    # https://www.f-legrand.fr/scidoc/docimg/numerique/tfd/periodique2/periodique2.html
    
    Arguments:
        x {vector} -- time series
        fs {int} -- acquisition frequency

    Returns
    -------
    list
        peak_list
    """

    f0 = fs / len(x)

    peak_list = []
    n_f0 = []

    x = np.array(x.to_list())

    N = len(x)
    n = np.arange(21)
    for k in n:
        e = np.exp(-2j * np.pi * f0 * k * np.arange(len(x)) / fs)
        peak = abs(sum(x * e))
        peak_list.append(peak)
        n_f0.append(k)

    return peak_list
