
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy import stats

from package import features as ft


def ihr_avg(seg_lim, steps_lim, s, ml=False):
    # Pour ML, le rapport est inversé !
    ihr_list = []
    for j in range(1, len(steps_lim) - 2):  # on ne prend pas en compte les premier et le dernier pas
        if steps_lim["Foot"].iloc[j] - steps_lim["Foot"].iloc[j + 2] == 0:  # on vérifie que c'est le même pied
            if ft.inside([steps_lim["HS"].iloc[j], steps_lim["HS"].iloc[j + 2]], seg_lim):
                # On est conscient que la détection de pas n'est pas fiable à 100%.
                # On va donc prendre le maximum autour de la détection des pas.
                det = det_max(s, steps_lim["HS"].iloc[j], steps_lim["HS"].iloc[j+2], ml=ml)
                if det != 0:
                    ihr_list.append(det)

    ihr_list = ft.rmoutliers(ihr_list)

    return np.mean(ihr_list), np.std(ihr_list)


def det_max(s, start, end, ml=False):
    # Pour déterminer le maximum du HR au voisinage des points de détection automatique
    det_list = []
    for k in range(30):
        for kk in range(5):
            # On fait varier le signal à son origine
            s_step = s[max(start - 15 + k, 0):end - 15 + k - 2 + kk]
            calcul = ihr(s_step)
            if calcul != 0:
                det_list.append(calcul)
            if ml:
                calcul = 100 - calcul
    if len(det_list) != 0:
        return max(det_list)
    else:
        return 0
        

def ihr(sig):
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
        return 100 * peak_pair_sum / (peak_impair_sum + peak_pair_sum)
        

def DFT(x, fs=100):
    # Function to calculate the discrete Fourier Transform coefficient from the fundamental frequency.
    # https://www.f-legrand.fr/scidoc/docimg/numerique/tfd/periodique2/periodique2.html

    f0 = 100 / len(x)

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
