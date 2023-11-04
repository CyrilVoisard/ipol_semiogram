# Objectif : détailler l'ensemble des 14 features conservées dans la version finale du semiogram.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy import stats

# os.chdir('/Users/cyril/Library/Mobile Documents/com~apple~CloudDocs/Borelli/4 - Classes et fonctions')
from package import features as ft


def DFT(x, fs=100):
    # Function to calculate the discrete Fourier Transform coefficient à partir de la fréquence fondamentale
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


def hr(sig):
    peak_list = DFT(sig)

    peak_pair_sum = sum(peak_list[2:21:2])
    peak_impair_sum = sum(peak_list[1:21:2])

    if (peak_impair_sum == 0) | (peak_impair_sum == 0):
        return 0
    else:
        return peak_pair_sum / peak_impair_sum


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


def det_max(s, start, end, i):
    # Pour déterminer le maximum du HR au voisinage des points de détection automatique
    det_list = []
    for k in range(30):
        for kk in range(5):
            # On fait varier le signal à son origine
            s_step = s[max(start - 15 + k, 0):end - 15 + k - 2 + kk]
            if i == 0:
                calcul = hr(s_step)
            else:
                calcul = ihr(s_step)
            if calcul != 0:
                det_list.append(calcul)
            else:
                print("Voici l'erreur :")
                plt.plot(s_step)
                plt.show()
    if len(det_list) != 0:
        if (det_list[0] == max(det_list)) & (det_list[-1] == max(det_list)):
            print("Attention valeur extrême !")
        return max(det_list)
    else:
        return 0


def det_max_ml(s, start, end, i):
    # Pour déterminer le maximum du HR au voisinage des points de détection automatique
    det_list = []
    for k in range(30):
        for kk in range(5):
            # On fait varier le signal à son origine
            s_step = s[max(start - 15 + k, 0):end - 15 + k - 2 + kk]
            if i == 0:
                calcul = 1 / hr(s_step)
            else:
                calcul = 100 - ihr(s_step)
            if calcul != 0:
                det_list.append(calcul)
            else:
                print("Voici l'erreur :")
                plt.plot(s_step)
                plt.show()
    if len(det_list) != 0:
        if (det_list[0] == max(det_list)) & (det_list[-1] == max(det_list)):
            print("Attention valeur extrême !")
        return max(det_list)
    else:
        return 0


def hr_ihr_moyen(seg_lim, steps_lim, s, i=1, ml=False):
    # Pour ML, le rapport est inversé !
    # Le i=1 signifie qu'on considère IHR et non pas HR
    # Le i=0 signifie qu'on considère HR et non pas IHR
    hr_ihr_list = []
    for j in range(1, len(steps_lim) - 2):  # on ne prend pas en compte les premier et le dernier pas
        if steps_lim.iloc[j, 0] - steps_lim.iloc[j + 2, 0] == 0:  # on vérifie que c'est le même pied
            if ft.inside([steps_lim.iloc[j, 4], steps_lim.iloc[j + 2, 4]], seg_lim):
                # On est conscient que la détection de pas n'est pas fiable à 100%.
                # On va donc prendre le maximum autour de la détection des pas.
                if not ml:
                    det = det_max(s, steps_lim.iloc[j, 4], steps_lim.iloc[j+2, 4], i=i)
                else:
                    det = det_max_ml(s, steps_lim.iloc[j, 4], steps_lim.iloc[j + 2, 4], i=i)
                if det != 0:
                    hr_ihr_list.append(det)

    hr_ihr_list = ft.rmoutliers(hr_ihr_list)

    return np.mean(hr_ihr_list), np.std(hr_ihr_list)
