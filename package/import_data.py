import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import sys


def load_XSens(filename):

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


def import_XSens(path, start=0, end=200, ordre=8, fc=14):
    data = load_XSens(path)

    data["FreeAcc_X"] = data["Acc_X"] - np.mean(data["Acc_X"][start:end])
    data["FreeAcc_Y"] = data["Acc_Y"] - np.mean(data["Acc_Y"][start:end])
    data["FreeAcc_Z"] = data["Acc_Z"] - np.mean(data["Acc_Z"][start:end])

    data = filtre_sig(data, "Acc", ordre, fc)
    data = filtre_sig(data, "FreeAcc", ordre, fc)
    data = filtre_sig(data, "Gyr", ordre, fc)

    return data


def filtre_sig(data, type_sig, ordre, fc):
    data[type_sig + "_X"] = filtre_passe_bas(data[type_sig + "_X"], ordre, fc)
    data[type_sig + "_Y"] = filtre_passe_bas(data[type_sig + "_Y"], ordre, fc)
    data[type_sig + "_Z"] = filtre_passe_bas(data[type_sig + "_Z"], ordre, fc)

    return data


def filtre_passe_bas(sig, ordre=8, fc=14, fe=100):
    # Fréquence d'échantillonnage fe en Hz
    # Fréquence de nyquist
    f_nyq = fe / 2.  # Hz

    # Préparation du filtre de Butterworth en passe-bas
    (b, a) = butter(N=ordre, Wn=(fc / f_nyq), btype='low', analog=False)

    # Application du filtre
    return filtfilt(b, a, sig)