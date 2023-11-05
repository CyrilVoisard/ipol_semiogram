# Objectif : détailler l'ensemble des 14 features conservées dans la version finale du semiogram.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from package import hr, smoothness


# --------------------------------------
# Average Speed : refers celerity during the walk.

def avg_speed(data_tronc, seg_lim, steps_lim=0, release_u_turn=False, freq=100):
    seg_lim = pd.DataFrame(seg_lim)
    # Hypothèse sur la distance
    distance = 20

    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    if release_u_turn:
        temps = (end - start - (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0])) / freq
    else:
        temps = (end - start) / freq

    return distance / temps


# --------------------------------------
# Springiness : refers to gait rhythmicity. => ok

def stride_time(data_tronc, seg_lim, steps_lim, freq=100):
    # Time between consecutive initial contact (IC) of the same foot, averaged across all strides within the trial
    # except during the U-turn

    t = get_stride_list(data_tronc, seg_lim, steps_lim, freq=freq)
    print("get_stride_list", t)

    return np.mean(t) / freq


def u_turn_time(data_tronc, seg_lim, steps_lim, freq=100):
    # Duration of the turn. Detected with the angular velocity around the cranio-caudal axis derived from
    # the IMU positioned on the lower back.
    d = (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / freq

    return d


# --------------------------------------
# Smoothness : refers to gait continuousness or non-intermittency. => non ok

def antero_posterieur_root_mean_square(data_tronc, seg_lim, steps_lim, freq=100):
    # Measure of dispersion of the medio-lateral acceleration data relative to zero.

    # On enlève le demi-tour
    data_tronc_go, data_tronc_back = sig_go_back(data_tronc, seg_lim, freq=freq, signal="FreeAcc_Z", norm=True)

    signal_go = rmoutliers(data_tronc_go.values.tolist(), limite=3)
    signal_back = rmoutliers(data_tronc_back.values.tolist(), limite=3)

    rms_go = np.sqrt(np.mean(np.square(signal_go)))
    rms_back = np.sqrt(np.mean(np.square(signal_back)))

    # rms = rms_go / 2 + rms_back / 2
    rms = min(rms_go, rms_back)

    return rms


def sparc_gyr(data_tronc, seg_lim, steps_lim, freq=100):
    return smoothness.sparc_rot_XS(data_tronc, seg_lim, steps_lim, signal="Gyr")


def ldlj_acc(data_tronc, seg_lim, steps_lim, signal='FreeAcc', freq=100):
    if signal in ["FreeAcc", "Acc"]:
        data_type = "accl"

    else:
        raise ValueError(
            '\n'.join(("The argument signal must be either 'FreeAcc' or 'Acc', otherwise use ldljv_rot_XS.")))

    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    # On enlève le demi-tour
    data_tronc_go, data_tronc_back = sig_go_back(data_tronc, seg_lim, freq=freq)

    # Pour l'aller
    sig_X_go = data_tronc_go[signal + "_X"]
    sig_Y_go = data_tronc_go[signal + "_Y"]
    sig_Z_go = data_tronc_go[signal + "_Z"]

    # Pour l'aller
    sig_X_back = data_tronc_back[signal + "_X"]
    sig_Y_back = data_tronc_back[signal + "_Y"]
    sig_Z_back = data_tronc_back[signal + "_Z"]

    sig_n2_go = np.sqrt(pow(sig_X_go, 2) + pow(sig_Y_go, 2) + pow(sig_Z_go, 2))
    sig_n2_back = np.sqrt(pow(sig_X_back, 2) + pow(sig_Y_back, 2) + pow(sig_Z_back, 2))

    ldl_acc_go = smoothness.log_dimensionless_jerk2(sig_n2_go, fs=freq, data_type=data_type)
    ldl_acc_back = smoothness.log_dimensionless_jerk2(sig_n2_back, fs=freq, data_type=data_type)

    return (ldl_acc_go + ldl_acc_back) / 2


def ldlj_gyr(data_tronc, seg_lim, steps_lim, signal="Gyr", phase="full", freq=100):
    if signal in ["Gyr"]:
        data_type = "speed"
    else:
        raise ValueError('\n'.join(("The argument signal must be 'Gyr', otherwise use ldlja_lin_XS.")))

    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    # data_tronc_demi_tour = data_tronc
    if phase == "u-turn":
        data = data_tronc[
            (data_tronc.iloc[:, 0] > seg_lim.iloc[1, 0] / freq) & (data_tronc.iloc[:, 0] < seg_lim.iloc[2, 0] / freq)]
    else:
        data = data_tronc[(data_tronc.iloc[:, 0] > start / freq) & (data_tronc.iloc[:, 0] < end / freq)]

    # Sélection des signaux
    sig_X_demi_tour = data[signal + "_X"]
    sig_Y_demi_tour = data[signal + "_Y"]
    sig_Z_demi_tour = data[signal + "_Z"]

    sig_n2_demi_tour = np.sqrt(pow(sig_X_demi_tour, 2) + pow(sig_Y_demi_tour, 2) + pow(sig_Z_demi_tour, 2))

    ldl_vit_demi_tour = smoothness.log_dimensionless_jerk2(sig_n2_demi_tour, fs=freq, data_type=data_type)

    return ldl_vit_demi_tour


# --------------------------------------
# Steadiness : refers to gait regularity. => ok

def variation_coeff_stride_time(data_tronc, seg_lim, steps_lim, freq=100):
    # Standard deviation of the vector of stride times divided by its average.
    t = get_stride_list(data_tronc, seg_lim, steps_lim, freq=freq)

    return 100 * np.std(t) / np.mean(t)


def variation_coeff_double_stance_time(data_tronc, seg_lim, steps_lim, freq=100):
    # Standard deviation of the vector of double stance times divided by its average.
    dst_t = get_double_stance_time_list(data_tronc, seg_lim, steps_lim, freq=freq)
    return 100 * np.std(dst_t) / np.mean(dst_t)


def p1_acc(data_tronc, seg_lim, steps_lim, freq=100):
    # Cranio-caudal step autocorrelation coefficient = first peak of the cranio-caudal correlation coefficient of
    # the lower back.
    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_tronc, seg_lim, steps_lim, freq=freq)
    p1 = max(p1_go, p1_back)
    # p1 = (p1_go + p1_back) / 2
    return p1


def p2_acc(data_tronc, seg_lim, steps_lim, freq=100):
    # Cranio-caudal step autocorrelation coefficient = first peak of the cranio-caudal correlation coefficient of
    # the lower back.
    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_tronc, seg_lim, steps_lim, freq=freq)
    p2 = max(p2_go, p2_back)
    # p2 = (p2_go + p2_back) / 2
    return p2


# --------------------------------------
# Sturdiness : refers to gait amplitude. => ok

def step_length(data_tronc, seg_lim, steps_lim, freq=100):
    # Total length (20 m) divided by the total number of steps after exclusion of the U-turn.
    n_tot = 0
    for i in range(len(steps_lim)):  # on ne prend pas en compte les premiers et le dernier pas
        # if inside([steps_lim["HS"][i]], seg_lim) | inside([steps_lim["TO"][i]], seg_lim):
        if inside([steps_lim["HS"][i], steps_lim["TO"][i]], seg_lim):
            # On ne prend pas en compte les pas du demi-tour
            n_tot = n_tot + 1

    return 20 / n_tot


# --------------------------------------
# Stability : refers to gait balance. => ok

def medio_lateral_root_mean_square(data_tronc, seg_lim, steps_lim, freq=100):
    # Measure of dispersion of the medio-lateral acceleration data relative to zero.

    # On enlève le demi-tour
    data_tronc_go, data_tronc_back = sig_go_back(data_tronc, seg_lim, freq=freq, signal="FreeAcc_Y", norm=True)

    signal_go = rmoutliers(data_tronc_go.values.tolist(), limite=3)
    signal_back = rmoutliers(data_tronc_back.values.tolist(), limite=3)

    rms_go = np.sqrt(np.mean(np.square(signal_go)))
    rms_back = np.sqrt(np.mean(np.square(signal_back)))

    # rms = rms_go / 2 + rms_back / 2
    rms = min(rms_go, rms_back)

    return rms


# --------------------------------------
# Symmetry : refers to inter-limb coordination during gait.

# To sum up, a fast Fourier transform was performed to draw the Fourier series of the signal. The sum of the amplitudes
# of the 10 first even harmonics is divided by the sum of the amplitudes of the 10 first odd harmonics.

def antero_posterior_HR(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the antero-posterior
    # acceleration signal from the trunk.

    s = data_tronc["FreeAcc_Z"]  # acceleration selon z
    steps_lim = steps_lim.sort_values(by="HS")

    hr_s, st_hr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=0, ml=False)

    # print("AAP_HR_single", hr_s, st_hr_s)

    return hr_s


def antero_posterior_iHR(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the antero-posterior
    # acceleration signal from the trunk.

    s = data_tronc["FreeAcc_Z"]  # acceleration selon z
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=1, ml=False)

    # print("AAP_IHR_single", ihr_s, st_ihr_s)

    return ihr_s


def medio_lateral_HR(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the first 10 unpaired harmonics to the first 10 paired harmonics of the medio-lateral
    # acceleration signal from the trunk.
    s = data_tronc["FreeAcc_Y"]  # acceleration selon y
    steps_lim = steps_lim.sort_values(by="HS")

    hr_s, st_hr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=0, ml=True)

    # print("ML_HR_single", hr_s)

    return hr_s


def medio_lateral_iHR(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the first 10 unpaired harmonics to the first 10 paired harmonics of the medio-lateral
    # acceleration signal from the trunk.
    s = data_tronc["FreeAcc_Y"]  # acceleration selon y
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=1, ml=True)

    # print("ML_IHR_single", ihr_s)

    return ihr_s


def cranio_caudal_HR(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the cranio-caudal
    # acceleration signal from the trunk.
    s = data_tronc["FreeAcc_X"]  # acceleration selon x
    steps_lim = steps_lim.sort_values(by="HS")

    hr_s, st_hr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=0, ml=False)

    # print("CC_HR", hr_s, st_hr_s)

    return hr_s


def cranio_caudal_iHR(data_tronc, seg_lim, steps_lim, freq=100) -> object:
    # Ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the cranio-caudal
    # acceleration signal from the trunk.
    s = data_tronc["FreeAcc_X"]  # acceleration selon x
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.hr_ihr_moyen(seg_lim, steps_lim, s, i=1, ml=False)

    # print("CC_IHR_single", ihr_s, st_ihr_s)

    return ihr_s


def p1_p2_acc(data_tronc, seg_lim, steps_lim, freq=100):
    # the ratio of the first (P1) to the second (P2) peak of the cranio-caudal correlation
    # coefficient of the lower back (P1P2aCC)

    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_tronc, seg_lim, steps_lim, freq=freq)

    rapp = find_nearest(np.array([p1_go / p2_go, p1_back / p2_back]), 1)

    return (1 - abs(rapp - 1))


def mean_swing_times_ratio(data_tronc, seg_lim, steps_lim, freq=100):
    # Ratio of the maximum (right or left) of averaged swing time divided by the minimum (right or left) of
    # averaged swing time.

    sw_center = [0, 0]
    for foot in [0, 1]:
        steps_lim_f = steps_lim[steps_lim["Foot"] == foot]
        if steps_lim_f.iloc[0, 4] > steps_lim_f.iloc[0, 3]:
            sw_center[foot] = 1

    msw = [[], []]

    for foot in [0, 1]:
        steps_lim_f = steps_lim[steps_lim["Foot"] == foot]
        hs_f = steps_lim_f["HS"].tolist()
        to_f = steps_lim_f["TO"].tolist()
        for i in range(sw_center[foot], len(hs_f) - 1):  # on ne prend pas en compte les premiers et le dernier pas
            # On ne prend pas en compte les pas du demi-tour
            if sw_center[foot]:
                if inside([hs_f[i], to_f[i]], seg_lim):
                    msw[foot].append(hs_f[i] - to_f[i])
            else:
                if inside([hs_f[i + 1], to_f[i]], seg_lim):
                    msw[foot].append(hs_f[i + 1] - to_f[i])

    msw_lf = np.mean(rmoutliers(msw[0]))
    msw_rf = np.mean(rmoutliers(msw[1]))

    m1 = max(msw_rf, msw_lf)
    m2 = min(msw_rf, msw_lf)

    return m2/m1


# --------------------------------------
# Synchronization : refers to inter-limb coordination during gait. => ok

def double_stance_time(data_tronc, seg_lim, steps_lim, freq=100):
    # Time between IC of one foot and the FC of the contralateral foot divided by the total time of the cycle time.

    dst_t = get_double_stance_time_list(data_tronc, seg_lim, steps_lim, freq=freq)

    if len(dst_t) == 0:
        print("erreur_dst", steps_lim)

    dst_t = rmoutliers(dst_t)

    return np.mean(dst_t)*100


# ---------------------------- Fonctions utiles générales ----------------------------

def rmoutliers(vec, limite=2):
    # print("Avant :", vec)
    z = np.abs(stats.zscore(vec))
    vec1 = []
    for i in range(len(vec)):
        if z[i] < limite:  # La valeur à partir de laquelle on peut supprimer les données peut varier
            vec1.append(vec[i])
    # print("Après :", vec1)
    return vec1


def inside(liste, seg_lim):
    seg_lim = pd.DataFrame(seg_lim)
    out = 0
    in_go = 0
    in_back = 0
    for x in liste:
        if x < seg_lim.iloc[0, 0]:
            out = 1
        else:
            if (x > seg_lim.iloc[1, 0]) and (x < seg_lim.iloc[2, 0]):
                out = 1
            else:
                if x > seg_lim.iloc[3, 0]:
                    out = 1
                else:
                    if (x <= seg_lim.iloc[1, 0]) and (x >= seg_lim.iloc[0, 0]):
                        in_go = 1
                    else:
                        if (x <= seg_lim.iloc[3, 0]) and (x >= seg_lim.iloc[2, 0]):
                            in_go = 1

    if out + in_go + in_back == 0:
        return 0
    else:
        if out == 1:
            return 0
        else:
            if in_go + in_back == 2:
                return 0
            else:
                if in_go + in_back == 1:
                    return 1


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def sig_go_back(data_tronc, seg_lim, freq=100, signal="aucun", norm=False):
    data_tronc_go = data_tronc[(seg_lim.iloc[0, 0] / freq < data_tronc["PacketCounter"])
                               & (data_tronc["PacketCounter"] < seg_lim.iloc[1, 0] / freq)]
    data_tronc_back = data_tronc[(seg_lim.iloc[3, 0] / freq > data_tronc["PacketCounter"])
                                 & (data_tronc["PacketCounter"] > seg_lim.iloc[2, 0] / freq)]
    if signal == "aucun":
        return data_tronc_go, data_tronc_back
    else:
        try:
            data_tronc_go = data_tronc_go[signal]
            data_tronc_back = data_tronc_back[signal]
            if norm:
                data_tronc_go = data_tronc_go - np.mean(data_tronc_go)
                data_tronc_go = data_tronc_back - np.mean(data_tronc_back)
            return data_tronc_go, data_tronc_back
        except:
            print("Pas possible de trouver le signal : ", signal)
            return data_tronc_go, data_tronc_back


# ---------------------------- Fonctions support pour les features ----------------------------

def get_stride_list(data_tronc, seg_lim, steps_lim, freq=100):
    # Get each time between consecutive initial contact (IC) of the same foot, averaged across all strides within
    # the trial except during the U-turn
    t = []

    steps_lim = steps_lim.sort_values(by="HS")
    for i in range(1, len(steps_lim) - 4): # On ne prend pas en compte les 2 derniers pas qui peuvent être freinés
        t_tot = steps_lim["HS"].iloc[i + 2] - steps_lim["HS"].iloc[i]
        if ((steps_lim["Foot"].iloc[i] + steps_lim["Foot"].iloc[i + 1] == 1)
                & (steps_lim["Foot"].iloc[i + 1] + steps_lim["Foot"].iloc[i + 2] == 1)):

            if inside([steps_lim["HS"].iloc[i], steps_lim["HS"].iloc[i + 2]], seg_lim):
                t.append(t_tot)

    t = rmoutliers(t)

    return t


def get_double_stance_time_list(data_tronc, seg_lim, steps_lim, freq=100):
    # Get for each cycle the time between IC of one foot and the FC of the contralateral foot divided by the total
    # time of the cycle time.

    dst_t = []

    steps_lim = steps_lim.sort_values(by="HS")
    for i in range(1, len(steps_lim) - 4):
        st1 = (steps_lim["TO"].iloc[i + 2] - steps_lim["HS"].iloc[i + 1])
        st2 = (steps_lim["TO"].iloc[i + 1] - steps_lim["HS"].iloc[i])
        t_tot = steps_lim["HS"].iloc[i + 2] - steps_lim["HS"].iloc[i]
        st = (st1 + st2) / t_tot
        if ((steps_lim["Foot"].iloc[i] + steps_lim["Foot"].iloc[i + 1] == 1) &
            (steps_lim["Foot"].iloc[i + 1] + steps_lim["Foot"].iloc[i + 2] == 1)) & (st1 > 0) & (st2 > 0):

            if inside([steps_lim["HS"].iloc[i], steps_lim["HS"].iloc[i + 2]], seg_lim):
                dst_t.append(st)

    dst_t = rmoutliers(dst_t)

    return dst_t


def get_p1_p2_autocorr(data_tronc, seg_lim, steps_lim, freq=100):
    # Get cranio-caudal step (P1) and stride (P2) autocorrelation coefficient = first and seconde peak
    # of the cranio-caudal correlation coefficient of the lower back.

    # On enlève le demi-tour, on sépare l'aller et le retour et on sélectionne la colonne d'intérêt
    sig_go, sig_back = sig_go_back(data_tronc, seg_lim, freq=freq, signal="FreeAcc_X", norm=False)
    go_coeff = autocorr(sig_go)

    p1_go, p2_go = peaks_3(go_coeff, data_tronc, seg_lim, steps_lim, freq)

    back_coeff = autocorr(sig_back)
    p1_back, p2_back = peaks_3(back_coeff, data_tronc, seg_lim, steps_lim, freq)

    # print("P1P2", p1_go, p1_back, p2_go, p2_back)
    return p1_go, p1_back, p2_go, p2_back


# ----------------------------Spéciales P1/P2 et autocorrélation ----------------------------
def peaks_3(vector, data_tronc, seg_lim, steps_lim, freq):
    p1_m1, p2_m1 = peaks_1(vector, data_tronc, seg_lim, steps_lim, freq)
    p1_m2, p2_m2 = peaks_2(vector, data_tronc, seg_lim, steps_lim, freq)

    return max(p1_m1, p1_m2), max(p2_m1, p2_m2)


def peaks_2(vector, data_tronc, seg_lim, steps_lim, freq):
    strT = int(stride_time(data_tronc, seg_lim, steps_lim) * freq)

    start_p1 = int(strT * 0.35)
    end_p1 = int(strT * 0.65)

    start_p2 = int(strT * 0.85)
    end_p2 = int(strT * 1.15)

    p1 = max(vector[start_p1:end_p1])
    p1_index = start_p1 + np.argmax(vector[start_p1:end_p1])

    p2 = max(vector[start_p2:end_p2])
    p2_index = start_p2 + np.argmax(vector[start_p2:end_p2])

    return p1, p2


def peaks_1(vector, data_tronc, seg_lim, steps_lim, freq):
    strT = int(stride_time(data_tronc, seg_lim, steps_lim) * freq)

    indexes_pic_go = indexes(vector[0:len(vector)//2], min_dist=strT * 0.35)

    if len(indexes_pic_go) == 0:
        return 0, 0
    else:
        index_p1 = indexes_pic_go[np.argmin(abs(indexes_pic_go - strT / 2))]
        index_p2 = indexes_pic_go[np.argmin(abs(indexes_pic_go - strT))]

        p1 = vector[index_p1]
        p2 = vector[index_p2]

        return p1, p2


def indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)
    # print("dy", dy)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    # print("where", np.where(dy == 0))
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        print("Signal seems to be flat !")
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def autocorr(f):
    N = len(f)
    fvi = np.fft.fft(f, n=2 * N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    d = N - np.arange(N)
    acf = acf / d # pour avoir un indicateur non biaisé
    acf = acf / acf[0]
    return acf

