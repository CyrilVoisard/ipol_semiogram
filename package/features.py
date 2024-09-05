import numpy as np
from scipy import stats

from package import hr, smoothness


# --------------------------------------
# Average speed: refers to velocity during the walk.

def avg_speed(seg_lim, freq, distance = 20, release_u_turn=False):
    """Compute the average speed of the trial. 
    Eq. 3 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        freq {int} -- acquisition frequency
        distance {int} -- walked distance (m)
        release_u_turn {bool} -- take into account the u_turn phase
    
    Returns
    -------
    float
        Average speed (m/s)
    """
    
    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    if release_u_turn:
        time = (end - start - (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0])) / freq
    else:
        time = (end - start) / freq

    return distance / time


# --------------------------------------
# Springiness: refers to gait rhythmicity.

def stride_time(seg_lim, steps_lim, freq):
    """Compute the average stride time : Time between consecutive initial contact (IC) of the same foot, averaged across all strides 
    within the trial except during the U-turn. 
    Eq. 4 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        Average stride duration (s)
    """

    t = get_stride_list(seg_lim, steps_lim)

    return np.mean(t) / freq


def u_turn_time(seg_lim, freq):
    """Compute the u_turn time : duration of the turn. Can be detected with the angular velocity around the cranio-caudal axis derived from
    the IMU positioned on the lower back. 
    Eq. 5 in the IPOL article. 

    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        U-turn duration (s)
    """
    
    d = (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / freq

    return d


# --------------------------------------
# Smoothness: refers to gait continuousness or non-intermittency.

def sparc_gyr(data_lb, seg_lim, freq):
    """Compute the gyration SPARC. 
    Eq. 6 in the IPOL article. 

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        SPARC gyration
    """
    
    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    # data exclusion 
    data = data_lb[(data_lb.iloc[:, 0] > start) & (data_lb.iloc[:, 0] < end)]

    # signals selection
    sig_X = data["Gyr_X"]
    sig_Y = data["Gyr_Y"]
    sig_Z = data["Gyr_Z"]

    sig_n2= np.sqrt(pow(sig_X, 2) + pow(sig_Y, 2) + pow(sig_Z, 2))

    sal_, _, _ = smoothness.sparc(sig_n2, fs=freq)
    
    return sal_


def ldlj_acc(data_lb, seg_lim, freq, signal='FreeAcc'):
    """Compute the log dimensionless jerk computed from linear acceleration. 
    Eq. 7 in the IPOL article. 

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        freq {int} -- acquisition frequency
        signal {str} -- 'FreeAcc' or 'Acc'

    Returns
    -------
    float
        LDLJ acc
    """
    
    if signal in ["FreeAcc", "Acc"]:
        data_type = "accl"

    else:
        raise ValueError(
            '\n'.join(("The argument signal must be either 'FreeAcc' or 'Acc', otherwise use ldljv_rot_XS.")))

    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    # without u-turn
    data_lb_go, data_lb_back = sig_go_back(data_lb, seg_lim)

    # go phase
    sig_X_go = data_lb_go[signal + "_X"]
    sig_Y_go = data_lb_go[signal + "_Y"]
    sig_Z_go = data_lb_go[signal + "_Z"]

    # back phase
    sig_X_back = data_lb_back[signal + "_X"]
    sig_Y_back = data_lb_back[signal + "_Y"]
    sig_Z_back = data_lb_back[signal + "_Z"]

    sig_n2_go = np.sqrt(pow(sig_X_go, 2) + pow(sig_Y_go, 2) + pow(sig_Z_go, 2))
    sig_n2_back = np.sqrt(pow(sig_X_back, 2) + pow(sig_Y_back, 2) + pow(sig_Z_back, 2))

    ldl_acc_go = smoothness.log_dimensionless_jerk2(sig_n2_go, fs=freq, data_type=data_type)
    ldl_acc_back = smoothness.log_dimensionless_jerk2(sig_n2_back, fs=freq, data_type=data_type)

    return (ldl_acc_go + ldl_acc_back) / 2

# --------------------------------------
# Steadiness: refers to gait regularity.

def variation_coeff_stride_time(seg_lim, steps_lim):
    """Compute the variation coefficient of stride time: standard deviation of the vector of stride times divided by its average. 
    Eq. 8 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        CVstrT (%)
    """
    
    t = get_stride_list(seg_lim, steps_lim)

    return 100 * np.std(t) / np.mean(t)


def variation_coeff_double_stance_time(seg_lim, steps_lim):
    """Compute the variation coefficient of double stance time: Standard deviation of the vector of double stance times divided by its average.
    Eq. 9 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        CVdstT (%)
    """
    
    dst_t = get_double_stance_time_list(seg_lim, steps_lim)
    
    return 100 * np.std(dst_t) / np.mean(dst_t)


def p1_acc(data_lb, seg_lim, steps_lim, freq):
    """Compute the cranio-caudal step autocorrelation coefficient: first peak of the cranio-caudal correlation coefficient of the lower back.
    Eq. 10 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        P1_aCC
    """
    
    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_lb, seg_lim, steps_lim, freq=freq)
    p1 = max(p1_go, p1_back)
    
    return p1


def p2_acc(data_lb, seg_lim, steps_lim, freq):
    """Compute the cranio-caudal stride autocorrelation coefficient: second peak of the cranio-caudal correlation coefficient of the lower back.
    Eq. 11 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        P2_aCC
    """
    
    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_lb, seg_lim, steps_lim, freq=freq)
    p2 = max(p2_go, p2_back)
    
    return p2


# --------------------------------------
# Sturdiness: refers to gait amplitude.

def step_length(seg_lim, steps_lim, distance = 20):
    """Compute the average step length: total length divided by the total number of steps after exclusion of the U-turn.
    Eq. 12 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        distance {int} -- walked distance (m)

    Returns
    -------
    float
        SteL (m)
    """

    n_tot = 0
    for i in range(len(steps_lim)):  # without the first and the last steps
        if inside([steps_lim["HS"][i], steps_lim["TO"][i]], seg_lim): # without u-turn
            n_tot = n_tot + 1

    return distance / n_tot


# --------------------------------------
# Stability: refers to gait balance.

def medio_lateral_root_mean_square(data_lb, seg_lim):
    """Compute the dispersion of the medio-lateral acceleration data relative to zero.
    Eq. 13 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 

    Returns
    -------
    float
        RMSml (m/s2)
    """

    # without u-turn
    data_lb_go, data_lb_back = sig_go_back(data_lb, seg_lim, signal="FreeAcc_Y", norm=True)

    signal_go = rmoutliers(data_lb_go.values.tolist(), z_lim=3)
    signal_back = rmoutliers(data_lb_back.values.tolist(), z_lim=3)

    rms_go = np.sqrt(np.mean(np.square(signal_go)))
    rms_back = np.sqrt(np.mean(np.square(signal_back)))

    rms = min(rms_go, rms_back)

    return rms


# --------------------------------------
# Symmetry: refers to inter-limb coordination during gait.

def p1_p2_acc(data_lb, seg_lim, steps_lim, freq):
    """Compute the ratio of the first (P1) to the second (P2) peak of the craniocaudal correlation coefficient of the lower back (P1P2aCC).
    Eq. 14 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        P1P2_aCC
    """

    p1_go, p1_back, p2_go, p2_back = get_p1_p2_autocorr(data_lb, seg_lim, steps_lim, freq=freq)

    rapp = find_nearest(np.array([p1_go / p2_go, p1_back / p2_back]), 1)

    return (1 - abs(rapp - 1))


def mean_swing_times_ratio(seg_lim, steps_lim, freq):
    """Compute the ratio of the maximum (right or left) of averaged swing time divided by the minimum (right or left) of 
    averaged swing time. 
    Eq. 15 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    float
        swTr
    """
    
    sw_center = [0, 0]
    for foot in [0, 1]:
        steps_lim_f = steps_lim[steps_lim["Foot"] == foot]
        if steps_lim_f["HS"].iloc[0] > steps_lim_f["TO"].iloc[0]:
            sw_center[foot] = 1

    msw = [[], []]

    for foot in [0, 1]:
        steps_lim_f = steps_lim[steps_lim["Foot"] == foot]
        hs_f = steps_lim_f["HS"].tolist()
        to_f = steps_lim_f["TO"].tolist()
        for i in range(sw_center[foot], len(hs_f) - 1):   # without the first and the last steps
            if sw_center[foot]:
                if inside([hs_f[i], to_f[i]], seg_lim):  # without u-turn
                    msw[foot].append(hs_f[i] - to_f[i])
            else:
                if inside([hs_f[i + 1], to_f[i]], seg_lim):  # without u-turn
                    msw[foot].append(hs_f[i + 1] - to_f[i])

    msw_lf = np.mean(rmoutliers(msw[0]))
    msw_rf = np.mean(rmoutliers(msw[1]))

    m1 = max(msw_rf, msw_lf)
    m2 = min(msw_rf, msw_lf)

    return m2/m1


def antero_posterior_iHR(data_lb, seg_lim, steps_lim):
    """Compute the power ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the anteroposterior
    acceleration signal from the trunk.
    Eq. 16 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        iHR_aAP (%)
    """

    s = data_lb["FreeAcc_Z"]  # anteroposterior acceleration
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.ihr_avg(seg_lim, steps_lim, s, ml=False)

    return ihr_s


def medio_lateral_iHR(data_lb, seg_lim, steps_lim):
    """Compute the power ratio of the first 10 unpaired harmonics to the first 10 paired harmonics of the mediolateral
    acceleration signal from the trunk.
    Eq. 17 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        iHR_aML (%)
    """
    
    s = data_lb["FreeAcc_Y"]  # mediolateral acceleration
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.ihr_avg(seg_lim, steps_lim, s, ml=True)


    return ihr_s


def cranio_caudal_iHR(data_lb, seg_lim, steps_lim) -> object:
    """Compute the power ratio of the first 10 paired harmonics to the first 10 unpaired harmonics of the craniocaudal
    acceleration signal from the trunk.
    Eq. 18 in the IPOL article. 
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        iHR_aAP (%)
    """
    
    s = data_lb["FreeAcc_X"]  # craniocaudal acceleration
    steps_lim = steps_lim.sort_values(by="HS")

    ihr_s, st_ihr_s = hr.ihr_avg(seg_lim, steps_lim, s, ml=False)

    return ihr_s


# --------------------------------------
# Synchronization: refers to inter-limb coordination during gait.

def double_stance_time(seg_lim, steps_lim):
    """Compute the double stance time ratio : time between IC of one foot and the FC of the contralateral foot divided by the 
    total time of the gait cycle. 
    Eq. 19 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    float
        dstT (%)
    """
    
    dst_t = get_double_stance_time_list(seg_lim, steps_lim)
    dst_t = rmoutliers(dst_t)

    return np.mean(dst_t)*100


# ---------------------------- Useful functions ----------------------------

def rmoutliers(v_in, z_lim=2):
    """Remove outliers from a vector.
    Alg. 1 in the IPOL article. 
    
    Arguments:
        v_in {vector} -- vector
        z_lim {float} -- z-score to be considered as outlier. default = 2. 

    Returns
    -------
    vector
        v_out : new vector without outliers
    """
    
    z_v = np.abs(stats.zscore(v_in))
    v_out = []
    for i in range(len(v_in)):
        if z_v[i] <= z_lim:
            v_out.append(v_in[i])

    return v_out


def inside(list, seg_lim):
    """Determine if a list of 2 events (corresponding to the HS and TO of a stride) is located within the bounds of the outbound and return phases of the trial.
    The function works correctly only if list has 2 elements.
    
    Arguments:
        list {list} -- list of 2 events (integers)
        seg_lim {dataframe} -- pandas dataframe with the boundaries of the trial

    Returns
    -------
    bool
    """

    if len(list) != 2:
        return False
        
    out = 0
    in_go = 0
    in_back = 0
    for x in list:
        if x < seg_lim.iloc[0, 0]:
            out = 1
        else:
            if seg_lim.iloc[1, 0] < x < seg_lim.iloc[2, 0]:
                out = 1
            else:
                if x > seg_lim.iloc[3, 0]:
                    out = 1
                else:
                    if seg_lim.iloc[0, 0] <= x <= seg_lim.iloc[1, 0]:
                        in_go = 1
                    else:
                        if seg_lim.iloc[2, 0] <= x <= seg_lim.iloc[3, 0]:
                            in_back = 1

    if out + in_go + in_back == 0: 
        return False
    else:
        if out == 1:
            return False
        if in_go + in_back == 2:
            return False
        return True


def find_nearest(array, value):
    """Find the nearest value of a given value in an array.
    
    Arguments:
        array {numpy array}
        value {float}

    Returns
    -------
    float
    """
    
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]


def sig_go_back(data_lb, seg_lim, signal="none", norm=False):
    """Extract the parts of a signal that correspond to the straight-line phases (forward and return) of the trial.
    
    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        signal {str} -- optionnal, name of the column to be extracted if there is one
        norm {bool} -- optionnal, only if signal is specified, True if normalization is to be applied

    Returns
    -------
    Pandas DataFrame 
    """
    
    data_lb_go = data_lb[(data_lb["PacketCounter"] > seg_lim.iloc[0, 0]) & (data_lb["PacketCounter"] < seg_lim.iloc[1, 0])]
    data_lb_back = data_lb[(data_lb["PacketCounter"] > seg_lim.iloc[2, 0]) & (data_lb["PacketCounter"] < seg_lim.iloc[3, 0])]
    if signal == "none":
        return data_lb_go, data_lb_back
    else:
        try:
            data_lb_go = data_lb_go[signal]
            data_lb_back = data_lb_back[signal]
            if norm:
                data_lb_go = data_lb_go - np.mean(data_lb_go)
                data_lb_go = data_lb_back - np.mean(data_lb_back)
            return data_lb_go, data_lb_back
        except:
            print("We did not find: ", signal)
            return data_lb_go, data_lb_back


# ---------------------------- Support functions for general features ----------------------------

def get_stride_list(seg_lim, steps_lim):
    """Compute the list of stride times: time between consecutive initial contact (HS for heel strike) of the same foot.
    Eq. 1 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    t : list
        Contains stride duration (in samples) without outliers
    """
    
    t = []

    steps_lim = steps_lim.sort_values(by="HS")
    for i in range(0, len(steps_lim) - 3):
        if steps_lim["Foot"].iloc[i] != steps_lim["Foot"].iloc[i+1] != steps_lim["Foot"].iloc[i+2]: # test foot alternation
            if inside([steps_lim["HS"].iloc[i], steps_lim["HS"].iloc[i + 2]], seg_lim):
                t_tot = steps_lim["HS"].iloc[i + 2] - steps_lim["HS"].iloc[i]
                t.append(t_tot)

    t = rmoutliers(t)  # remove outliers

    return t


def get_double_stance_time_list(seg_lim, steps_lim):
    """Compute the list of double stance time ratio: time between IC of one foot and the FC of the contralateral foot divided by the 
    total time of the gait cycle.
    Eq. 2 in the IPOL article. 
    
    Arguments:
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events

    Returns
    -------
    dst_t : list
        list of double stance time ratio (%) without outliers 
    """

    dst_t = []

    steps_lim = steps_lim.sort_values(by="HS")
    for i in range(0, len(steps_lim) - 3):
        st1 = (steps_lim["TO"].iloc[i + 2] - steps_lim["HS"].iloc[i + 1])
        st2 = (steps_lim["TO"].iloc[i + 1] - steps_lim["HS"].iloc[i])
        t_tot = steps_lim["HS"].iloc[i + 2] - steps_lim["HS"].iloc[i]
        st = (st1 + st2) / t_tot
        if (steps_lim["Foot"].iloc[i] != steps_lim["Foot"].iloc[i+1] != steps_lim["Foot"].iloc[i+2]) & (st1 > 0) & (st2 > 0): # test foot alternation

            if inside([steps_lim["HS"].iloc[i], steps_lim["HS"].iloc[i + 2]], seg_lim):
                dst_t.append(st)

    dst_t = rmoutliers(dst_t)  # remove outliers

    return dst_t



# ---------------------------- autocorrelation function ----------------------------
def get_p1_p2_autocorr(data_lb, seg_lim, steps_lim, freq):
    """Get cranio-caudal step (P1) and stride (P2) autocorrelation coefficient = first and second peak of the cranio-caudal acceleration autocorrelation coefficient of the lower back.

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    (p1_go, p1_back, p2_go, p2_back) : tuple
        p1 and p2 for go and back phases 
    """
    
    # consider separatly go and back phases, without uturn phase
    sig_go, sig_back = sig_go_back(data_lb, seg_lim, signal="FreeAcc_X", norm=False)
    go_coeff = autocorr(sig_go)

    p1_go, p2_go = peaks_3(go_coeff, data_lb, seg_lim, steps_lim, freq)

    back_coeff = autocorr(sig_back)
    p1_back, p2_back = peaks_3(back_coeff, data_lb, seg_lim, steps_lim, freq)

    return p1_go, p1_back, p2_go, p2_back
    

def peaks_3(vector, data_lb, seg_lim, steps_lim, freq):
    """Select the best peak detection method between peak_1 and peak_2 in order to find autocorrelation peaks corresponding to P1 and P2.
    
    Arguments:
        vector {numpy array} -- numpy array corresponding to the craniocaudal acceleration autocorrelation indicator
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    (p1, p2_go) : tuple 
        p1 and p2 estimation from vector
    """
    
    p1_m1, p2_m1 = peaks_1(vector, data_lb, seg_lim, steps_lim, freq)
    p1_m2, p2_m2 = peaks_2(vector, data_lb, seg_lim, steps_lim, freq)

    return max(p1_m1, p1_m2), max(p2_m1, p2_m2)


def peaks_2(vector, data_lb, seg_lim, steps_lim, freq):
    """Second peak detection method to find autocorrelation peaks corresponding to P1 and P2.
    Detection of maximum autocorrelation greater than 0.3 and a frame around the mean stride duration (P2) and half the mean stride duration (P1).
    Eq. 10 and 11 in the IPOL article. 
    
    Arguments:
        vector {numpy array} -- numpy array corresponding to the craniocaudal acceleration autocorrelation indicator
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    (p1, p2) : tuple 
        p1 and p2 estimation from vector
    """
    
    strT = int(stride_time(seg_lim, steps_lim, freq=freq) * freq)

    start_p1 = int(strT * 0.35)
    end_p1 = int(strT * 0.65)

    start_p2 = int(strT * 0.85)
    end_p2 = int(strT * 1.15)

    p1 = max(vector[start_p1:end_p1])
    p1_index = start_p1 + np.argmax(vector[start_p1:end_p1])

    p2 = max(vector[start_p2:end_p2])
    p2_index = start_p2 + np.argmax(vector[start_p2:end_p2])

    return p1, p2


def peaks_1(vector, data_lb, seg_lim, steps_lim, freq):
    """ First peak detection method to find autocorrelation peaks corresponding to P1 and P2.
    Detection of all autocorrelation maxima greater than 0.3, then selection of those closest to the average stride duration (P2) and half the average stride duration (P1). 
    
    Arguments:
        vector {numpy array} -- numpy array corresponding to the craniocaudal acceleration autocorrelation indicator
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        seg_lim {dataframe} -- pandas dataframe with phases events 
        steps_lim {dataframe} -- pandas dataframe with gait events
        freq {int} -- acquisition frequency

    Returns
    -------
    (p1, p2) : tuple 
        p1 and p2 estimation from vector
    """
    
    strT = int(stride_time(seg_lim, steps_lim, freq=freq) * freq)

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

    Finds the numeric index of the peaks in *y* by taking its first-order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Arguments
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres: float between [0., 1.]
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
    peaks : ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first-order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        print("Signal seems to be flat !")
        return np.array([])

    if len(zeros):
        # compute first-order difference of zero indexes
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
            # set leftmost values to leftmost non-zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non-zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first-order difference
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
    """Autocorrelation estimation. Eq. 10 and Appendix B in the IPOL article. 
    Appendix 

    Arguments
    ----------
    f : ndarray
        1D amplitude data to compute autocorrelation.

    Returns
    -------
    acf : ndarray
        Array containing non biased estimator for autocorrelation
    """
    
    N = len(f)
    fvi = np.fft.fft(f, n=2 * N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    d = N - np.arange(N)
    acf = acf / d  # non biased estimation
    acf = acf / acf[0]
    return acf

