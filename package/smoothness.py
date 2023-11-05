import numpy as np


def sparc_rot_XS(data_lb, seg_lim, steps_lim, signal="Gyr", phase="full"):

    start = seg_lim.iloc[0, 0]
    end = seg_lim.iloc[3, 0]

    # data_lb_demi_tour = data_lb
    if phase == "u-turn":
        data = data_lb[
            (data_lb.iloc[:, 0] > seg_lim.iloc[1, 0] / 100) & (data_lb.iloc[:, 0] < seg_lim.iloc[2, 0] / 100)]
    else:
        data = data_lb[(data_lb.iloc[:, 0] > start / 100) & (data_lb.iloc[:, 0] < end / 100)]

    # Sélection des signaux
    sig_X_demi_tour = data[signal + "_X"]
    sig_Y_demi_tour = data[signal + "_Y"]
    sig_Z_demi_tour = data[signal + "_Z"]

    sig_n2_demi_tour = np.sqrt(pow(sig_X_demi_tour, 2) + pow(sig_Y_demi_tour, 2) + pow(sig_Z_demi_tour, 2))

    sal_demi_tour, _, _ = sparc(sig_n2_demi_tour, fs=100)

    return sal_demi_tour


# Here after : Copyright (c) 2015, Sivakumar Balasubramanian <siva82kb@gmail.com>

def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.
    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    Examples
    --------
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5*pow(t, 2))
    sal, _, _ = sparc(move, fs=100.)
    '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    # print(len(movement))
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
    # print("nfft", nfft)

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)


def dimensionless_jerk2(movement, fs, data_type):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5*pow(t, 2))
    dl = dimensionless_jerk(move, fs=100.)
    '%.5f' % dl
    '-335.74684'
    """
    # first ensure the movement type is valid.
    if data_type in ('speed', 'accl', 'jerk'):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(movement))
        dt = 1. / fs
        movement_dur = len(movement) * dt
        # get scaling factor:
        _p = {'speed': 3,
              'accl': 1,
              'jerk': -1}
        p = _p[data_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return - scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'speed', 'accl' or 'jerk'.")))


def log_dimensionless_jerk2(movement, fs, data_type):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5*pow(t, 2))
    ldl = log_dimensionless_jerk(move, fs=100.)
    %.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk2(movement, fs, data_type)))


def dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    dimensionless jerk metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5*pow(t, 2))
    dl = dimensionless_jerk(move, fs=100.)
    '%.5f' % dl
    '-335.74684'
    """
    # first enforce data into an numpy array.
    movement = np.array(movement)

    # calculate the scale factor and jerk.
    movement_peak = max(abs(movement))
    dt = 1. / fs
    movement_dur = len(movement) * dt
    jerk = np.diff(movement, 2) / pow(dt, 2)
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)

    # estimate dj
    return - scale * sum(pow(jerk, 2)) * dt


def log_dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    log dimensionless jerk metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5*pow(t, 2))
    ldl = log_dimensionless_jerk(move, fs=100.)
    '%.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk(movement, fs)))
