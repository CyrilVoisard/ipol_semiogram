import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import sys

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
    parser.add_argument('-i1', metavar='data_rf',
                        help='Time series for the right foot sensor.')
    parser.add_argument('-i2', metavar='data_lf',
                        help='Time series for the right foot sensor.')
    parser.add_argument('-f', metavar='freq',
                        help='Acquistion frequency.')
    parser.add_argument('-a', metavar='age', type=int,
                        help='Age of the subject.')
    parser.add_argument('-mi', metavar='min_z', type=int,
                        help='Minimum for Z-score.')
    parser.add_argument('-ma', metavar='max_z', type=int,
                        help='Maximum for Z-score.')
    args = parser.parse_args()

    # DEBUG: print metadata
    print("ok charge", args.age)    
    sys.exit(0)
