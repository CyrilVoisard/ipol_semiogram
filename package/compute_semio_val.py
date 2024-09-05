import numpy as np
import json
from package import features as ft


def compute_semio_val(distance, steps_lim, seg_lim, data_lb, freq):
    """Compute the Z-score for each criterion from the calculation of the 17 parameters

    Arguments:
        distance {int} -- walked distance in meters 
        steps_lim {dataframe} -- pandas dataframe with gait events
        seg_lim {dataframe} -- pandas dataframe with phases events 
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        freq {int} -- acquisition frequency

    Returns
    -------
    Tuple
        criteria_names {list} -- criteria labels
        semio_val {list} -- z-scores for each criterion
        parameters {list} -- values and z-scores for each parameter
    """
    # criteria list
    criteria_names = ['Springiness', 'Sturdiness', 'Smoothness', 'Steadiness', 'Stability', 'Symmetry', 'Synchronisation', 'Average Speed']
    
    parameters = []

    # find the model
    path = 'models/reference.json'
    with open(path) as json_data:
        norms = np.array(json.load(json_data))

    # springiness
    spr_feat = [ft.stride_time(seg_lim, steps_lim, freq=freq),
                ft.u_turn_time(seg_lim, freq=freq)]
    spr_ref = [norms[norms[:, 0] == 'Spr_StrT'], norms[norms[:, 0] == 'Spr_UtrT']]
    spr, parameters = crit_z_score(spr_feat, spr_ref, parameters)

    # sturdiness
    stu_feat = [ft.step_length(seg_lim, steps_lim, distance=distance)]
    stu_ref = [norms[norms[:, 0] == 'Stu_L']]
    stu, parameters = crit_z_score(stu_feat, stu_ref, parameters)

    # smoothness
    smo_feat = [ft.sparc_gyr(data_lb, seg_lim, freq=freq),
                ft.ldlj_acc(data_lb, seg_lim, freq=freq)]
    smo_ref = [norms[norms[:, 0] == 'Smo_SPARC'], norms[norms[:, 0] == 'Smo_LDLAcc']]
    smo, parameters = crit_z_score(smo_feat, smo_ref, parameters)

    # steadiness
    ste_feat = [ft.variation_coeff_stride_time(seg_lim, steps_lim),
                ft.variation_coeff_double_stance_time(seg_lim, steps_lim),
                ft.p1_acc(data_lb, seg_lim, steps_lim),
                ft.p2_acc(data_lb, seg_lim, steps_lim)]
    ste_ref = [norms[norms[:, 0] == 'Ste_cvstrT'], norms[norms[:, 0] == 'Ste_cvdsT'], norms[norms[:, 0] == 'Ste_P1_aCC_F2'],
               norms[norms[:, 0] == 'Ste_P2_aCC_LB2']]
    ste, parameters = crit_z_score(ste_feat, ste_ref, parameters)

    # stability
    sta_feat = [ft.medio_lateral_root_mean_square(data_lb, seg_lim)]
    sta_ref = [norms[norms[:, 0] == 'Sta_RMS_aML_LB']]
    sta, parameters = crit_z_score(sta_feat, sta_ref, parameters)

    # symmetry
    sym_feat = [ft.p1_p2_acc(data_lb, seg_lim, steps_lim),
                ft.mean_swing_times_ratio(seg_lim, steps_lim),
                ft.antero_posterior_iHR(data_lb, seg_lim, steps_lim),
                ft.medio_lateral_iHR(data_lb, seg_lim, steps_lim),
                ft.cranio_caudal_iHR(data_lb, seg_lim, steps_lim)]
    sym_ref = [norms[norms[:, 0] == 'Sym_P1P2_aCC_LB2'], norms[norms[:, 0] == 'Sym_swTr'], norms[norms[:, 0] == 'Sym_HFaAP'],
               norms[norms[:, 0] == 'Sym_HFaML'], norms[norms[:, 0] == 'Sym_HFaCC']]
    sym, parameters = crit_z_score(sym_feat, sym_ref, parameters)

    # synchronisation
    syn_feat = [ft.double_stance_time(seg_lim, steps_lim)]
    syn_ref = [norms[norms[:, 0] == 'Syn_dsT']]
    syn, parameters = crit_z_score(syn_feat, syn_ref, parameters)

    # average speed
    avg_feat = [ft.avg_speed(seg_lim, freq, distance=distance, release_u_turn=True)]
    avg_ref = [norms[norms[:, 0] == 'AvgSpeed']]
    avg, parameters = crit_z_score(avg_feat, avg_ref, parameters)

    # final values
    semio_val = [spr, stu, smo, ste, sta, sym, syn, avg]

    return criteria_names, semio_val, parameters


def crit_z_score(crit_feat, crit_ref, parameters):
    """Compute the Z-score for a criterion. 
    Step 1 of Algorithm 2 in the IPOL article. 

    Arguments:
        crit_feat {list} -- list of parameter values corresponding to the criterion 
        crit_ref {list} -- list of parameter references corresponding to the criterion
        parameters {list} -- list of all parameter values along with their corresponding Z-scores, updated with each criterion.

    Returns
    -------
    float
        Z-score for the criterion
    """
    
    z_scores = []
    for j in range(len(crit_feat)):
        feat, m, sd, f = crit_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = crit_feat[j]
        parameters.append(val)
        parameters.append(f * (val - m) / sd)
        z_scores.append(f * (val - m) / sd)
        
    return np.average(z_scores), parameters
    
