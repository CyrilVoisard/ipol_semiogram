# Objectif : A partir des données de features, proposer les valeurs comparatives en SD des 7 paramètres et de la
# vitesse en fonction de la classe d'âge du patient.

import numpy as np
import scipy
import os
import os.path as osp
import json
from package import features as ft


def compute_semio_val(age, steps_lim, seg_lim, data_tronc, freq):
    # Liste des 7 paramètres, auxquels on ajoutera la vitesse moyenne
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation', 'Average Speed']

    perso_feat = []

    # Chargement du fichier modèle en fonction de l'âge
    ages = [[0, 100], [0, 17], [18, 39], [40, 59], [60, 79], [80, 100]]
    if age is None:
        ref = 0
    else:
        for i in range(1, 6):
            if ages[i][0] < age < ages[i][1]:
                ref = i

    age1 = ages[ref][0]
    age2 = ages[ref][1]
    r = 'ScoreT7S-' + str(age1) + '-' + str(age2)
    path = osp.join('models', r) + '.json'
    with open(path) as json_data:
        d = np.array(json.load(json_data))
        # print(d)

    # Springiness : RAS
    spr_feat = [ft.stride_time(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.u_turn_time(data_tronc, seg_lim, steps_lim, freq=freq)]
    spr_ref = [d[d[:, 0] == 'Spr_StrT'], d[d[:, 0] == 'Spr_UtrT']]
    spr_z_scores = []
    for j in range(len(spr_feat)):
        feat, m, sd, f = spr_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = spr_feat[j]
        print("val", val)
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        spr_z_scores.append(f * (val - m) / sd)
    spr = np.average(spr_z_scores)

    # Sturdiness : RAS.
    stu_feat = [ft.step_length(data_tronc, seg_lim, steps_lim, freq=freq)]
    stu_ref = [d[d[:, 0] == 'Stu_L']]
    stu_z_scores = []
    for j in range(len(stu_feat)):
        feat, m, sd, f = stu_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = stu_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        stu_z_scores.append(f * (val - m) / sd)
    stu = np.average(stu_z_scores)

    # Smoothness : RAS
    smo_feat = [ft.sparc_gyr(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.ldlj_acc(data_tronc, seg_lim, steps_lim, freq=freq)]

# [ft.antero_posterieur_root_mean_square(data_tronc, seg_lim, steps_lim, freq=freq),
    smo_ref = [d[d[:, 0] == 'Smo_SPARC'], d[d[:, 0] == 'Smo_LDLAcc']] #d[d[:, 0] == 'Smo_RMS_aAP_LB'],
    smo_z_scores = []
    for j in range(len(smo_feat)):
        feat, m, sd, f = smo_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = smo_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        smo_z_scores.append(f * (val - m) / sd)
    smo = np.average(smo_z_scores)

    # Steadiness : RAS
    ste_feat = [ft.variation_coeff_stride_time(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.variation_coeff_double_stance_time(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.p1_acc(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.p2_acc(data_tronc, seg_lim, steps_lim, freq=freq)]
    ste_ref = [d[d[:, 0] == 'Ste_cvstrT'], d[d[:, 0] == 'Ste_cvdsT'], d[d[:, 0] == 'Ste_P1_aCC_F2'],
               d[d[:, 0] == 'Ste_P2_aCC_LB2']]
    ste_z_scores = []
    for j in range(len(ste_feat)):
        feat, m, sd, f = ste_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = ste_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        # print("val=", val)
        # print("m=", m)
        # print("sd=", sd)
        # print("f=", f)
        ste_z_scores.append(f * (val - m) / sd)
    ste = np.average(ste_z_scores)

    # Stability : RAS.
    sta_feat = [ft.medio_lateral_root_mean_square(data_tronc, seg_lim, steps_lim, freq=freq)]
    sta_ref = [d[d[:, 0] == 'Sta_RMS_aML_LB']]
    sta_z_scores = []
    for j in range(len(sta_feat)):
        feat, m, sd, f = sta_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = sta_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        sta_z_scores.append(f * (val - m) / sd)
    sta = np.average(sta_z_scores)

    # Symmetry :RAS.
    sym_feat = [ft.p1_p2_acc(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.mean_swing_times_ratio(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.antero_posterior_iHR(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.medio_lateral_iHR(data_tronc, seg_lim, steps_lim, freq=freq),
                ft.cranio_caudal_iHR(data_tronc, seg_lim, steps_lim, freq=freq)]
    sym_ref = [d[d[:, 0] == 'Sym_P1P2_aCC_LB2'], d[d[:, 0] == 'Sym_swTr'], d[d[:, 0] == 'Sym_HFaAP'],
               d[d[:, 0] == 'Sym_HFaML'], d[d[:, 0] == 'Sym_HFaCC']]
    sym_z_scores = []
    for j in range(len(sym_feat)):
        feat, m, sd, f = sym_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = sym_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        sym_z_scores.append(f * (val - m) / sd)
    sym = np.average(sym_z_scores)

    # Synchronisation : RAS.
    syn_feat = [ft.double_stance_time(data_tronc, seg_lim, steps_lim, freq=freq)]
    syn_ref = [d[d[:, 0] == 'Syn_dsT']]
    syn_z_scores = []
    for j in range(len(syn_feat)):
        feat, m, sd, f = syn_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = syn_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        syn_z_scores.append(f * (val - m) / sd)
    syn = np.average(syn_z_scores)

    # Average speed : RAS.
    avg_feat = [ft.avg_speed(data_tronc, seg_lim, steps_lim, release_u_turn=True, freq=freq)]
    avg_ref = [d[d[:, 0] == 'AvgSpeed']]
    avg_z_scores = []
    for j in range(len(avg_feat)):
        feat, m, sd, f = avg_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = avg_feat[j]
        perso_feat.append(val)
        perso_feat.append(f * (val - m) / sd)
        avg_z_scores.append(f * (val - m) / sd)
    avg = np.average(avg_z_scores)

    # Valeurs finales
    semio_val = [spr, stu, smo, ste, sta, sym, syn, avg]
    # print("semio_val=", semio_val)

    return properties, semio_val, perso_feat


def aggregate(semio_val_par_date):
    if len(semio_val_par_date) > 0:
        semio_val_par_date_array = np.array(semio_val_par_date)
        result = []
        for f in range(len(semio_val_par_date[0])):
            result.append([np.mean(semio_val_par_date_array[:, f]), np.std(semio_val_par_date_array[:, f])])
        return result
    else:
        return None
