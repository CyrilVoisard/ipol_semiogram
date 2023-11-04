import numpy as np
import scipy
import os
import os.path as osp
import json

import matplotlib
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import math

#os.chdir('/Users/cyril/Library/Mobile Documents/com~apple~CloudDocs/Borelli/4 - Classes et fonctions')
from semio_package import features as ft

home = os.getcwd()


def new_radar(semio_val, id_exp, max = 2, min =-10, age = None):
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # Make figure background the same colors as axes
    plt.figure(figsize=(11, 8), facecolor='white')
    matplotlib.rc('axes', facecolor='white')
    matplotlib.rc('grid', color='white', linewidth=1, linestyle='-')
    # Use a polar axes
    try:
        axes = plt.subplot(111, polar=True, axisbg='white')
        axes.grid(False)
    except:
        axes = plt.subplot(111, polar=True, facecolor='white')
        axes.grid(False)

    # Set ticks to the number of properties (in radians)
    t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
    plt.xticks(t, [])
    axes.set_rlabel_position(48)
    # Set yticks from -4 to +4
    rang = [i for i in range(min, max + 1)]
    plt.yticks(rang)

    # Limites des axes
    plt.ylim(min, max)
    axes.spines['polar'].set_visible(False)

    # Plot des différentes valeurs de l'échelle
    for i in rang:
        if i == 0:  # Pour 0, particulier : pointillés en gras
            style = 'dashed'
            line = 2
        else:
            style = ':'
            line = 1
        values = [i, i, i, i, i, i, i]
        points = [(x, y) for x, y in zip(t, values)]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO, ] + \
                [path.Path.LINETO, ] * (len(values) - 1) + \
                [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=False, linewidth=line, linestyle=style)
        axes.add_patch(_patch)

    # Nommer les axes dans l'ordre : revoir l'ordre pour regrouper les tendances ????????
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2:
            ha = "left"
        plt.text(angle_rad, 4.75, properties[i], size=14, horizontalalignment=ha, verticalalignment="center")

    # Couleurs pour caractériser la vitesse de marche
    red_patch = patches.Patch(color='red', label='< -2SD', fill=False)
    orange_patch = patches.Patch(color='orange', label='[-2SD ; -1SD ]', fill=False)
    green_patch = patches.Patch(color='green', label='[-1SD ; 1SD ]', fill=False)
    cyan_patch = patches.Patch(color='cyan', label='[1SD ; 2SD ]', fill=False)
    blue_patch = patches.Patch(color='blue', label='[>2SD ]', fill=False)
    white_patch = patches.Patch(color='white', fill=False, label=' ')

    # Tranches d'âges et comparaison : pas encore utilisé.
    ages = [[0, 100], [0, 17], [18, 39], [40, 59], [60, 79], [80, 100]]
    if age is None:
        r = 0
        lab = "Not spec"
    else:
        for i in range(1, 6):
            if ages[i][0] < age < ages[i][1]:
                r = i
                lab = str(ages[i]) + '-' + str(ages[i][1]) + ' ans'

    black_patch = patches.Patch(color='black', label="Sujets sains", fill=False, linestyle='dashed')
    white_patch2 = patches.Patch(color='white', fill=False, label=lab)

    # Légende pour les couleurs de la vitesse
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(
        handles=[red_patch, orange_patch, green_patch, cyan_patch, blue_patch, white_patch, black_patch, white_patch2],
        bbox_to_anchor=(1.02, 1), borderaxespad=0.1, loc=2, ncol=1, prop=fontP, title="Vitesse")

    # Tracer les points de semio_val
    if len(semio_val) != 8:
        print("ERROR : The number of arguments of semio_val must be 8 !")
    else:
        for i in range(7):
            if semio_val[i] > max:
                semio_val[i] = max
            if semio_val[i] < min:
                semio_val[i] = min
            if math.isnan(semio_val[i]):
                semio_val[i] = 0
        points = [(x, y) for x, y in zip(t, semio_val[:7])]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO, ] + \
                [path.Path.LINETO, ] * (len(semio_val[:7]) - 1) + \
                [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)

        if semio_val[7] <= -2:
            col = "red"
        if -2 < semio_val[7] < -1:
            col = "orange"
        if -1 <= semio_val[7] <= 1:
            col = "green"
        if 1 < semio_val[7] < 2:
            col = "cyan"
        if semio_val[7] >= 2:
            col = "blue"

        _patch = patches.PathPatch(_path, fill=False, linewidth=2, edgecolor=col)
        axes.add_patch(_patch)

        # On trace des cercles au niveau des valeurs
        plt.scatter(points[:, 0], points[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10)
    
    plt.suptitle(('Semiogram pour ' + str(id_exp)), fontsize = 12, x = 0.5, y = 0.995)



def new_radar_aggreg(semio_val_par_date, date, id_patient, max = 2, min = -10, age=None):

    semio_val_ag = aggregate(semio_val_par_date)

    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    matplotlib.rc('axes', facecolor='white')
    matplotlib.rc('grid', color='white', linewidth=1, linestyle='-')
    # Use a polar axes
    try:
        axes = plt.subplot(111, polar=True, axisbg='white')
        axes.grid(False)
    except:
        axes = plt.subplot(111, polar=True, facecolor='white')
        axes.grid(False)

    # Set ticks to the number of properties (in radians)
    t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
    plt.xticks(t, [])
    axes.set_rlabel_position(48)
    # Set yticks from -4 to +4
    rang = [i for i in range(min, max + 1)]
    plt.yticks(rang)

    # Limites des axes
    plt.ylim(min, max)
    axes.spines['polar'].set_visible(False)

    # Plot des différentes valeurs de l'échelle
    for i in rang:
        if i == 0:  # Pour 0, particulier : pointillés en gras
            style = 'dashed'
            line = 2
        else:
            style = ':'
            line = 1
        values = [i, i, i, i, i, i, i]
        points = [(x, y) for x, y in zip(t, values)]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO, ] + \
                [path.Path.LINETO, ] * (len(values) - 1) + \
                [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=False, linewidth=line, linestyle=style)
        axes.add_patch(_patch)

    # Nommer les axes dans l'ordre : revoir l'ordre pour regrouper les tendances ????????
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2 :
            ha = "left"
        plt.text(angle_rad, 4.75, properties[i], size=14, horizontalalignment=ha, verticalalignment="center")

    # Couleurs pour caractériser la vitesse de marche
    red_patch = patches.Patch(color='red', label='< -2SD', fill=False)
    orange_patch = patches.Patch(color='orange', label='[-2SD ; -1SD ]', fill=False)
    green_patch = patches.Patch(color='green', label='[-1SD ; 1SD ]', fill=False)
    cyan_patch = patches.Patch(color='cyan', label='[1SD ; 2SD ]', fill=False)
    blue_patch = patches.Patch(color='blue', label='[>2SD ]', fill=False)
    white_patch = patches.Patch(color='white', fill=False, label=' ')

    # Tranches d'âges et comparaison
    ages = [[0, 100], [0, 17], [18, 39], [40, 59], [60, 79], [80, 100]]
    if age is None:
        r = 0
        lab = "Not spec"
    else:
        for i in range(1, 6):
            if ages[i][0] < age < ages[i][1]:
                r = i
                lab = str(ages[i]) + '-' + str(ages[i][1]) + ' ans'

    black_patch = patches.Patch(color='black', label="Sujets sains", fill=False, linestyle='dashed')
    white_patch2 = patches.Patch(color='white', fill=False, label=lab)

    # Légende pour les couleurs de la vitesse
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(
        handles=[red_patch, orange_patch, green_patch, cyan_patch, blue_patch, white_patch, black_patch, white_patch2],
        bbox_to_anchor=(1.02, 1), borderaxespad=0.1, loc=2, ncol=1, prop=fontP, title="Vitesse")

    # Tracer les points de semio_val
    if len(semio_val_ag) != 8:
        print("ERROR : The number of arguments of semio_val_ag must be 8 !")
    else:
        for i in range(7):
            if semio_val_ag[i][0] > max:
                semio_val_ag[i][0] = max
                semio_val_ag[i][1] = 0
            if semio_val_ag[i][0] < min:
                semio_val_ag[i][0] = min
                semio_val_ag[i][1] = 0
            if math.isnan(semio_val_ag[i][0]):
                semio_val_ag[i] = 0
                semio_val_ag[i][1] = 0

        semio_val_ag = np.array(semio_val_ag)

        # Plot des moyennes, en trait plein ----------------------------------------------------------------
        points = [(x, y) for x, y in zip(t, semio_val_ag[:, 0][:7])]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO, ] + \
                [path.Path.LINETO, ] * (len(semio_val_ag[:, 0][:7]) - 1) + \
                [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)

        # Pour la vitesse, on ne prend que la moyenne
        if semio_val_ag[:, 0][7] <= -2:
            col = "red"
        if -2 < semio_val_ag[:, 0][7] < -1:
            col = "orange"
        if -1 <= semio_val_ag[:, 0][7] <= 1:
            col = "green"
        if 1 < semio_val_ag[:, 0][7] < 2:
            col = "cyan"
        if semio_val_ag[:, 0][7] >= 2:
            col = "blue"

        _patch = patches.PathPatch(_path, fill=False, linewidth=4, edgecolor=col)
        axes.add_patch(_patch)

        # On trace des cercles au niveau des valeurs
        plt.scatter(points[:, 0], points[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10)

        # Plot des deviations hautes, en remplissage  sur 1 sd ---------------------------------------------------------
        #print(semio_val_ag)
        sd_up = []
        for i in range(8):
            #print(semio_val_ag[i][0])
            lim_haute = np.min([semio_val_ag[i][0] + semio_val_ag[i][1], max])
            sd_up.append(lim_haute)

        points_up = [(x, y) for x, y in zip(t, sd_up[:7])]
        points_up.append(points_up[0])
        points_up = np.array(points_up)
        codes_up = [path.Path.MOVETO, ] + \
                   [path.Path.LINETO, ] * (len(sd_up[:7]) - 1) + \
                   [path.Path.CLOSEPOLY]
        _path_up = path.Path(points_up, codes_up)

        if sd_up[7] <= -2:
            col_up = "red"
        if -2 < sd_up[7] < -1:
            col_up = "orange"
        if -1 <= sd_up[7] <= 1:
            col_up = "green"
        if 1 < sd_up[7] < 2:
            col_up = "cyan"
        if sd_up[7] >= 2:
            col_up = "blue"

        _patch_up = patches.PathPatch(_path_up, fill=False, linewidth=1, linestyle='dashed', edgecolor=col_up)
        axes.add_patch(_patch_up)

        # Plot des deviations basses, en remplissage  sur 1 sd ---------------------------------------------------------
        sd_down = []
        for i in range(8):
            lim_basse = np.max([semio_val_ag[i][0] - semio_val_ag[i][1], min])
            sd_down.append(lim_basse)

        points_down = [(x, y) for x, y in zip(t, sd_down[:7])]
        points_down.append(points_down[0])
        points_down = np.array(points_down)
        codes_down = [path.Path.MOVETO, ] + \
                     [path.Path.LINETO, ] * (len(sd_down[:7]) - 1) + \
                     [path.Path.CLOSEPOLY]
        _path_down = path.Path(points_down, codes_down)

        if sd_down[7] <= -2:
            col_down = "red"
        if -2 < sd_down[7] < -1:
            col_down = "orange"
        if -1 <= sd_down[7] <= 1:
            col_down = "green"
        if 1 < sd_down[7] < 2:
            col_down = "cyan"
        if sd_down[7] >= 2:
            col_down = "blue"

        # Remplissage
        area_up = patches.Polygon(points_up, color=col_up, alpha=0.2)
        axes.add_patch(area_up)
        area_norm = patches.Polygon(points, color='white', alpha=1)
        axes.add_patch(area_norm)
        area_down = patches.Polygon(points, color=col_down, alpha=0.2)
        axes.add_patch(area_down)
        area_norm = patches.Polygon(points_down, color='white', alpha=1)
        axes.add_patch(area_norm)

        # Plot des différentes valeurs de l'échelle
        for i in rang:
            if i == 0:  # Pour 0, particulier : pointillés en gras
                style = 'dashed'
                line = 2
            else:
                style = ':'
                line = 1
            values = [i, i, i, i, i, i, i]
            points = [(x, y) for x, y in zip(t, values)]
            points.append(points[0])
            points = np.array(points)
            codes = [path.Path.MOVETO, ] + \
                    [path.Path.LINETO, ] * (len(values) - 1) + \
                    [path.Path.CLOSEPOLY]
            _path = path.Path(points, codes)
            _patch = patches.PathPatch(_path, fill=False, linewidth=line, linestyle=style)
            axes.add_patch(_patch)

        _patch_down = patches.PathPatch(_path_down, fill=False, linewidth=1, linestyle='dashed', edgecolor=col_down)
        axes.add_patch(_patch_down)

        plt.suptitle(('Semiogram aggrégé à la date ' + str(date) +  ' pour ' + str(id_patient)), fontsize = 12, x = 0.5, y = 0.995)

    return None




def compute_semio_val(data_tronc, seg_lim, steps_lim, age = None):
    # Liste des 7 paramètres, auxquels on ajoutera la vitesse moyenne
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation', 'Average Speed']

    # Paramètres fixés
    freq = 100  # Fréquence d'acquisition des XSens : 100 Hz

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
    path = osp.join("/Users/cyril/Library/Mobile Documents/com~apple~CloudDocs/Borelli/4 - Classes et fonctions/semiogram_package_poubelle", r) + '.json'
    with open(path) as json_data:
        d = np.array(json.load(json_data))
        #print(d)

    # Springiness : RAS
    spr_feat = [ft.stride_time(data_tronc, seg_lim, steps_lim), ft.u_turn_time(data_tronc, seg_lim, steps_lim)]
    spr_ref = [d[d[:, 0] == 'Spr_StrT'], d[d[:, 0] == 'Spr_UtrT']]
    spr_z_scores = []
    for j in range(len(spr_feat)):
        feat, m, sd, f = spr_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = spr_feat[j]
        spr_z_scores.append(f * (val - m) / sd)
    spr = np.average(spr_z_scores)

    # Sturdiness : RAS.
    stu_feat = [ft.step_length(data_tronc, seg_lim, steps_lim)]
    stu_ref = [d[d[:, 0] == 'Stu_L']]
    stu_z_scores = []
    for j in range(len(stu_feat)):
        feat, m, sd, f = stu_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = stu_feat[j]
        stu_z_scores.append(f * (val - m) / sd)
    stu = np.average(stu_z_scores)

    # Smoothness : RAS
    smo_feat = [ft.antero_posterieur_root_mean_square(data_tronc, seg_lim, steps_lim), ft.sparc_gyr(data_tronc, seg_lim, steps_lim), ft.ldlj_acc_LB(data_tronc, seg_lim, steps_lim), ft.ldlj_gyr_LB(data_tronc, seg_lim, steps_lim)]
    smo_ref = [d[d[:, 0] == 'Smo_RMS_aAP_LB'], d[d[:, 0] == 'Smo_SPARC_gyr'], d[d[:, 0] == 'Smo_ldlj_acc_LB'], d[d[:, 0] == 'Smo_ldlj_gyr_LB']]
    smo_z_scores = []
    for j in range(len(smo_ref)):
        feat, m, sd, f = smo_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = smo_feat[j]
        smo_z_scores.append(f * (val - m) / sd)
        #smo_z_scores.append(0)
    smo = np.average(smo_z_scores)

    # Steadiness : RAS
    ste_feat = [ft.variation_coeff_stride_time(data_tronc, seg_lim, steps_lim), ft.variation_coeff_double_stance_time(data_tronc, seg_lim, steps_lim),
                ft.p1_acc(data_tronc, seg_lim, steps_lim), ft.p2_acc(data_tronc, seg_lim, steps_lim)]
    ste_ref = [d[d[:, 0] == 'Ste_cvstrT'], d[d[:, 0] == 'Ste_cvdsT'], d[d[:, 0] == 'Ste_P1_aCC_F2'],
               d[d[:, 0] == 'Ste_P2_aCC_LB2']]
    ste_z_scores = []
    for j in range(len(ste_feat)):
        feat, m, sd, f = ste_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = ste_feat[j]
        ste_z_scores.append(f * (val - m) / sd)
    ste = np.average(ste_z_scores)

    # Stability : RAS.
    sta_feat = [ft.medio_lateral_root_mean_square(data_tronc, seg_lim, steps_lim)]
    sta_ref = [d[d[:, 0] == 'Sta_RMS_aML_LB']]
    sta_z_scores = []
    for j in range(len(sta_feat)):
        feat, m, sd, f = sta_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = sta_feat[j]
        sta_z_scores.append(f * (val - m) / sd)
    sta = np.average(sta_z_scores)

    # Symmetry :RAS.
    sym_feat = [ft.p1_p2_acc(data_tronc, seg_lim, steps_lim),
                ft.mean_swing_times_ratio(data_tronc, seg_lim, steps_lim),
                ft.antero_posterior_iHR(data_tronc, seg_lim, steps_lim),
                ft.medio_lateral_iHR(data_tronc, seg_lim, steps_lim),
                ft.cranio_caudal_iHR(data_tronc, seg_lim, steps_lim)]
    sym_ref = [d[d[:, 0] == 'Sym_P1P2_aCC_LB2'], d[d[:, 0] == 'Sym_swTr'], d[d[:, 0] == 'Sym_HFaAP'],
               d[d[:, 0] == 'Sym_HFaML'], d[d[:, 0] == 'Sym_HFaCC']]
    sym_z_scores = []
    for j in range(len(sym_feat)):
        feat, m, sd, f = sym_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = sym_feat[j]
        sym_z_scores.append(f * (val - m) / sd)
    sym = np.average(sym_z_scores)

    # Synchronisation : RAS.
    syn_feat = [ft.double_stance_time(data_tronc, seg_lim, steps_lim)]
    syn_ref = [d[d[:, 0] == 'Syn_dsT']]
    syn_z_scores = []
    for j in range(len(syn_feat)):
        feat, m, sd, f = syn_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = syn_feat[j]
        syn_z_scores.append(f * (val - m) / sd)
    syn = np.average(syn_z_scores)

    # Average speed : RAS.
    avg_feat = [ft.avg_speed(data_tronc, seg_lim, steps_lim)]
    avg_ref = [d[d[:, 0] == 'AvgSpeed']]
    avg_z_scores = []
    for j in range(len(avg_feat)):
        feat, m, sd, f = avg_ref[j].tolist()[0]
        m = float(m)
        sd = float(sd)
        f = int(f)
        val = avg_feat[j]
        avg_z_scores.append(f * (val - m) / sd)
    avg = np.average(avg_z_scores)

    # Valeurs finales
    semio_val = [spr, stu, smo, ste, sta, sym, syn, avg]
    # print("semio_val=", semio_val)

    return semio_val


def aggregate(semio_val_par_date):
    if len(semio_val_par_date) > 0:
        semio_val_par_date_array = np.array(semio_val_par_date)
        result = []
        for f in range(len(semio_val_par_date[0])):
            result.append([np.mean(semio_val_par_date_array[:, f]), np.std(semio_val_par_date_array[:, f])])
        return result
    else:
        return None
