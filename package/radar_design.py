# Objectif : mettre en place le radar en fonction des différents paramètres de référence

import os
import numpy as np
import matplotlib as mpl
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import math
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def new_radar(semio_val, ref, output, age, min_r=-10, max_r=4):
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    mpl.rc('axes', facecolor='white')
    mpl.rc('grid', color='white', linewidth=1, linestyle='-')
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
    # rang = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    rang = [i for i in range(min_r, max_r + 1, 2)]
    rang_plot = [i for i in range(min_r, max_r + 1, 2)]
    plt.yticks(rang_plot, weight='bold', size=10)

    # Limites des axes
    # plt.ylim(-4, 4)
    plt.ylim(min_r, max_r)
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
        plt.text(angle_rad, 4.75, properties[i], size=22, horizontalalignment=ha, verticalalignment="center",
                 weight='bold')

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

    black_patch = patches.Patch(color='black', label="Healthy subject", fill=False, linestyle='dashed')
    # white_patch2 = patches.Patch(color='white', fill=False, label=lab)

    # Légende pour les couleurs de la vitesse
    fontP = FontProperties()
    fontP.set_size('small')
    legende = plt.legend(
        handles=[red_patch, orange_patch, green_patch, cyan_patch, blue_patch, white_patch, black_patch],
        # , white_patch2],
        bbox_to_anchor=(1.5, 1.5), borderaxespad=0.1, loc='upper right', ncol=1, prop=fontP, title="Average speed", fontsize=18)
    title = legende.get_title()
    title.set_weight("bold")
    title.set_size(20)
    legende.legendPatch.set_linewidth(2)

    # Tracer les points de semio_val
    if len(semio_val) != 8:
        print("ERROR : The number of arguments of semio_val must be 8 !")
    else:
        for i in range(7):
            if semio_val[i] > max_r:
                semio_val[i] = max_r
            if semio_val[i] < min_r:
                semio_val[i] = min_r
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

        # On enregistre la nouvelle figure
        titre = ref + "_semiogram"
        path_out = output + '/' + titre + '.png'
        plt.savefig(path_out)
        plt.close()
        print("RAS image")


def new_radar_aggreg(semio_val_ag, date, id, output, age=None, min_r=-10, max_r=4):
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    mpl.rc('axes', facecolor='white')
    mpl.rc('grid', color='white', linewidth=1, linestyle='-')
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
    # rang = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    rang = [i for i in range(min_r, max_r + 1, 2)]
    rang_plot = [i for i in range(min_r, max_r + 1, 2)]
    plt.yticks(rang_plot, weight='bold', size=10)

    # Limites des axes
    plt.ylim(min_r, max_r)
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
        plt.text(angle_rad, 4.75, properties[i], size=22, horizontalalignment=ha, verticalalignment="center",
                 weight='bold')

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

    black_patch = patches.Patch(color='black', label="Healthy subject", fill=False, linestyle='dashed')
    # white_patch2 = patches.Patch(color='white', fill=False, label=lab)

    # Légende pour les couleurs de la vitesse
    fontP = FontProperties()
    fontP.set_size('small')
    legende = axes.legend(
        handles=[red_patch, orange_patch, green_patch, cyan_patch, blue_patch, white_patch, black_patch],
        # , white_patch2],
        bbox_to_anchor=(1.5, 1.5), borderaxespad=0.1, loc='upper right', ncol=1, prop=fontP, title="Average speed",
        fontsize=18)
    title = legende.get_title()
    title.set_weight("bold")
    title.set_size(20)
    legende.legendPatch.set_linewidth(2)

    # Tracer les points de semio_val
    if len(semio_val_ag) != 8:
        print("ERROR : The number of arguments of semio_val_ag must be 8 !")
    else:
        for i in range(7):
            if semio_val_ag[i][0] > max_r:
                semio_val_ag[i][0] = max_r
                semio_val_ag[i][1] = 0
            if semio_val_ag[i][0] < min_r:
                semio_val_ag[i][0] = min_r
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
        sd_up = [min(semio_val_ag[i][0] + semio_val_ag[i][1], max_r) for i in range(8)]

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
        sd_down = [max(semio_val_ag[i][0] - semio_val_ag[i][1], min_r) for i in range(8)]

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

        # On enregistre la nouvelle figure
        titre = id + '_' + date + "_aggregate_semiogram"
        path_out = output + '/' + titre + '.png'
        plt.savefig(path_out)
        plt.close()

    return None


def new_radar_superpose(semio_val_dict, type_=None, ref=0, output=0, age=0, min_r=-10, max_r=4, color_diff=False, name="test"):
    print("color_diff", color_diff)
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # A mettre en global ?
    line_list = ["-", "--", "."]
    marker_list = ["o", "s", "D"]

    colors = ["darkred", "red", "darkorange",
              "gold", "limegreen", "dodgerblue"]
    nodes = [0.0, 0.5, 0.65, 0.77, 0.86, 1.0]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    norm = mpl.colors.Normalize(vmin=-20, vmax=4)

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    mpl.rc('axes', facecolor='white')
    mpl.rc('grid', color='white', linewidth=1, linestyle='-')
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
    # rang = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    rang = [i for i in range(min_r, max_r + 1, 2)]
    rang_plot = [i for i in range(min_r, max_r + 1, 2)]
    plt.yticks(rang_plot, weight='bold', size=10)

    # Limites des axes
    # plt.ylim(-4, 4)
    plt.ylim(min_r, max_r)
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
        plt.text(angle_rad, 4.75, properties[i], size=22, horizontalalignment=ha, verticalalignment="center",
                 weight='bold')

    # Installer la légende
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), orientation='vertical', location='left',
                      shrink=0.9, anchor=(-0.5, 0.5), aspect=40)
    cb.set_label(label='Average Speed', size=22, weight='bold')
    cb.ax.tick_params(size=10)
    cax = cb.ax

    # Tracer les points de semio_val_dict
    for k in range(len(semio_val_dict)):
        print("semio_val_dict", semio_val_dict)
        print(k)
        semio_label = list(semio_val_dict.keys())[k]
        line_ = line_list[len(semio_val_dict) - 1 - k]
        marker_ = marker_list[len(semio_val_dict) - 1 - k]
        semio_val = list(semio_val_dict.values())[k]
        print(semio_label, line_, marker_)
        if len(semio_val) != 8:
            print("ERROR : The number of arguments of semio_val must be 8 !")
        else:
            for i in range(7):
                if semio_val[i] > max_r:
                    semio_val[i] = max_r
                if semio_val[i] < min_r:
                    semio_val[i] = min_r
                if math.isnan(semio_val[i]):
                    semio_val[i] = 0
            points = [(x, y) for x, y in zip(t, semio_val[:7])]
            points.append(points[0])
            points = np.array(points)
            codes = [path.Path.MOVETO, ] + \
                    [path.Path.LINETO, ] * (len(semio_val[:7]) - 1) + \
                    [path.Path.CLOSEPOLY]
            _path = path.Path(points, codes)

            _patch = patches.PathPatch(_path, linestyle=line_, fill=False, linewidth=2,
                                       edgecolor=cmap1(norm(semio_val[7])), label=semio_label)

            if (len(semio_val_dict)==2) & color_diff:
                print("On entre dans la mise en couleur")
                if k == 1:
                    points_M6 = points
                    marker_M6 = marker_
                    semio_label_M6 = semio_label
                    _patch_M6 = _patch
                    area_M6 = patches.Polygon(points, color=cmap1(norm(semio_val[7])), alpha=0.2)
                    axes.add_patch(area_M6)
                if k == 0:
                    points_M0 = points
                    marker_M0 = marker_
                    semio_label_M0 = semio_label
                    _patch_M0 = _patch
                    area_M0 = patches.Polygon(points, color="white", alpha=1)
                    axes.add_patch(area_M0)
            #else:
            axes.add_patch(_patch)
                # On trace des cercles au niveau des valeurs
            axes.scatter(points[:, 0], points[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                             marker=marker_, label=semio_label)

            cax.scatter(0.5, semio_val[7], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                        marker=marker_)
            # cax.hlines(-3, -1, 2, colors = 'white', linewidth = 2, linestyles = 'o')

    if len(semio_val_dict) > 1:
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.25), ncol=2, title="Trials", title_fontsize="xx-large",
                   fontsize="x-large", )
        if len(semio_val_dict) == 2:

            # D'abord le fond
            axes.add_patch(area_M6)
            axes.add_patch(area_M0)

            # Ensuite les lignes
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

            # Ensuite M0
            axes.add_patch(_patch_M0)
            axes.scatter(points_M0[:, 0], points_M0[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                         marker=marker_M0)

            # Ensuite M6
            axes.add_patch(_patch_M6)
            axes.scatter(points_M6[:, 0], points_M6[:, 1], linewidth=2, s=50, color='white', edgecolor='black',
                         zorder=10,
                         marker=marker_M6)
    path_out = os.path.join(output, name + ".svg")
    plt.savefig(path_out, dpi=200,
                    transparent=True, bbox_inches="tight")

