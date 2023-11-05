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


def new_radar_superpose(semio_val_dict, output=0, age=0, min_r=-10, max_r=4, name="test"):
    """
    Create the final radar of the semiogram, allowing for it to be a representation with overlay.
    
    Arguments
    ----------
    semio_val_dict: dict
                   The dictionnary with different semiogram to plot (1 or 2). The key may be the name of the trial, en the values are 
                   lists with the 8 semio values in the right order. 
    output        : str, optional
                   Folder in which the figure may be saved.
    age           : int, optional
                   Indicates the amount of zero padding to be done to the movement
                   data for estimating the spectral arc length. [default = 4]
    min_r         : int, optional
                   The minimum z-score allowed in the final representation. 
    max_r         : int, optional
                   The maximum z-score allowed in the final representation. 
    name          : str, optional
                   name of the saved file.
                   
    Notes
    -----
    The function does not return a result but saves a figure corresponding to the requested semiogram.
    """
    
    properties = ['Springiness', 'Sturdiness', 'Smoothness',
                  'Steadiness', 'Stability', 'Symmetry',
                  'Synchronisation']

    # representations in the case of an overlay
    line_list = ["-", "--", "."]
    marker_list = ["o", "s", "D"]

    colors = ["darkred", "red", "darkorange",
              "gold", "limegreen", "dodgerblue"]
    nodes = [0.0, 0.5, 0.65, 0.77, 0.86, 1.0]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    norm = mpl.colors.Normalize(vmin=-20, vmax=4)

    # make figure background the same colors as axes
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    mpl.rc('axes', facecolor='white')
    mpl.rc('grid', color='white', linewidth=1, linestyle='-')
    
    # use a polar axes
    try:
        axes = plt.subplot(111, polar=True, axisbg='white')
        axes.grid(False)
    except:
        axes = plt.subplot(111, polar=True, facecolor='white')
        axes.grid(False)

    # set ticks to the number of properties (in radians)
    t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
    plt.xticks(t, [])
    axes.set_rlabel_position(48)
    
    # set yticks from min_r to max_r
    rang = [i for i in range(min_r, max_r + 1, 2)]
    rang_plot = [i for i in range(min_r, max_r + 1, 2)]
    plt.yticks(rang_plot, weight='bold', size=10)
    plt.ylim(min_r, max_r)
    axes.spines['polar'].set_visible(False)

    # plot the scale
    for i in rang:
        if i == 0:  # for 0, bold lines
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

    # name axes
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2:
            ha = "left"
        plt.text(angle_rad, 4.75, properties[i], size=22, horizontalalignment=ha, verticalalignment="center",
                 weight='bold')

    # legend
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), orientation='vertical', location='left',
                      shrink=0.9, anchor=(-0.5, 0.5), aspect=40)
    cb.set_label(label='Average Speed', size=22, weight='bold')
    cb.ax.tick_params(size=10)
    cax = cb.ax

    # points for semio_val_dict
    for k in range(len(semio_val_dict)):
        semio_label = list(semio_val_dict.keys())[k]
        line_ = line_list[len(semio_val_dict) - 1 - k]
        marker_ = marker_list[len(semio_val_dict) - 1 - k]
        semio_val = list(semio_val_dict.values())[k]
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

            if (len(semio_val_dict)==2):
                if k == 1:
                    points_new = points
                    marker_new = marker_
                    semio_label_new = semio_label
                    _patch_new = _patch
                    area_new = patches.Polygon(points, color=cmap1(norm(semio_val[7])), alpha=0.2)
                    axes.add_patch(area_new)
                if k == 0:
                    points_ref = points
                    marker_ref = marker_
                    semio_label_ref = semio_label
                    _patch_ref = _patch
                    area_ref = patches.Polygon(points, color="white", alpha=1)
                    axes.add_patch(area_ref)

            axes.add_patch(_patch)
            # circles for values
            axes.scatter(points[:, 0], points[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                             marker=marker_, label=semio_label)

            cax.scatter(0.5, semio_val[7], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                        marker=marker_)

    # if overlay, you need to reorganize the layers to achieve the correct visual representation highlighting the progressions.
    if len(semio_val_dict) > 1:
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.25), ncol=2, title="Trials", title_fontsize="xx-large",
                   fontsize="x-large", )
        if len(semio_val_dict) == 2:

            # D'abord le fond
            axes.add_patch(area_new)
            axes.add_patch(area_ref)

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

            # ref
            axes.add_patch(_patch_ref)
            axes.scatter(points_ref[:, 0], points_ref[:, 1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10,
                         marker=marker_ref)

            # new
            axes.add_patch(_patch_new)
            axes.scatter(points_new[:, 0], points_new[:, 1], linewidth=2, s=50, color='white', edgecolor='black',
                         zorder=10,
                         marker=marker_new)
    # save the fig
    path_out = os.path.join(output, name + ".svg")
    plt.savefig(path_out, dpi=200,
                    transparent=True, bbox_inches="tight")

