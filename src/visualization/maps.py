import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import rpack
import src.utils.geometry as geo

def basic(geodf, plot_series, fig=None, ax=None, **plot_kw):
    plot_geodf = geodf.join(plot_series, how='inner')
    if ax is None:
        fig, ax = plt.subplots(1)
    plot_geodf.to_crs('epsg:4326').plot(ax=ax, column=plot_series.name,
                                        legend=True, **plot_kw)
    ax.set_axis_off()
    return fig, ax, plot_geodf


def word_prop(word, global_words, valid_geos, word_vectors, geodf,
              **basic_kwargs):
    iloc_word = np.where(global_words['word'].values == word)[0][0]
    proj_cells = pd.Series(word_vectors[:, iloc_word],
                           index=valid_geos, name='plot')
    return basic(geodf, proj_cells, **basic_kwargs)


def clusters(geodf, cell_clusters, valid_cells, **plot_kw):
    plot_series = pd.Series(cell_clusters + 1,
                            name='hierarc', index=valid_cells)
    return basic(geodf, plot_series, categorical=True, **plot_kw)


def overlap_clusters(geodf, cluster_data, fig=None, ax=None, **plot_kw):
    '''
    Plot a cluster map with a legend.
    '''
    if ax is None:
        fig, ax = plt.subplots(1)
    plot_geodf = prep_cluster_plot(geodf, cluster_data)
    unique_labels = sorted(plot_geodf['label'].unique())
    if 'homeless' in unique_labels:
        colors = gen_distinct_colors(len(unique_labels) - 1) + [(0.5, 0.5, 0.5)]
    else:
        colors = gen_distinct_colors(len(unique_labels))
    for i, (_, data) in enumerate(plot_geodf.groupby('label')):
        color = colors[i]
        data.plot(color=color, ax=ax, **plot_kw)

    handles = []
    for c, l in zip(colors, unique_labels):
        # The colours will correspond because groupby sorts by the column by
        # which we group, and we sorted the unique labels.
        handles.append(mlines.Line2D([], [], color=c, marker='o',
                                     linestyle='None', markersize=6, label=l))
    ax.legend(handles=handles)

    ax.set_axis_off()
    return fig, ax, plot_geodf


def gen_distinct_colors(num_colors):
    '''
    Generate `num_colors` distinct colors for a discrete colormap, in the format
    of a list of tuples of normalized RGB values.
    '''
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (40 + np.random.rand() * 20) / 100.
        saturation = (80 + np.random.rand() * 20) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def colored_pts_legend(container, label_color, marker='o', markersize=6,
                       loc='upper right'):
    '''
    Adds to a legend with colored points to a `container`, which can be a plt ax
    or figure. The color of the points and their associated labels are given
    respectively as values and keys of the dict `label_color`.
    '''
    handles = []
    for l, c in label_color.items():
        handles.append(mlines.Line2D([], [], color=c, marker=marker, label=l,
                                     linestyle='None', markersize=markersize))
    container.legend(handles=handles, loc=loc)
    return container


def prep_cluster_plot(geodf, cluster_data):
    '''
    Prepares the GeoDataFrame `geodf` to be passed to `overlap_clusters` or
    `interactive.clusters`, by adding a column for the cluster labels, allowing
    for overlapping clusters.
    '''
    valid_cells = geodf.index
    if isinstance(cluster_data, np.ndarray):
        cell_dict = {
            cell_id: [clust]
            for cell_id, clust in zip(valid_cells, cluster_data)}
    elif isinstance(cluster_data, dict):
        if len(cluster_data) == len(valid_cells):
            cell_dict = dict(zip(valid_cells, cluster_data.values()))
        else:
            # translate dict cluster: [cells] to cell: [clusters]
            cell_dict = {cell_id: [] for cell_id in valid_cells}
            for cluster, cells in cluster_data.items():
                for c in cells:
                    cell_dict[valid_cells[c]].append(cluster)
    else:
        raise TypeError('''cluster_data must either be an array of cluster
                        labels (as is the case for the result from hierarchical
                        clustering), or a dictionary mapping clusters to a list
                        of cells, or cells to a list of clusters''')

    plot_geodf = geodf.join(pd.Series(cell_dict, name='clusters'), how='inner')
    plot_geodf['label'] = (
        plot_geodf['clusters'].apply(lambda x: '+'.join([str(c+1) for c in x])))
    plot_geodf.loc[plot_geodf['label'] == '0', 'label'] = 'homeless'
    return plot_geodf


def get_width_ratios(geodf, cc_list, ratio_lgd=0.05, latlon_proj='epsg:4326'):
    '''
    Get the width ratios to pass to `position_axes`.
    '''
    has_lgd = ratio_lgd > 0
    width_ratios = np.ones(len(cc_list) + int(has_lgd))
    for i, cc in enumerate(cc_list):
        cc_mask = geodf.index.str.startswith(cc)
        width, height = geo.calc_shape_dims(geodf.loc[cc_mask],
                                            latlon_proj=latlon_proj)
        width_ratios[i] = width / height
    if has_lgd:
        width_ratios[:-1] = ratio_lgd * (1 - ratio_lgd) / width_ratios[:-1].sum()
    return width_ratios


def position_axes(width_ratios, total_width, total_height=None, ratio_lgd=None):
    '''
    Positions the axes defined by the `width_ratios` such that they do not
    overlap and the total width fits `total_width`.  Uses an algorithm of
    so-called 'rectangle packing'. `total_width` must be an integer.
    '''
    max_ratio = width_ratios.max()
    sum_ratios = width_ratios.sum()
    if max_ratio > 1.2:
        if sum_ratios < 1.6 * max_ratio:
            norm = sum_ratios
        else:
            norm = max_ratio
    else:
        # 4 = max number of countries per line
        norm = sum_ratios / (1 + len(width_ratios) // 4)
    int_widths = np.round(width_ratios * total_width / norm).astype(int)
    int_heights = (int_widths / width_ratios).astype(int)
    sizes = list(zip([int(w) for w in int_widths],
                     [int(h) for h in int_heights]))
    positions = rpack.pack(
        sizes, max_width=total_width, max_height=total_height
    )
    bboxes = np.array([
        [left, bot, w, h]
        for ((left, bot), w, h) in zip(positions, int_widths, int_heights)
    ])
    # ensure legend is on right border:
    last_w = bboxes[-1, 2]
    last_l = bboxes[-1, 0]
    if ratio_lgd is not None and last_l + last_w < total_width:
        same_line = bboxes[:, 1] == bboxes[-1, 1]
        to_shift_left = bboxes[same_line, 0] > last_l

        if to_shift_left.sum() > 1:
            right_margin = total_width - np.max(bboxes[same_line][to_shift_left, 0]
                                                + bboxes[same_line][to_shift_left, 2])
        else:
            right_margin = 0

        if (~to_shift_left).sum() > 1:
            margin_to_closest_left = np.min(
                last_l
                - bboxes[same_line][~to_shift_left, 0][:-1]
                - bboxes[same_line][~to_shift_left, 2][:-1]
            )
        else:
            # else it's the leftmost bbox
            margin_to_closest_left = 0

        to_shift_idc = np.nonzero(same_line)[0][to_shift_left]
        bboxes[to_shift_idc, 0] = (
            bboxes[to_shift_idc, 0]
            - last_w
            - right_margin
            - margin_to_closest_left
        )
        bboxes[-1, 0] = total_width - last_w

    total_height = np.max(bboxes[:, 1] + bboxes[:, 3])
    normed_bboxes = bboxes.astype(float)
    normed_bboxes[:, 0] = normed_bboxes[:, 0] / total_width
    normed_bboxes[:, 2] = normed_bboxes[:, 2] / total_width
    normed_bboxes[:, 1] = normed_bboxes[:, 1] / total_height
    normed_bboxes[:, 3] = normed_bboxes[:, 3] / total_height
    return normed_bboxes, (total_width, total_height)


def joint_cc(geodf, cluster_data, data_dict, cmap=None, show=True,
             **fig_kwargs):
    '''
    Plots cluster maps of different regions described in `data_dict`, with cells
    given in `geodf`, and their associated clusters in `cluster_data`. Adds a
    single legend for all the maps.
    '''
    fig, axes = plt.subplots(ncols=len(data_dict.keys()) + 1, **fig_kwargs)
    plot_geodf = prep_cluster_plot(geodf, cluster_data)
    unique_labels = sorted(plot_geodf['label'].unique())
    nr_cats = len(unique_labels)
    if cmap is None:
        gen_colors_fun = gen_distinct_colors
    else:
        gen_colors_fun = lambda n: list(plt.get_cmap(cmap, n).colors)
    if 'homeless' in unique_labels:
        colors = gen_colors_fun(nr_cats - 1) + [(0.5, 0.5, 0.5, 1)]
    else:
        colors = gen_colors_fun(nr_cats)
    label_color = dict(zip(unique_labels, colors))

    for ax, (cc, reg_dict) in zip(axes[:-1], data_dict.items()):
        xy_proj = reg_dict['xy_proj']
        cc_idx = plot_geodf.index.str.startswith(cc)
        cc_geodf = plot_geodf.loc[cc_idx].to_crs(xy_proj)
        for lab, lab_geodf in cc_geodf.groupby('label'):
            lab_geodf.plot(ax=ax, color=label_color[lab],) #, **plot_kw)
        shape_df = reg_dict['shape_df'].to_crs(xy_proj)
        shape_df.plot(ax=ax, color='none', edgecolor='black')
        ax.set_axis_off()

    cax = axes[-1]
    handles = []
    for l, c in label_color.items():
        # The colours will correspond because groupby sorts by the column by
        # which we group, and we sorted the unique labels.
        handles.append(mlines.Line2D([], [], color=c, marker='o',
                                     linestyle='None', markersize=6, label=l))
    cax.legend(handles=handles)
    cax.set_axis_off()
    if show:
        fig.show()
    return fig, axes


def cluster_level(
    level, regions, figsize=None, cmap=None,
    save_path=None, show=True, fig=None, axes=None, **kwargs
):
    if axes is None:
        fig, axes = plt.subplots(len(regions), figsize=figsize)

    if len(regions) == 1:
        axes = (axes,)

    if cmap is not None:
        level.attr_color_to_labels(cmap=cmap)

    label_color = level.colors
    for ax, reg in zip(axes, regions):
        cc_geodf = reg.cells_geodf.join(level.labels, how='inner')

        for label, label_geodf in cc_geodf.groupby('labels'):
            # Don't put a cmap in kwargs['plot'] because here we use a
            # fixed color per cluster.
            label_geodf.plot(
                ax=ax, color=label_color[label], **kwargs.get('plot', {})
            )

        reg.shape_geodf.plot(
            ax=ax, color='none', edgecolor='black', linewidth=0.5
        )
        ax.set_title(reg.readable)
        ax.set_axis_off()

    fig = ax.get_figure()

    # The colours will correspond because groupby sorts by the column by
    # which we group, and we sorted the unique labels.
    fig = colored_pts_legend(fig, label_color, **kwargs.get('legend', {}))
    if show:
        fig.show()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig, axes


def choropleth(
    plot_series, regions, axes=None, cax=None, cmap=None,
    norm=None, vmin=None, vmax=None, vcenter=None,
    cbar_label=None, null_color='gray', save_path=None, show=True,
    **plot_kwargs
):
    if axes is None:
        fig, axes = plt.subplots(len(regions))

    if len(regions) == 1:
        axes = (axes,)

    if norm is None:
        if vmin is None:
            vmin = plot_series.min()
        if vmax is None:
            vmax = plot_series.max()
        if vcenter is None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    for ax, reg in zip(axes, regions):
        area_gdf = reg.shape_geodf
        area_gdf.plot(ax=ax, color=null_color, edgecolor='none', alpha=0.3)
        plot_df = reg.cells_geodf.join(plot_series, how='inner')
        plot_df.plot(
            column=plot_series.name, ax=ax, norm=norm, cmap=cmap, **plot_kwargs
        )
        area_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
        ax.set_title(reg.readable)
        ax.set_axis_off()

    fig = ax.get_figure()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if cax is None:
        divider = make_axes_locatable(ax)
        # Create an axes on the right side of ax. The width of cax will be 5% of ax
        # and the padding between cax and ax will be fixed at 0.1 inch.
        cax = divider.append_axes('right', size='5%', pad=0.1)

    _ = fig.colorbar(sm, cax=cax, label=cbar_label, shrink=0.8)

    if show:
        fig.show()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, axes
