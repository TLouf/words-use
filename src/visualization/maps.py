import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import rpack
import src.utils.geometry as geo


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


def colored_poly_legend(container, label_color, **lgd_kwargs):
    '''
    Adds to a legend with colored points to a `container`, which can be a plt ax
    or figure. The color of the points and their associated labels are given
    respectively as values and keys of the dict `label_color`.
    '''
    handles = [mpatches.Patch(color=c, label=l) for l, c in label_color.items()]
    kwargs = {**{'handlelength': 1, 'handleheight': 1}, **lgd_kwargs}
    container.legend(handles=handles, **kwargs)
    return container


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


def cluster_level(
    level, regions, figsize=None, cmap=None, show_lgd=True,
    save_path=None, show=True, fig=None, axes=None, **kwargs
):
    '''
    Plot a clustering level, with cells of the regions coloured according to the cluster
    they belong to, shown in a legend. They are not drawn (left in white/transparent) if
    no information is available on their belonging to a cluster.
    '''
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
            ax=ax, color='none', edgecolor='black', linewidth=0.5,
        )
        ax.set_title(reg.readable)
        ax.set_axis_off()

    fig = ax.get_figure()

    if show_lgd:
        lgd_container = ax if len(regions) == 1 else fig
        lgd_kwargs = {**{'loc': 'center right'}, **kwargs.get('legend', {})}
        # The colours will correspond because groupby sorts by the column by
        # which we group, and we sorted the unique labels.
        _ = colored_poly_legend(lgd_container, label_color, **lgd_kwargs)

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
    cbar_kwargs=None, **plot_kwargs
):
    '''
    Make a choropleth map from continuous values given in `plot_series` for some given
    regions. A colorbar will be drawn, either in the given `cax` or to the right of the
    last ax in `axes`. Cells missing in `plot_series` are coloured in `null_color`.
    '''
    if cbar_kwargs is None:
        cbar_kwargs = {}

    if axes is None:
        fig, axes = plt.subplots(len(regions))

    if isinstance(axes, matplotlib.axes.Axes):
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

    if vmin is not None:
        norm.vmin = vmin
    if vmax is not None:
        norm.vmax = vmax
    
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

    cbar = fig.colorbar(sm, cax=cax, label=cbar_label, **cbar_kwargs)

    if show:
        fig.show()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, axes
