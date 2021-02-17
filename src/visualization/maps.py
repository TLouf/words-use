import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

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
    proj_cnt = pd.Series(word_vectors[:, iloc_word],
                         index=valid_geos, name='plot')
    return basic(geodf, proj_cnt, **basic_kwargs)


def clusters(geodf, cnt_clusters, valid_cnt, **plot_kw):
    plot_series = pd.Series(cnt_clusters + 1,
                            name='hierarc', index=valid_cnt)
    return basic(geodf, plot_series, categorical=True, **plot_kw)


def overlap_clusters(geodf, cluster_dict, valid_cnt,
                     fig=None, ax=None, **plot_kw):
    '''
    Plot a cluster map with a legend.
    '''
    if ax is None:
        fig, ax = plt.subplots(1)
    plot_geodf = prep_cluster_plot(geodf, cluster_dict, valid_cnt)
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


def prep_cluster_plot(geodf, cluster_data, valid_cnt):
    '''
    Prepares the GeoDataFrame `geodf` to be passed to `overlap_clusters` or
    `interative.clusters`, by adding a column for the cluster labels, allowing
    for overlapping clusters.
    '''
    if isinstance(cluster_data, np.ndarray):
        cnt_dict = {fips: [clust] for fips, clust in zip(valid_cnt, cluster_data)}
    elif isinstance(cluster_data, dict):
        if len(cluster_data) == len(valid_cnt):
            cnt_dict = dict(zip(valid_cnt, cluster_data.values()))
        else:
            # revert dict cluster: [counties] to cnt: [clusters]
            cnt_dict = {fips: [] for fips in valid_cnt}
            for cluster, counties in cluster_data.items():
                for i_cnt in counties:
                    cnt_dict[valid_cnt[i_cnt]].append(cluster)
    else:
        raise TypeError('''cluster_data must either be an array of cluster
                        labels (as is the case for the result from hierarchical
                        clustering), or a dictionary mapping clusters to a list
                        of counties, or counties to a list of clusters''')

    plot_geodf = geodf.join(pd.Series(cnt_dict, name='clusters'), how='inner')
    plot_geodf['label'] = (
        plot_geodf['clusters'].apply(lambda x: '+'.join([str(c+1) for c in x])))
    plot_geodf.loc[plot_geodf['label'] == '0', 'label'] = 'homeless'
    return plot_geodf
