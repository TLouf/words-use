import numpy as np
import matplotlib.pyplot as plt
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