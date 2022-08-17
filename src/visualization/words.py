import numpy as np
import matplotlib.pyplot as plt
import wordcloud


def cloud(word_weights, figsize=None, save_path=None, dpi=300,
          fig=None, ax=None, **kwargs):
    '''
    Plot a word cloud with the words given by the keys of `word_weights`, and
    their relative sizes by its values.
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    fig = ax.get_figure()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # width, height = fig.dpi_scale_trans.transform(fig.get_size_inches())
    cloud_kwargs = {
        'background_color': 'white', 'scale': width/600, 'colormap': 'Dark2',
        'width': int(width), 'height': int(height), **kwargs
    }
    cloud = wordcloud.WordCloud(**cloud_kwargs).generate_from_frequencies(dict(word_weights))
    _ = ax.imshow(cloud, interpolation='bilinear')
    ax.set_axis_off()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


def stem(word_weights, orientation='horizontal', figsize=None, save_path=None,
         fig=None, ax=None):
    # word_weights is series
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    fig = ax.get_figure()

    y_pos = np.arange(len(word_weights))
    heads = word_weights.values
    s = ax.stem(y_pos, heads, orientation=orientation, basefmt='none')
    ax.set_yticks(y_pos, labels=word_weights.index.values)
    ax.invert_yaxis()
    ax.set_xlabel(word_weights.name)
    ax.grid(True)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax
