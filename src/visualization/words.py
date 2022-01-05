import matplotlib.pyplot as plt
import wordcloud


def make_word_cloud(word_weights, figsize, save_path=None, dpi=300):
    '''
    Plot a word cloud with the words given by the keys of `word_weights`, and
    their relative sizes by its values.
    '''
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    width, height = fig.dpi_scale_trans.transform(figsize)
    cloud_kwargs = {
        'background_color': 'white', 'scale': width/600, 'colormap': 'Dark2',
        'width': 600, 'height': 300
    }
    cloud = wordcloud.WordCloud(**cloud_kwargs).generate_from_frequencies(word_weights)
    _ = ax.imshow(cloud, interpolation='bilinear')
    ax.set_axis_off()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        
    return fig, ax
