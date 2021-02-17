import re
import numpy as np
import pandas as pd

def distribs_extract(list_fips, freq_file_format):
    '''
    Reads all data files corresponding to `freq_file_format`, cleans them and
    calculate global counts.
    '''
    counties_dict = {}
    global_words = pd.DataFrame({'count': [], 'nr_counties': []},
                                index=pd.Index([], name='word'))
    for i, fips in enumerate(list_fips):
        print(f'{i+1} / {len(list_fips)}', end='\r')
        filename = str(freq_file_format).format(fips)
        word_counts = pd.read_table(filename, names=['word', 'count'],
                                    index_col=0, keep_default_na=False)
        # remove characters outside of the range of the English language + all
        # latin accents
        pattern = re.compile("[^a-zA-Z\u00C0-\u00FF]")
        word_counts.index = word_counts.index.str.replace(pattern, '')
        word_counts = word_counts.loc[word_counts.index != '']
        word_counts = word_counts.groupby('word', sort=False)['count'].sum()
        # rather do a set intersection??
        global_words = global_words.join(word_counts.loc[word_counts > 5],
                                         rsuffix='_cnt', how='outer')
        is_in_county = global_words['count_cnt'].notnull()
        global_words['nr_counties'] = global_words['nr_counties'].fillna(0)
        global_words.loc[is_in_county, 'nr_counties'] += 1
        global_words[['count', 'count_cnt']] = global_words[['count', 'count_cnt']].fillna(0)
        global_words['count'] += global_words['count_cnt']
        global_words = global_words.drop(columns=['count_cnt'])
        grp_rank_counts = word_counts.unique()
        min_count_mask = grp_rank_counts > 0
        grp_rank_counts = grp_rank_counts / np.sum(grp_rank_counts)
        ranks = np.arange(1, len(grp_rank_counts[min_count_mask])+1)
        log_ranks = np.log(ranks)
        normed_log_ranks = log_ranks / np.max(log_ranks)
        freq_distrib = word_counts.value_counts().rename('freq')
        freq_distrib /= freq_distrib.sum()
        counties_dict[fips] = {
            'word_counts': word_counts, 'grp_rank_counts': grp_rank_counts,
            'normed_log_ranks': normed_log_ranks, 'freq_distrib': freq_distrib}
    return global_words, counties_dict
