import re
import numpy as np
import scipy.stats
import geopandas as geopd
from shapely.geometry import Point
from esda.getisord import G_Local
import src.data.text_process as text_process


class My_G_local(G_Local):
    def __init__(
        self,
        y,
        w,
        transform="R",
        permutations=999,
        star=False,
        keep_simulations=True,
    ):
        y = np.asarray(y).flatten()
        self.n = len(y)
        self.y = y
        self.w = w
        self.w_original = w.transform
        self.w.transform = self.w_transform = transform.lower()
        self.permutations = permutations
        self.star = star
        self.calc()
        self.p_norm = 1 - scipy.stats.norm.cdf(np.abs(self.Zs))
        if permutations:
            self.__crand(keep_simulations)
            if keep_simulations:
                self.sim = sim = self.rGs.T
                self.EG_sim = sim.mean(axis=0)
                self.seG_sim = sim.std(axis=0)
                self.VG_sim = self.seG_sim * self.seG_sim
                self.z_sim = (self.Gs - self.EG_sim) / self.seG_sim
                self.p_z_sim = 1 - scipy.stats.norm.cdf(np.abs(self.z_sim))


def get_cell_word_counts(tweets_df, cells_geodf, places_geodf, cells_in_places,
                         lang, latlon_proj='epsg:4326'):
    '''
    Get the word counts by cell from the tweets in `tweets_df`.
    '''
    is_poi = places_geodf['area'] == 0
    # Start with points
    has_gps = tweets_df['coordinates'].notnull()
    place_is_poi = tweets_df['place_id'].isin(is_poi.loc[is_poi].index)
    is_point = has_gps | place_is_poi
    tweets_pts_df = tweets_df.loc[is_point].copy()
    tweets_pts_df['geometry'] = None
    pt_is_gps = has_gps.loc[is_point]
    # Tweets with POI place:
    pois_cells = geopd.sjoin(
        places_geodf.loc[is_poi], cells_geodf,
        op='within', rsuffix='cell', how='inner')
    tweets_pts_df.loc[~pt_is_gps, 'cell_id'] = (
        tweets_pts_df.loc[~pt_is_gps]
                     .join(pois_cells['cell_id'],
                           on='place_id', how='inner')['cell_id'])
    # Tweets with GPS coordinates:
    tweets_pts_df.loc[pt_is_gps, 'geometry'] = (
        tweets_pts_df.loc[pt_is_gps, 'coordinates']
                     .apply(lambda x: Point(x['coordinates'])))
    tweets_gps_geo = geopd.GeoDataFrame(
        tweets_pts_df.loc[pt_is_gps, 'geometry'], crs=latlon_proj)
    tweets_gps_geo = tweets_gps_geo.to_crs(places_geodf.crs)
    tweets_gps_cells = geopd.sjoin(
        tweets_gps_geo, cells_geodf[['geometry']],
        op='within', rsuffix='cell', how='inner')['index_cell']
    tweets_pts_df.loc[pt_is_gps, 'cell_id'] = tweets_gps_cells

    tweets_pts_df = tweets_pts_df.loc[tweets_pts_df['cell_id'].notnull()]
    tweets_pts_df = text_process.clean(tweets_pts_df, lang=lang)
    #patt = re.compile(r"\b(?P<word>[a-zA-Z\u00C0-\u00FF]+)\b")
    patt = re.compile(r"\b[a-zA-Z\u00C0-\u00FF]+\b")
    cell_counts_from_pts = (
        tweets_pts_df.set_index('cell_id')['filtered_text']
                     .str.findall(patt)
                     .explode()
                     .rename('word')
                     .to_frame()
                     .groupby(['cell_id', 'word'])
                     .size()
                     .rename('count'))

    # Tweets with bounding box place:
    tweets_bbox_df = text_process.clean(tweets_df.loc[~is_point], lang=lang)
    places_counts_df = (
        tweets_bbox_df.set_index('place_id')['filtered_text']
                      .str.findall(patt)
                      .explode()
                      .rename('word')
                      .to_frame()
                      .groupby(['place_id', 'word'])
                      .size()
                      .rename('count'))
    # Other way to do it, which surprisingly turns out to be slower than the
    # above.
    # places_counts_df = (tweets_bbox_df.set_index('place_id')['filtered_text']
    #                                   .str.extractall(patt)
    #                                   .groupby(['place_id', 'word'])
    #                                   .size()
    #                                   .rename('count'))

    intersect_counts = (places_counts_df.to_frame()
                                        .join(cells_in_places, how='inner'))
    intersect_counts['count'] *= intersect_counts['ratio']
    cell_counts_from_bbox = (
        intersect_counts.groupby(['word', 'cell_id'])[['count', 'ratio']].sum())

    cell_counts_from_pts = cell_counts_from_pts.to_frame().assign(ratio=1)
    cell_counts = cell_counts_from_bbox.add(cell_counts_from_pts, fill_value=0)
    cell_counts.loc[cell_counts['ratio'] > 1, 'ratio'] = 1
    return cell_counts


def combine_cell_counts(df1, df2):
    '''
    Combine a pair of `cell_counts`, adding counts together and taking the
    maximum ratio.
    '''
    comb = lambda s1, s2: np.maximum(s1, s2) if s1.name == 'ratio' else s1 + s2
    return df1.combine(df2, comb, fill_value=0)


def agg_chunk_counts(list_cell_counts):
    '''
    From the list of word counts by cells obtained on chunks of raw data
    `list_cell_counts`, returns first cell counts summed over the whole data
    set, and the global counts, aggregating over the whole region. Additionally,
    the words in cell_counts are filtered to exclude probable proper nouns,
    which we wish to ignore for dialect detection.
    '''
    cell_counts = list_cell_counts[0].copy()
    total_nr_tokens = cell_counts['count'].sum()
    for add_cell_counts in list_cell_counts[1:]:
        cell_counts = cell_counts.add(add_cell_counts, fill_value=0)
        total_nr_tokens += add_cell_counts['count'].sum()
    print(f'We have {total_nr_tokens:.0f} tokens.')
    reg_counts = get_reg_counts(cell_counts)
    cell_counts = (
        cell_counts.reset_index()
                   .assign(word=lambda df: df['word'].str.lower())
                   .groupby(['word', 'cell_id'])[['count', 'ratio']]
                   .sum())
    return cell_counts, reg_counts


def get_reg_counts(cell_counts):
    '''
    Get the word counts aggregated over all cells in `cells_count`. For every
    word normalised to its lowered version, count regardless of case and the
    number of time it was not written in its lower form.
    '''
    reg_counts_w_case = cell_counts.groupby('word')[['count']].sum()
    reg_counts_w_case['word_lower'] = reg_counts_w_case.index.str.lower()
    not_lower = reg_counts_w_case.index != reg_counts_w_case['word_lower']
    reg_counts_w_case.loc[not_lower, 'count_upper'] = (
        reg_counts_w_case.loc[not_lower, 'count'])
    # Here we can round and cast to int, because if everything was previously
    # done correctly (and this makes for a nice test), summing over all cells
    # should give an integer, as counts from bbox places should be entirely
    # spread among intersected cells.
    reg_counts = (
        reg_counts_w_case.groupby('word_lower')
                         .sum()
                         .rename_axis('word')
                         .round()
                         .astype(int)
                         .sort_values(by='count', ascending=False))

    cell_counts['word_lower'] = (
        cell_counts.index.get_level_values(level='word').str.lower())
    reg_counts['nr_cells'] = (
        cell_counts.groupby(['word_lower', 'cell_id'])['ratio']
                   .max()
                   .groupby('word_lower')
                   .sum()
                   .rename_axis('word'))
    cell_counts = cell_counts.drop(columns=['word_lower'])
    return reg_counts


def agg_by_lower(raw_cell_counts):
    '''
    Aggregate the counts by (case sensitive) word and by cell in
    `raw_cell_counts` to counts by case-insensitive words by cell.
    '''
    # The ratio does not make any sense if summed and we don't need it anymore.
    cell_counts = (
        raw_cell_counts.reset_index()
                       .assign(word=lambda df: df['word'].str.lower())
                       .groupby(['word', 'cell_id'])[['count']]
                       .sum())
    return cell_counts


def filter_cell_counts(raw_cell_counts, reg_counts, upper_th=1.1, sum_th=1e4,
                       cell_tokens_decade_crit=None, min_nr_cells=0):
    '''
    Filter out rows in cell counts based on multiple criteria. It first filters
    out words considered as proper nouns as they were capitalized more than
    `upper_th` of the time. It also filters out words found in less than
    `min_nr_cells` cells. Finally, we sum the occurences of all words in every
    cell to filter out cells `cell_tokens_decade_crit` decades below the
    geometric average. The default values of the thresholds imply no filtering
    at all.
    '''
    total_nr_tokens = raw_cell_counts['count'].sum()
    upper_mask = reg_counts['count_upper'] / reg_counts['count'] > upper_th
    reg_counts['is_proper'] = upper_mask
    nr_cell_mask = reg_counts['nr_cells'] >= min_nr_cells
    reg_counts['nr_cell_mask'] = nr_cell_mask
    masks = [~upper_mask, nr_cell_mask]
    cell_counts = filter_part_multidx(raw_cell_counts, masks)

    cell_sum = cell_counts.groupby('cell_id')['count'].sum()
    if cell_tokens_decade_crit:
        # For countries containing deserts like Australia, geometric mean can be
        # very low, so take at least the default `sum_th`.
        sum_th = max(10**(np.log10(cell_sum).mean() - cell_tokens_decade_crit),
                     sum_th)
    cell_is_relevant = cell_sum > sum_th
    print(f'Keeping {cell_is_relevant.sum()} cells out of '
            f'{cell_is_relevant.shape[0]} with threshold {sum_th:.2e}')
    cell_counts = filter_part_multidx(cell_counts, [cell_is_relevant])

    filtered_nr_tokens = cell_counts['count'].sum()
    rel_diff = (total_nr_tokens - filtered_nr_tokens) / total_nr_tokens
    print(f'We had {total_nr_tokens:.0f} tokens, and filtering brought it down',
          f'to {filtered_nr_tokens:.0f}, so we lost {100*rel_diff:.3g}%.')
    return cell_counts, reg_counts


def filter_part_multidx(cell_counts, masks):
    '''
    As it's a MultiIndexed frame, can't just .loc, satisfactorily fast solution
    that I found is to use an inner join and keep only the original columns.
    '''
    cols = cell_counts.columns
    filtered_counts = cell_counts.copy()
    for m in masks:
        m_series = m.loc[m].rename('col_to_remove')
        # If nothing to filter out, don't bother joining
        if m_series.shape[0] < m.shape[0]:
            filtered_counts = filtered_counts.join(m_series, how='inner')[cols]
    return filtered_counts


def to_vectors(cell_counts, mask):
    '''
    Transpose count data to matrix, each line corresponding to a cell, each
    column to a word.
    '''
    mask_series = mask.loc[mask].rename('x')
    # reindex is to set the same word order as in the region_counts, so beware
    # that `mask` is indexed in same order as region_counts
    word_vectors = (
        cell_counts.join(mask_series, how='inner')[['count']]
                   .unstack(level='cell_id')
                   .reindex(mask_series.index)
                   .fillna(0)
                   .astype(int)
                   .to_numpy())
    return word_vectors.T


def vec_to_metric(word_counts_vectors, reg_counts, word_vec_var='', w=None):
    '''
    Transforms `word_vectors`, the matrix of cell counts obtained with
    `to_vectors` above, to cell proportions, and a metric given by
    `word_vec_var`, if given. If not, or if it does not match one of the
    implemented metrics, return the  proportions.
    '''
    word_vectors = (word_counts_vectors.T / word_counts_vectors.sum(axis=1)).T

    if word_vec_var == 'normed_freqs':
        reg_distrib = (reg_counts['count'] / reg_counts['count'].sum()).values
        word_vectors -= reg_distrib
        word_vectors = word_vectors / np.abs(word_vectors).max(axis=0)

    elif word_vec_var == 'polar':
        reg_distrib = (reg_counts['count'] / reg_counts['count'].sum()).values
        word_vectors = ((word_vectors - reg_distrib)
                        / (word_vectors + reg_distrib))

    elif word_vec_var == 'Gi_star':
        ## permutations??
        for idx_word in range(word_vectors.shape[1]):
            y = word_vectors[:, idx_word]
            lg_star = My_G_local(y, w, transform='R', star=True, permutations=0)
            word_vectors[:, idx_word] = lg_star.Zs

    elif 'tf-idf' in word_vec_var:
        total_nr_cells = reg_counts['nr_cells'].max()
        doc_freqs = (reg_counts['nr_cells'] / total_nr_cells).values
        if word_vec_var == 'smooth_tf-idf':
            word_vectors = np.log(1 + word_vectors) * np.log(1 + 1/doc_freqs)
        elif word_vec_var == 'raw_tf-idf':
            word_vectors = word_vectors * np.log(1/doc_freqs)

    return word_vectors
