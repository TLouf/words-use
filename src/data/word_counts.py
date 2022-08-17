from __future__ import annotations

import re
import numpy as np
import scipy.stats
import scipy.sparse
from numba import njit, prange
import pandas as pd
import geopandas as geopd
from shapely.geometry import Point
import libpysal
from esda.getisord import G_Local
import ray
import src.data.text_process as text_process
import src.utils.smooth as smooth
import src.utils.parallel as parallel


def get_cell_of_pt_tweets(
    tweets_pts_df, cells_geodf, poi_places_geodf, latlon_proj='epsg:4326'
):
    if tweets_pts_df.shape[0] > 0:
        pt_is_gps = tweets_pts_df['pt_is_gps']
        tweets_pts_df['geometry'] = None

        # Tweets with POI place:
        pois_cells = geopd.sjoin(
            poi_places_geodf, cells_geodf,
            predicate='within', rsuffix='cell', how='inner')
        tweets_pts_df.loc[~pt_is_gps, 'cell_id'] = (
            tweets_pts_df.loc[~pt_is_gps]
                         .join(pois_cells['cell_id'],
                               on='place_id', how='inner')['cell_id'])

        # Tweets with GPS coordinates:
        if pt_is_gps.any():
            sample_coord = tweets_pts_df.loc[pt_is_gps, 'coordinates'].iloc[0]
            if isinstance(sample_coord, list):
                make_pt_fun = Point
            elif isinstance(sample_coord, dict) and 'coordinates' in sample_coord:
                make_pt_fun = lambda x: Point(x['coordinates'])
            else:
                raise ValueError('Wrong coordinates column')

            pt_geoms = tweets_pts_df.loc[pt_is_gps, 'coordinates'].apply(make_pt_fun)
            tweets_gps_geo = geopd.GeoDataFrame(geometry=pt_geoms, crs=latlon_proj)
            tweets_gps_geo = tweets_gps_geo.to_crs(cells_geodf.crs)
            tweets_gps_cells = geopd.sjoin(
                tweets_gps_geo, cells_geodf[['geometry']],
                predicate='within', rsuffix='cell', how='inner'
            )['index_cell']
            tweets_pts_df.loc[pt_is_gps, 'cell_id'] = tweets_gps_cells

    else:
        cols = tweets_pts_df.columns.tolist() + ['geometry', 'cell_id']
        tweets_pts_df = pd.DataFrame(columns=cols)

    return tweets_pts_df


def separate_pts_from_bboxes(tweets_df, cells_geodf, places_geodf, latlon_proj='epsg:4326'):
    is_poi = places_geodf['area'] == 0
    has_gps = tweets_df['coordinates'].notnull()
    place_is_poi = tweets_df['place_id'].isin(is_poi.loc[is_poi].index)
    is_point = has_gps | place_is_poi

    tweets_bbox_df = tweets_df.loc[~is_point].copy()
    
    tweets_pts_df = tweets_df.loc[is_point].copy()
    tweets_pts_df['pt_is_gps'] = has_gps.loc[is_point]
    tweets_pts_df = get_cell_of_pt_tweets(
        tweets_pts_df, cells_geodf, places_geodf.loc[is_poi], latlon_proj=latlon_proj
    )
    tweets_pts_df = tweets_pts_df.loc[tweets_pts_df['cell_id'].notnull()]
    return tweets_bbox_df, tweets_pts_df


def get_cell_word_counts(tweets_df, cells_geodf, places_geodf, cells_in_places,
                         lang, latlon_proj='epsg:4326', lower_all=False):
    '''
    Get the word counts by cell from the tweets in `tweets_df`. `lower_all` to
    get faster execution, at the cost of being able to distinguish proper nouns
    later on.
    '''
    word_patt = re.compile(r'\b[^\W\d_]+?\b')
    tweets_bbox_df, tweets_pts_df = separate_pts_from_bboxes(
        tweets_df, cells_geodf, places_geodf, latlon_proj=latlon_proj
    )
    # Start with points
    tweets_pts_df = text_process.clean(tweets_pts_df, lang=lang)
    tweets_pts_df = tweets_pts_df.set_index('cell_id')['filtered_text']
    if lower_all:
        tweets_pts_df = tweets_pts_df.str.lower()
    cell_counts_from_pts = (
        tweets_pts_df.str.findall(word_patt)
                     .explode()
                     .rename('word')
                     .to_frame()
                     .groupby(['word', 'cell_id'])
                     .size()
                     .rename('count'))
    # Tweets with bounding box place:
    tweets_bbox_df = text_process.clean(tweets_bbox_df, lang=lang)
    tweets_bbox_df = tweets_bbox_df.set_index('place_id')['filtered_text']
    if lower_all:
        tweets_bbox_df = tweets_bbox_df.str.lower()
    places_counts_df = (
        tweets_bbox_df.str.findall(word_patt)
                      .explode()
                      .rename('word')
                      .to_frame()
                      .groupby(['place_id', 'word'])
                      .size()
                      .rename('count'))
    # Other way to do it, which surprisingly turns out to be slower than the
    # above.
    # places_counts_df = (tweets_bbox_df.str.extractall(patt)
    #                                   .groupby(['place_id', 'word'])
    #                                   .size()
    #                                   .rename('count'))
    print(f'{tweets_bbox_df.shape[0]} tweets with bbox places and '
          f'{tweets_pts_df.shape[0]} tweets with coords remaining after text cleaning.')

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
        reg_counts_w_case.loc[not_lower, 'count']
    )
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
                         .sort_values(by='count', ascending=False)
    )

    cell_counts['word_lower'] = (
        cell_counts.index.get_level_values(level='word').str.lower()
    )
    reg_counts['nr_cells'] = (
        cell_counts.groupby(['word_lower', 'cell_id'])['ratio']
                   .max()
                   .groupby('word_lower')
                   .sum()
                   .rename_axis('word')
    )
    cell_counts = cell_counts.drop(columns=['word_lower'])
    return reg_counts, cell_counts


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


def calc_first_word_masks(reg_counts, upper_th=1, min_nr_cells=0):
    upper_mask = reg_counts['count_upper'] / reg_counts['count'] > upper_th
    reg_counts['is_proper'] = upper_mask
    nr_cell_mask = reg_counts['nr_cells'] >= min_nr_cells
    reg_counts['nr_cell_mask'] = nr_cell_mask
    return reg_counts


def filter_cell_counts(raw_cell_counts, masks, sum_th=1e4,
                       cell_tokens_decade_crit=None):
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
    return cell_counts


def filter_part_multidx(cell_counts, masks):
    '''
    As it's a MultiIndexed frame, can't just .loc, satisfactorily fast solution that I
    found is to use an inner join and keep only the original columns. Tested using
    `.loc` or `.reindex`, but this inner join method turns out to be the fastest, by
    far. `.reindex` is better in that it modifies the `.index.levels[i_lvl]` values
    though, while the join does not.
    '''
    cols = cell_counts.columns
    filtered_counts = cell_counts.copy()
    for m in masks:
        m_series = m.loc[m].rename('col_to_remove')
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


@njit(parallel=True)
def get_top_words_count(data, indices, indptr, max_rank):
    # cells x words
    nrows = indptr.shape[0] - 1
    print(nrows)
    col_top_idc = np.empty((nrows, max_rank), dtype=np.int32)
    for i in prange(nrows):
        row_idc = indices[indptr[i]: indptr[i+1]]
        row_data = data[indptr[i]: indptr[i+1]]
        col_top_idc[i, :] = row_idc[row_data.argsort()[-max_rank:]]
    top_uidc = np.unique(col_top_idc)

    word_counts_vectors = np.zeros((nrows, top_uidc.shape[0]), dtype=np.float64)
    for i in prange(nrows):
        row_idc = indices[indptr[i]: indptr[i+1]]
        list_mat_iloc = []
        list_data_iloc = []
        # The following works because top_uidc and row_idc are sorted.
        m = 0
        for k, uid in enumerate(top_uidc):
            if uid == row_idc[m]:
                list_mat_iloc.append(k)
                list_data_iloc.append(m)
                m += 1
        mat_iloc = np.asarray(list_mat_iloc)
        data_iloc = np.asarray(list_data_iloc)
        row_data = data[indptr[i]: indptr[i+1]][data_iloc]
        word_counts_vectors[i][mat_iloc] = row_data

    return word_counts_vectors, top_uidc


@njit(parallel=True)
def get_top_words_idc(data, indices, indptr, max_rank):
    # cells x words
    nrows = indptr.shape[0] - 1
    print(nrows)
    col_top_idc = np.empty((nrows, max_rank), dtype=np.int32)
    for i in prange(nrows):
        row_idc = indices[indptr[i]: indptr[i+1]]
        row_data = data[indptr[i]: indptr[i+1]]
        col_top_idc[i, :] = row_idc[row_data.argsort()[-max_rank:]]
    top_uidc = np.unique(col_top_idc)

    return top_uidc


def rank_filter(cell_counts_mat, max_rank):
    # Take max_rank of the top words from each cell. Does not work properly if
    # max_rank == 0, but well that shouldn't happen.
    word_counts_vectors = cell_counts_mat.tocsr()
    data = word_counts_vectors.data
    indices = word_counts_vectors.indices
    indptr = word_counts_vectors.indptr
    word_idc = get_top_words_idc(data, indices, indptr, max_rank)

    word_counts_vectors = word_counts_vectors.toarray()[:, word_idc]
    print(word_counts_vectors.shape) #, word_idc)
    return word_counts_vectors, word_idc


def disordered_sparse_series_to_coo(raw_series, i_index, j_index):
    '''
    MultiIndex of `series` can be potentially partial, and the matrix' indices
    should match the ones given in `i_index` and `j_index`, which may have
    values which are not in the MultiIndex and not ordered the same way.
    '''
    masks = [pd.Series(True, index=idx) for idx in (j_index, i_index)]
    series = filter_part_multidx(raw_series.to_frame(), masks)[raw_series.name]

    series_iloc_i_level = series.index.names.index(i_index.name)
    series_i_level = series.index.levels[series_iloc_i_level]
    join_series = pd.Series(range(len(series_i_level)), index=series_i_level, name='cc_codes')
    order_df = pd.DataFrame({'gc_codes': range(len(i_index))}, index=i_index)
    # join on words and get corresponding code in global_counts' and
    # cell_counts' indices.
    order_df = order_df.join(join_series, how='left')
    nr_missing_cells = order_df['cc_codes'].isnull().sum()
    print(f"{nr_missing_cells} cells were not found in cell_counts.")
    i_order = order_df.sort_values(by='cc_codes')['gc_codes'].values
    i = i_order[series.index.codes[series_iloc_i_level]]
    
    lvl_j_index = series.index.names.index(j_index.name)
    series_j_level = series.index.levels[lvl_j_index]
    join_series = pd.Series(range(len(series_j_level)), index=series_j_level, name='cc_codes')
    order_df = pd.DataFrame({'gc_codes': range(len(j_index))}, index=j_index)
    order_df = order_df.join(join_series, how='left')
    nr_missing_words = order_df['cc_codes'].isnull().sum()
    print(f"{nr_missing_words} words were not found in cell_counts.")
    j_order = order_df.sort_values(by='cc_codes')['gc_codes'].values
    j = j_order[series.index.codes[lvl_j_index]]

    values = series.values

    sparse_mat = scipy.sparse.coo_matrix(
        (values, (i, j)), shape=(len(i_index), len(j_index))
    )
    return sparse_mat


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
            self._G_Local__crand(keep_simulations)
            if keep_simulations:
                self.sim = sim = self.rGs.T
                self.EG_sim = sim.mean(axis=0)
                self.seG_sim = sim.std(axis=0)
                self.VG_sim = self.seG_sim * self.seG_sim
                self.z_sim = (self.Gs - self.EG_sim) / self.seG_sim
                self.p_z_sim = 1 - scipy.stats.norm.cdf(np.abs(self.z_sim))

def calc_G_vectors(word_props_vecs, w):
    word_Zs_vecs = np.zeros_like(word_props_vecs)
    for word_idx, vec in enumerate(word_props_vecs):
        lg_star = My_G_local(vec, w, transform='R', star=True, permutations=0)
        word_Zs_vecs[word_idx] = lg_star.Zs
    return word_Zs_vecs

def calc_G(word_props_vecs, w, permutations=999):
    # word_Zs_vecs = np.zeros_like(word_props_vecs)
    attrs = ['p_sim', 'z_sim', 'p_norm', 'Zs']
    lg_star_dict = {key: [] for key in attrs}
    for word_idx, vec in enumerate(word_props_vecs):
        lg_star = My_G_local(vec, w, transform='R', star=True, permutations=permutations)
        for key in attrs:
            lg_star_dict[key].append(getattr(lg_star, key))
        # lg_star_list.append(lg_star.p_sim, lg_star.z_sim, lg_star.p_norm, lg_star.Zs)
        # word_Zs_vecs[word_idx] = lg_star.Zs
        # p_values[word_idx] = lg_star
    return lg_star_dict


def vec_to_metric(word_counts_vectors, whole_reg_counts, word_vec_var='',
                  cell_sums=None, global_sum=None, w=None):
    '''
    Transforms `word_vectors`, the matrix of cell counts obtained with
    `to_vectors` above, to cell proportions, and a metric given by
    `word_vec_var`, if given. If not, or if it does not match one of the
    implemented metrics, return the  proportions.
    '''
    reg_counts = whole_reg_counts.loc[whole_reg_counts['cell_counts_mask']]
    if global_sum is None:
        global_sum = reg_counts['count'].sum()

    if cell_sums is None:
        cell_sums = word_counts_vectors.sum(axis=1)

    word_vectors = (word_counts_vectors.T / cell_sums).T

    if word_vec_var == 'normed_freqs':
        reg_distrib = (reg_counts['count'] / global_sum).values
        word_vectors = word_vectors - reg_distrib
        word_vectors = word_vectors / np.abs(word_vectors).max(axis=0)

    elif word_vec_var == 'polar':
        reg_distrib = (reg_counts['count'] / global_sum).values
        word_vectors = ((word_vectors - reg_distrib)
                        / (word_vectors + reg_distrib))

    elif word_vec_var.startswith('Gi_star'):
        refs = parallel.split_task(calc_G_vectors, word_vectors.T, w)
        word_vectors = np.concatenate(ray.get(refs), axis=0).T

    elif word_vec_var == 'z_score':
        reg_distrib = (reg_counts['count'] / global_sum).values
        diff = word_vectors - reg_distrib
        n = word_vectors.shape[0]
        std = np.sqrt(np.sum(diff**2, axis=0) / (n-1))
        word_vectors = diff / std

    elif 'tf-idf' in word_vec_var:
        total_nr_cells = reg_counts['nr_cells'].max()
        doc_freqs = (reg_counts['nr_cells'] / total_nr_cells).values
        if word_vec_var == 'smooth_tf-idf':
            word_vectors = np.log(1 + word_vectors) * np.log(1 + 1/doc_freqs)
        elif word_vec_var == 'raw_tf-idf':
            word_vectors = word_vectors * np.log(1/doc_freqs)

    return word_vectors


_WORD_COUNTS_VEC_ATTR = [
    'nr_tokens_bw', 'presence_th', 'max_word_rank',
    'smooth_wdist_fun', 'smooth_wdist_fun_kwargs'
]
class WordCountsVectors(np.ndarray):
    '''
    np.ndarray subclass to store the parameters relative to its calculation.
    '''
    def __new__(
        cls,
        input_array: np.ndarray,
        cell_sums: np.ndarray | None = None,
        nr_tokens_bw: float | None = None,
        presence_th: float | None = None,
        max_word_rank: float | None = None,
        smooth_wdist_fun: callable | None = None,
        smooth_wdist_fun_kwargs: dict | None = None,
    ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.cell_sums = cell_sums
        obj.nr_tokens_bw = nr_tokens_bw
        obj.presence_th = presence_th
        obj.max_word_rank = max_word_rank
        obj.smooth_wdist_fun = smooth_wdist_fun
        if smooth_wdist_fun_kwargs is None:
            smooth_wdist_fun_kwargs = {}
        obj.smooth_wdist_fun_kwargs = smooth_wdist_fun_kwargs
        # Finally, we must return the newly created object:
        return obj


    def __array_finalize__(self, obj):
        if obj is None: return
        for attr in _WORD_COUNTS_VEC_ATTR:
            setattr(self, attr, getattr(obj, attr, None))
        self.cell_sums = getattr(obj, 'cell_sums', None)


    @classmethod
    def from_lang(cls, lang, word_mask_col=None, **init_kwargs):
        cell_counts = lang.get_cell_counts()
        global_counts = lang.global_counts
        cell_sums = init_kwargs.get('cell_sums')

        if init_kwargs.get('nr_tokens_bw') is not None:
            cells_index = lang.relevant_cells
            ordered_neighbors, nn_ordered_d = smooth.order_nn(
                lang.cells_geodf.loc[cells_index]
            )
            nn_token_sums, nn_bw_mask = smooth.count_bw(
                cell_counts, cells_index, ordered_neighbors, init_kwargs.get('nr_tokens_bw')
            )
            wdist_fun = init_kwargs.get('smooth_wdist_fun')
            wdist_fun_kwargs = init_kwargs.get('smooth_wdist_fun_kwargs', {})
            nn_weights = smooth.calc_kernel_weights(
                nn_ordered_d, nn_bw_mask, nn_token_sums,
                wdist_fun=wdist_fun, **wdist_fun_kwargs
            )
            cell_counts_mat = smooth.get_smoothed_counts(
                cell_counts, cells_index, ordered_neighbors, nn_weights,
            )
            # cell_counts_mat is nr_cells x nr_words, both taken from
            # cell_counts (so there may be fewer words than in global_counts,
            # and they are ordered alphabetically).
            presence_th = init_kwargs.get('presence_th', 1)
            max_rank = (cell_counts_mat >= presence_th).sum(axis=1).min()
            print(f'done, max_rank: {max_rank}')
            # if cell_sums is None:
            cell_sums = np.asarray(cell_counts_mat.sum(axis=1)).flatten()

            word_counts_vectors, word_idc = rank_filter(
                cell_counts_mat, max_rank
            )
            print(word_idc)
            # join_series = pd.Series(0, name='tmp', index=word_idx)
            ordered_words = global_counts.loc[global_counts['cell_counts_mask']].index.argsort()
            # global_counts['cell_counts_mask'] = False
            ordered_words_in_mat = ordered_words[word_idc]
            # TODO: parametrise this 1e4
            # th_idx = (global_counts['count'] > 1e4).argmin()
            # rows_to_keep = ordered_words_in_mat[ordered_words_in_mat < th_idx]
            # # join_series = pd.Series(True, name='cell_counts_mask', index=word_idx)
            global_counts['cell_counts_mask'] = False
            col_idc = global_counts.columns.get_loc('cell_counts_mask')
            global_counts.iloc[ordered_words_in_mat, col_idc] = True
            # # global_counts = global_counts.join(join_series).fillna(False)

            # Reorder cols of word_counts_vectors to match ordering of
            # global_counts.
            idx_to_reorder = ordered_words_in_mat.argsort()
            # nr_keep = (ordered_words_in_mat < th_idx).sum()
            # # nr_keep = global_counts['cell_counts_mask'].sum()
            word_counts_vectors = word_counts_vectors[:, idx_to_reorder]
            # word_counts_vectors = word_counts_vectors[:, :nr_keep]
            array = word_counts_vectors

        elif word_mask_col is not None:
            array = disordered_sparse_series_to_coo(
                cell_counts['count'],
                lang.relevant_cells,
                global_counts.loc[global_counts[word_mask_col]].index
            ).toarray()

        else:
            # TODO: deprecate
            global_counts['cell_counts_mask'] = False
            col = global_counts.columns.get_loc('cell_counts_mask')
            max_word_rank = int(init_kwargs['max_word_rank'])
            global_counts.iloc[:max_word_rank, col] = True
            array = disordered_sparse_series_to_coo(
                cell_counts['count'],
                lang.relevant_cells,
                global_counts.loc[global_counts['cell_counts_mask']].index
            ).toarray()

        
        if cell_sums is None:
            cell_sums = (
                cell_counts.groupby('cell_id')['count']
                            .sum()
                            .reindex(lang.relevant_cells, fill_value=0)
                            .values
            )
        init_kwargs['cell_sums'] = cell_sums
        return cls(array, **init_kwargs)


    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in _WORD_COUNTS_VEC_ATTR
            if getattr(self, attr) is not None
        }


_WORD_VEC_ATTR = [
    'word_vec_var', 'spatial_weights_class', 'spatial_weights_kwargs',
    'var_th', 'z_th', 'p_th'
]
class WordVectors(np.ndarray):
    def __new__(
        cls,
        input_array: np.ndarray,
        word_vec_var: str = '',
        spatial_weights_class: libpysal.weights.W | None = None,
        spatial_weights_kwargs: dict | None = None,
        var_th: float | None = None,
        z_th: float | None = None,
        p_th: float | None = None,
    ):
        obj = np.asarray(input_array).view(cls)
        obj.word_vec_var = word_vec_var
        obj.spatial_weights_class = spatial_weights_class
        if spatial_weights_kwargs is None:
            spatial_weights_kwargs = {}
        obj.spatial_weights_kwargs = spatial_weights_kwargs
        obj.var_th = var_th
        obj.z_th = z_th
        obj.p_th = p_th
        return obj


    def __array_finalize__(self, obj):
        if obj is None: return
        for attr in _WORD_VEC_ATTR:
            setattr(self, attr, getattr(obj, attr, None))


    @classmethod
    def from_lang(cls, lang, mask_col='cell_counts_mask', **init_kwargs):
        word_counts_vectors = lang.word_counts_vectors

        kwargs = {
            'word_vec_var': init_kwargs.get('word_vec_var', ''),
            'cell_sums': word_counts_vectors.cell_sums,
            'global_sum': lang.global_counts['count'].sum()
        }

        if kwargs['word_vec_var'].startswith('Gi_star'):
            w_class = init_kwargs['spatial_weights_class']
            w_kwargs = init_kwargs.get('spatial_weights_kwargs', {})
            w = w_class.from_dataframe(
                lang.cells_geodf.loc[lang.relevant_cells], **w_kwargs
            )
            kwargs['w'] = libpysal.weights.fill_diagonal(w)
            kw_str = ', '.join(f'{key}={value}' for key, value in w_kwargs.items())
            init_kwargs['word_vec_var'] = f"Gi_star(w={w_class.__name__}({kw_str}))"

        array = vec_to_metric(
            word_counts_vectors,
            lang.global_counts.loc[lang.global_counts[mask_col]],
            **kwargs
        )

        return cls(array, **init_kwargs)


    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in _WORD_VEC_ATTR
            if getattr(self, attr) is not None
        }


    def filter(self, lang):
        cell_counts_mask = lang.global_counts['cell_counts_mask']

        if self.var_th is None and self.z_th is None and self.p_th is None:
            lang.global_counts['is_regional'] = cell_counts_mask
            return self

        elif self.var_th is None:
            # assumes moran has been done
            is_regional = (
                (lang.global_counts['z_value'] > self.z_th)
                & (lang.global_counts['p_value'] < self.p_th)
            )

        else:
            mask_index = cell_counts_mask.loc[cell_counts_mask].index
            var = self.var(axis=0)
            mean = self.mean(axis=0)
            mask_values = var > self.var_th
            is_regional = pd.DataFrame({'is_regional': mask_values,
                                        'var': var,
                                        'mean': mean},
                                       index=mask_index)

        if 'is_regional' in lang.global_counts.columns:
            dropped_cols = is_regional.columns
            lang.global_counts = lang.global_counts.drop(columns=dropped_cols)

        lang.global_counts = lang.global_counts.join(is_regional)
        lang.global_counts['is_regional'] = (
            lang.global_counts['is_regional'].fillna(False))
        mask = lang.global_counts['is_regional'].loc[cell_counts_mask].values
        self = self[:, mask].copy()
        nr_kept = self.shape[1]
        print(f'Keeping {nr_kept} words out of {mask.shape[0]}')

        return self
