import pandas as pd
import numpy as np
import scipy.spatial.distance
import scipy.sparse

def gaussian(d, sigma=1):
    return (1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-d.T**2 / (2 * sigma**2))).T


def power_law(d, alpha=1):
    # first column forced to 1
    d[d == 0] = 1
    d = d**-alpha
    # sum of other columns normed to 1.
    d[:, 1:] = (d[:, 1:].T / d[:, 1:].sum(axis=1)).T
    return d


def order_nn(gdf):
    centers = gdf.geometry.centroid
    coords = np.array(list(zip(centers.x, centers.y)))
    d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    ordered_neighbors = np.argsort(d, axis=1)
    nn_ordered_d = np.take_along_axis(d, ordered_neighbors, 1)
    return ordered_neighbors, nn_ordered_d


def count_bw(cell_counts, cells_index, ordered_neighbors, count_th):
    '''
    select neighbors using bandwidth based on counts
    `cells_index` must be sorted
    '''
    cell_token_sums = cell_counts.groupby('cell_id')['count'].sum().reindex(cells_index).fillna(0)
    nn_token_sums = cell_token_sums.values[ordered_neighbors]
    nn_bw_mask = nn_token_sums.cumsum(axis=1) < count_th
    last_nn_in_bw = nn_bw_mask.argmin(axis=1)
    print('min-max number of neighbors in bandwidh:', last_nn_in_bw.min(), last_nn_in_bw.max())
    nr_cells = nn_bw_mask.shape[0]
    col_nr = np.tile(np.arange(nr_cells), (nr_cells, 1))
    nn_bw_mask = ~(col_nr.T <= last_nn_in_bw).T
    return nn_token_sums, nn_bw_mask


def calc_kernel_weights(nn_ordered_d, nn_bw_mask, nn_token_sums,
                        wdist_fun=None, **wdist_fun_kwargs):
    if wdist_fun is None:
        wdist_fun = gaussian
    token_weights = (nn_token_sums.T / nn_token_sums.sum(axis=1)).T
    nn_bw_d = np.ma.masked_array(nn_ordered_d, nn_bw_mask)
    # TODO: kwarg for norm kind?
    max_d_in_bw = nn_bw_d.max(axis=1)
    # If max is zero we just want to keep 0, so we replace the zero by whatever other real number.
    max_d_in_bw[max_d_in_bw == 0] = 1
    normed_nn_bw_d = (nn_bw_d.T / max_d_in_bw).T
    # sigma = nn_bw_d.max(axis=1)
    dist_weights = wdist_fun(normed_nn_bw_d, **wdist_fun_kwargs)
    nn_weights = (dist_weights * token_weights)
    nn_weights = (nn_weights.T / nn_weights.sum(axis=1)).T
    return nn_weights


def get_max_rank(cell_counts_mat, ordered_weights, presence_th=1):
    binary_weights = ordered_weights.copy()
    binary_weights[binary_weights > 0] = 1
    sum_weights = binary_weights.sum(axis=1)
    print(sum_weights.min(), sum_weights.max())
    nbh_sums = cell_counts_mat.dot(binary_weights.T)
    max_rank = (nbh_sums >= presence_th).sum(axis=0).min()
    return max_rank


def get_smoothed_counts(cell_counts, cells_index, ordered_neighbors, nn_weights):
    '''
    From the multiindex Series `cell_counts` giving the word counts by cell,
    return the sparse matrix `cell_counts_mat` of the shape nr_cells x nr_words,
    which has smoothed counts based on `nn_weights`.
    '''
    # cells_index must be sorted
    i_cell_id = cell_counts.index.names.index('cell_id')
    empty_cells = cells_index.difference(cell_counts.index.levels[i_cell_id])
    codes = cell_counts.index.codes
    values = cell_counts['count'].values

    # We copy to unfreeze and be able to increment below.
    i = codes[i_cell_id].copy()
    j = codes[cell_counts.index.names.index('word')]
    ssrt = np.searchsorted(cells_index, empty_cells)
    # For every missing cell we must add a row full of zeros to the sparse
    # matrix, so we must increment the row indices of its non-zero values
    # accordingly:
    for k, s in enumerate(ssrt):
        i[i >= s] += 1
    # We don't provide shape, because even if there's missing data in some
    # cells, we want to know which cells they are, so should add null data to
    # the series for these cells prior to generating the sparse matrix
    cell_counts_mat = scipy.sparse.coo_matrix((values, (i, j))).tocsr()

    values = nn_weights.compressed()
    j = ordered_neighbors[~nn_weights.mask]
    nr_cells = ordered_neighbors.shape[0]
    i = np.tile(np.arange(nr_cells), (nr_cells, 1)).T[~nn_weights.mask]
    ordered_weights = scipy.sparse.coo_matrix(
        (values, (i, j)), shape=(nr_cells, nr_cells)
    ).tocsr()
    # max_rank = get_max_rank(cell_counts_mat, ordered_weights, presence_th=presence_th)
    print('multiplying...')
    cell_counts_mat = ordered_weights.dot(cell_counts_mat)
    return cell_counts_mat
