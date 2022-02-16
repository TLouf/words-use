from dataclasses import dataclass, field, asdict, InitVar
from pathlib import Path
from typing import Union, List, Optional, Callable
from itertools import chain
import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance
import pandas as pd
import esda
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import graph_tool.all as gt
import src.data.word_counts as word_counts
import src.visualization.maps as map_viz
import src.visualization.eval as eval_viz
from dotenv import load_dotenv
load_dotenv()

OSLOM_DIR = Path(os.environ['OSLOM_DIR'])

def gen_oslom_res_path(data_path, oslom_opt_params=None, suffix=''):
    '''
    Generate a path where to save the results of OSLOM, based on the data file
    on which it will be run and with which parameters.
    '''
    parent_path = data_path.parent
    data_filename = data_path.parts[-1]
    data_str = data_filename.split('.')[0]
    params_str = ''.join([s.replace(' ', '=') for s in oslom_opt_params])
    oslom_res_path = parent_path / f"res_{data_str}{params_str}{suffix}"
    return oslom_res_path


def gen_net_file_path(net_file_path_fmt, decomposition, metric, transfo=None):
    if transfo is None:
        transfo_str = 'inverse'
    else:
        transfo_str = transfo.__name__
    # Remove parentheses for Oslom, couldn't make it parse file names with
    # parentheses correctly.
    decomposition_str = str(decomposition).replace('(', '-').replace(')', '')
    oslom_net_file_path = Path(
        str(net_file_path_fmt).format(decomposition=decomposition_str,
                                      metric=metric, transfo_str=transfo_str)
        )
    return oslom_net_file_path


def run_oslom(oslom_dir, data_path, res_path=None, oslom_opt_params=None,
              directional=False):
    '''
    Run the compiled OSLOM located in `oslom_dir` on the network data contained
    in `data_path`, saving the results in `res_path`.
    '''
    if oslom_opt_params is None:
        oslom_opt_params = []
    if res_path is None:
        res_path = gen_oslom_res_path(
            data_path, oslom_opt_params=oslom_opt_params)
    dir_prefix = (not directional) * 'un'
    oslom_exec_path = oslom_dir / f'oslom_{dir_prefix}dir'
    # Beware data_path and res_path must be correct file names for bash, putting
    # them in quotation marks raises problems in OSLOM, which I couldn't fix.
    cmd_list = [str(oslom_exec_path), '-f',
                str(data_path), '-o', str(res_path), '-w'] + oslom_opt_params
    f = open(data_path.parent / data_path.name.replace('.dat', '.log'), 'w')
    p = subprocess.Popen(cmd_list, stdout=f, stderr=f)
    return p


def read_oslom_res(oslom_res_path):
    '''
    Read the tp files containing the results of an OSLOM run into a dictionary
    `cluster_dict` containing, for each returned level, a dictionary with the
    cluster labels as keys and the lists of contained counties as values. The
    key -1 is reserved for singletons.
    '''
    cluster_dict = {}
    for file_path in chain(oslom_res_path.glob('tp[1-9]'),
                           oslom_res_path.glob('tp')):
        lvl_match = re.search("tp([1-9]+)", str(file_path))
        if lvl_match is None:
            lvl = 0
        else:
            lvl = int(lvl_match.groups()[0])
        cluster_dict[lvl] = {-1: []}
        print(f'- level {lvl}')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        pattern = "#module ([0-9]+) size: ([0-9]+) bs: (.+)"
        for i in range(len(lines) // 2):
            mod_line = lines[2*i]
            ids_line = lines[2*i+1]
            matches = re.search(pattern, mod_line).groups()
            clust_nr = int(matches[0])
            mod_cnties = ids_line.split(' ')
            if len(mod_cnties[:-1]) > 1:
                print(f'cluster {clust_nr}, size: {matches[1]}, p-value: ',
                      matches[2])
                cluster_dict[lvl][clust_nr] = [int(c) for c in mod_cnties[:-1]]
            # else it's noise, and there's just one county in the list
            else:
                cnt_id = int(mod_cnties[0])
                cluster_dict[lvl][-1].append(cnt_id)
    if len(cluster_dict) == 0:
        raise FileNotFoundError(f'There is no result in {oslom_res_path}.')
    return cluster_dict


def run_sbm(data_path, rec_types=None, nr_merge_split=10):
    if rec_types is None:
        rec_types = ["real-normal"]
    G = gt.graph_tool.load_graph_from_csv(
        str(data_path), strip_whitespace=False, csv_options={'delimiter': ' '},
        eprop_names=['weight'], eprop_types=['float'],)
    state = gt.minimize_nested_blockmodel_dl(
        G, state_args=dict(recs=[G.ep.weight], rec_types=rec_types))
    # L1 = state.entropy()

    # # improve solution with merge-split
    # impr_state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4, sampling=True)
    # for _ in range(nr_merge_split):
    #     _ = impr_state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
    # L2 = state.entropy()
    # print(L2 - L1)
    return state


def get_clusters_agg(cutree):
    '''
    From a matrix (n_samples x n_levels), returns a matrix (n_levels x
    max_nr_clusters) giving the assignment of the lowest level's clusters at
    higher levels, thus showing which clusters get aggregated with which at each
    aggregation step.
    '''
    n_lvls = cutree.shape[1]
    levels_x_clust = np.zeros((n_lvls, n_lvls + 1), dtype=int)
    levels_x_clust[-1, :] = np.arange(0, n_lvls + 1)
    for i in range(n_lvls-2, -1, -1):
        lvl_clusts = cutree[:, i]
        lower_lvl_clusts = cutree[:, i+1]
        # For every cluster in the lower level,
        for clust in np.unique(lower_lvl_clusts):
            # we select the higher level cluster to which it belongs. Because we
            # started from the less aggregated level, all members of that
            # cluster will belong to the same cluster in the more aggregated
            # level, so we take the higher level cluster of the first one.
            agg_lvl = lvl_clusts[lower_lvl_clusts == clust][0]
            levels_x_clust[i, :][levels_x_clust[i+1, :] == clust] = agg_lvl
    levels_x_clust += 1
    return levels_x_clust


def chunk_moran(list_i, word_vectors, contiguity):
    moran_dict = {'I': [], 'z_value': [], 'p_value': []}
    for i in list_i:
        y = word_vectors[:, i]
        mi = esda.moran.Moran(y, contiguity)
        moran_dict['I'].append(mi.I)
        moran_dict['z_value'].append(mi.z_sim)
        moran_dict['p_value'].append(mi.p_sim)
    return moran_dict


def broken_stick(n_components):
    var = 1 / n_components * np.ones(n_components)
    for i in range(2, n_components+1):
        var[-i] = var[-i+1] + 1 / (n_components + 1 - i)
    return var / n_components


@dataclass
class Clustering:
    source_data: InitVar[Union[dict, np.ndarray]]
    cells_ids: InitVar[np.ndarray]
    method_repr: str
    method_obj: Optional[Callable] = None
    method_args: Optional[list] = None
    method_kwargs: Optional[dict] = None
    labels: pd.Series = field(init=False)
    colors: Optional[dict] = None
    nr_clusters: Optional[int] = None
    prop_homeless: Optional[float] = None
    kwargs_str: Optional[str] = None
    score: Optional[float] = None

    def __post_init__(self, source_data, cells_ids):
        self.cell_dict = self.get_cell_dict(source_data, cells_ids)
        self.init_labels()
        
        if self.colors is None:
            self.attr_color_to_labels()
            
        if self.kwargs_str is None:
            self.kwargs_str = '_params=({})'.format(
                '_'.join([f'{key}={value}'
                          for key, value in self.method_kwargs.items()]))


    def __str__(self):
        self_dict = asdict(self)
        exclude_keys = ['labels', 'colors']
        return str({key: value
                    for key, value in self_dict.items()
                    if key not in exclude_keys})


    def __repr__(self):
        self_dict = asdict(self)
        exclude_keys = ['labels', 'colors']
        attr_str = ', '.join([f'{key}={getattr(self, key)!r}'
                              for key in self_dict.keys()
                              if key not in exclude_keys])
        return f'Clustering({attr_str})'


    def get_cell_dict(self, source_data, cells_ids):
        if isinstance(source_data, np.ndarray):
            cell_dict = {
                cell_id: [clust]
                for cell_id, clust in zip(cells_ids, source_data)}
        elif isinstance(source_data, dict):
            if len(source_data) == len(cells_ids):
                cell_dict = dict(zip(cells_ids, source_data.values()))
            else:
                # translate dict cluster: [cells] to cell: [clusters]
                cell_dict = {cell_id: [] for cell_id in cells_ids}
                for cluster, cells in source_data.items():
                    for c in cells:
                        cell_dict[cells_ids[c]].append(cluster)
        else:
            raise TypeError(
                '''source_data must either be an array of cluster
                labels (as is the case for the result from hierarchical
                clustering), or a dictionary mapping clusters to a list of
                cells, or cells to a list of clusters''')

        self.cell_dict = cell_dict
        return self.cell_dict


    def init_labels(self):
        labels = pd.Series(self.cell_dict, name='labels')
        labels = (
            labels.apply(lambda x: '+'.join([str(c+1) for c in x])))
        homeless_mask = labels == '0'
        labels.loc[homeless_mask] = 'homeless'
        self.nr_clusters = labels.nunique()
        self.prop_homeless = homeless_mask.sum() / homeless_mask.shape[0]
        self.labels = labels


    def get_binary_matrix(self, other_matrix=None):
        nr_cells = self.labels.shape[0]
        binary_matrix = np.zeros((nr_cells, self.nr_clusters), dtype=int)
        mask = self.labels != 'homeless'
        clust_arr = (
            self.labels.loc[mask]
                       .apply(lambda lb: np.array(lb.split('+')))
                       .values
                       .astype(int))
        clust_arr = clust_arr - 1
        for i, clusts in zip(np.where(mask.values)[0], clust_arr):
            binary_matrix[i, clusts] += 1

        # If we wish to compare to another matrix, permute the cluster numbers
        # so as to maximise the overlap with the clustering corresponding to the
        # other matrix. In other words, make the correspondence between the two
        # sets of cluster labels.
        if other_matrix is not None:
            i_col_list = list(range(other_matrix.shape[1]))
            while len(i_col_list) > 1:
                max_score = 0
                for pot_i_og in i_col_list[:-1]:
                    avg_size = (binary_matrix[:, pot_i_og].sum()
                                + other_matrix.sum(axis=0)) / 2
                    # Count how many cells have the same assignment, divide by
                    # average cluster size
                    is_overlap = binary_matrix.T == other_matrix[:, pot_i_og]
                    overlap = is_overlap.sum(axis=1) / avg_size
                    pot_i_dest = np.argmax(overlap)
                    new_score = overlap[pot_i_dest]
                    if new_score > max_score:
                        max_score = new_score
                        i_dest = pot_i_dest
                        i_og = pot_i_og
                        
                binary_matrix[:, i_og], binary_matrix[:, i_dest] = (
                    binary_matrix[:, i_dest], binary_matrix[:, i_og].copy())
                print(i_og, i_dest)
                i_col_list.remove(i_og)
                if i_og != i_dest:
                    i_col_list.remove(i_dest)
                    
        return binary_matrix


    def attr_color_to_labels(self, cmap=None):
        # make it an attribute to have consistent coloring?
        unique_labels = sorted(self.labels.unique())
        if cmap is None:
            gen_colors_fun = map_viz.gen_distinct_colors
        else:
            gen_colors_fun = lambda n: list(plt.get_cmap(cmap, n).colors)
        nr_cats = len(unique_labels)
        if 'homeless' in unique_labels:
            # Cover case in which only homeless cells.
            if nr_cats == 1:
                colors = [(0.5, 0.5, 0.5, 1)]
            else:
                colors = gen_colors_fun(nr_cats - 1) + [(0.5, 0.5, 0.5, 1)]
        else:
            colors = gen_colors_fun(nr_cats)

        self.colors = dict(zip(unique_labels, colors))


    def map_plot(self, geodf, shape_geodf, fig=None, ax=None, cax=None,
                 show=True, save_path=None, cmap=None, xy_proj='epsg:3857',
                 **kwargs):
        if cmap is not None:
            self.attr_color_to_labels(cmap=cmap)

        if ax is None:
            fig, ax = plt.subplots(1)

        plot_geodf = (geodf.join(self.labels, how='inner')
                           .to_crs(xy_proj))

        for label, label_geodf in plot_geodf.groupby('labels'):
            # Don't put a cmap in kwargs['plot'] because here we use a fixed
            # color per cluster.
            label_geodf.plot(ax=ax, color=self.colors[label],
                             **kwargs.get('plot', {}))

        shape_geodf.to_crs(xy_proj).plot(ax=ax, color='none', edgecolor='black')
        ax.set_axis_off()

        if cax:
            # The colours will correspond because groupby sorts by the column by
            # which we group, and we sorted the unique labels.
            cax = map_viz.colored_pts_legend(cax, self.colors,
                                             **kwargs.get('legend', {}))

        if save_path:
            fig.savefig(save_path)
        if show:
            fig.show()
        return fig, ax


    def reach_plot(self, fig=None, ax=None, figsize=None, show=True):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)

        optics = self.method_obj
        reachability = optics.reachability_[optics.ordering_]
        ordered_labels = optics.labels_[optics.ordering_]
        space = np.arange(len(ordered_labels))

        for klass in np.unique(ordered_labels)[1:]:
            Xk = space[ordered_labels == klass]
            Rk = reachability[ordered_labels == klass]
            ax.plot(Xk, Rk)
        ax.plot(space[ordered_labels == -1], reachability[ordered_labels == -1],
                'k.', alpha=0.3, ms=1)
        ax.set_ylabel('Reachability (epsilon distance)')
        ax.set_title('Reachability Plot')
        if show:
            fig.show()
        return fig, ax


    def silhouette_plot(self, proj_vectors, metric=None):
        # convention that in labels 0 means noise
        cluster_labels = self.labels.values.astype(int) - 1
        if metric is None:
            if self.method_kwargs is not None:
                metric = self.method_kwargs.get('metric', 'euclidean')
            else:
                metric = 'euclidean'
        fig, ax = eval_viz.silhouette(proj_vectors, cluster_labels,
                                      metric=metric)
        return fig, ax


@dataclass
class HierarchicalClustering:
    levels: List[Clustering]
    method_repr: str
    method_args: Optional[list] = None
    method_kwargs: Optional[dict] = None
    kwargs_str: Optional[str] = None
    linkage: Optional[np.ndarray] = None
    cut_tree: Optional[np.ndarray] = None


    def __post_init__(self):
        if self.kwargs_str is None:
            self.kwargs_str = '_params=({})'.format(
                '_'.join([f'{key}={value}'
                         for key, value in self.method_kwargs.items()]))

        if self.method_repr == 'shc.linkage':
            # not implemented for other methods
            self.attr_lvl_colors()


    def __str__(self):
        self_dict = asdict(self)
        exclude_keys = ['linkage', 'cut_tree']
        return str({key: value
                    for key, value in self_dict.items()
                    if key not in exclude_keys})


    def __repr__(self):
        self_dict = asdict(self)
        exclude_keys = ['linkage', 'cut_tree']
        attr_str = ', '.join([f'{key}={getattr(self, key)!r}'
                              for key in self_dict.keys()
                              if key not in exclude_keys])
        return f'HierarchicalClustering({attr_str})'


    @classmethod
    def from_scipy_agglo(cls, vectors, cells_ids, max_n_clusters=None,
                         **linkage_kwargs):
        method_repr = 'shc.linkage'
        linkage = shc.linkage(vectors, **linkage_kwargs)
        if max_n_clusters is None:
            n_clusters = None
        else:
            n_clusters = np.asarray(range(2, max_n_clusters + 1))
        cut_tree = shc.cut_tree(linkage, n_clusters=n_clusters)
        # colors={} to skip attr_color_to_labels
        levels = [
            Clustering(lvl, cells_ids, method_repr, colors={},
                       method_kwargs={**linkage_kwargs, 'lvl': i})
            for i, lvl in enumerate(cut_tree.T)]
        return cls(levels, method_repr, method_kwargs=linkage_kwargs,
                   linkage=linkage, cut_tree=cut_tree)

    @classmethod
    def from_oslom_run(cls, oslom_net_file_path, cells_ids,
                       oslom_opt_params=None):
        if oslom_opt_params is None:
            oslom_opt_params = []
        method_repr = 'oslom'
        oslom_res_path = gen_oslom_res_path(
            oslom_net_file_path, oslom_opt_params=oslom_opt_params)
        _ = run_oslom(OSLOM_DIR, oslom_net_file_path, oslom_res_path,
                      oslom_opt_params=oslom_opt_params)
        levels_dict = read_oslom_res(oslom_res_path)
        levels = [Clustering(lvl, cells_ids, method_repr)
                  for lvl in levels_dict.values()]
        return cls(levels, method_repr,
                   kwargs_str=''.join(oslom_opt_params))


    @classmethod
    def from_oslom_res(cls, oslom_net_file_path, cells_ids, metric,
                       transfo=None, oslom_opt_params=None):
        if transfo is None:
            transfo_str = 'inverse'
        else:
            transfo_str = transfo.__name__
        if oslom_opt_params is None:
            oslom_opt_params = []
        kwargs_str = f'_metric={metric}_transfo={transfo_str}' + ''.join(oslom_opt_params)
        method_repr = 'oslom'
        oslom_res_path = gen_oslom_res_path(
            oslom_net_file_path, oslom_opt_params=oslom_opt_params)
        levels_dict = read_oslom_res(oslom_res_path)
        levels = [Clustering(lvl, cells_ids, method_repr, kwargs_str=kwargs_str)
                  for lvl in levels_dict.values()]
        return cls(levels, method_repr, kwargs_str=kwargs_str)


    @classmethod
    def from_sbm_res(cls, state, cells_ids, **sbm_kwargs):
        '''
        From the output of `sbm_run`, state
        '''
        method_repr = 'sbm'
        kwargs = {**sbm_kwargs, **{'transfo': sbm_kwargs['transfo'].__name__}}
        kwargs_str = '_' + '_'.join([f'{key}={value}'
                                     for key, value in kwargs.items()])
        levels = state.get_bs()
        levels_dict = {}
        cells_clusters = np.asarray(levels[0])
        levels_dict[0] = cells_clusters.copy()
        for l in range(1, len(levels)):
            cells_clusters = np.asarray([levels[l][c] for c in cells_clusters])
            levels_dict[l] = cells_clusters.copy()
        levels = [Clustering(lvl, cells_ids, method_repr, kwargs_str=kwargs_str)
                  for lvl in levels_dict.values()]
        return cls(levels, method_repr, kwargs_str=kwargs_str)


    def get_clusters_agg(self):
        '''
        From a matrix (n_samples x n_levels), returns a matrix (n_levels x
        max_nr_clusters) giving the assignment of the lowest level's clusters at
        higher levels, thus showing which clusters get aggregated with which at
        each aggregation step.
        '''
        n_lvls = len(self.levels)
        sorted_levels = sorted(self.levels, key=lambda x: getattr(x, 'nr_clusters'))
        levels_x_clust = np.zeros((n_lvls, n_lvls + 1), dtype=int)
        levels_x_clust[-1, :] = np.arange(0, n_lvls + 1)
        for i in range(n_lvls-2, -1, -1):
            lvl_clusts = sorted_levels[i].labels.values.astype(int) - 1
            lower_lvl_clusts = sorted_levels[i+1].labels.values.astype(int) - 1
            # For every cluster in the lower level,
            for clust in np.unique(lower_lvl_clusts):
                # We select the higher level cluster to which it belongs.
                # Because we started from the less aggregated level, all members
                # of that cluster will belong to the same cluster in the more
                # aggregated level, so we take the higher level cluster of the
                # first one.
                agg_lvl = lvl_clusts[lower_lvl_clusts == clust][0]
                levels_x_clust[i, :][levels_x_clust[i+1, :] == clust] = agg_lvl
        levels_x_clust += 1

        return levels_x_clust


    def attr_lvl_colors(self, cmap=None):
        og_m = self.get_clusters_agg()
        nr_levels = len(self.levels)
        self.levels[-1].attr_color_to_labels(cmap=cmap)
        prev_color_dict = self.levels[-1].colors

        for lvl in range(nr_levels-1, 0, -1):
            before = og_m[lvl]
            after = og_m[lvl-1]
            before_labels_dict = {lbl: set() for lbl in after}
            for albl, blbl in zip(after, before):
                before_labels_dict[albl].add(blbl)
            lvl_color_dict = {}
            list_used_labels = []
            for label, before_labels in before_labels_dict.items():
                if len(before_labels) == 1:
                    before_label = before_labels.pop()
                    c = prev_color_dict[str(before_label)]
                    list_used_labels.append(before_label)
                elif label in before_labels:
                    c = prev_color_dict[str(label)]
                    list_used_labels.append(label)
            else:
                    cell_attr = self.levels[lvl].labels
                    bigger_clust = cell_attr[cell_attr.astype(int).isin(before_labels)].value_counts().idxmax()
                    c = prev_color_dict[str(bigger_clust)]
                lvl_color_dict[str(label)] = c

            self.levels[lvl-1].colors = lvl_color_dict
            prev_color_dict = lvl_color_dict.copy()


    def plot_dendrogram(self, coloring_lvl=-1, ax=None, **shc_dendro_kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 7))
        fig = ax.get_figure()

        clust_coloring = self.levels[coloring_lvl]
        hex_color_dict = {key: mcolors.to_hex(c)
                          for key, c in clust_coloring.colors.items()}
        leaves_colors = clust_coloring.labels.map(hex_color_dict)
        leaf_colors_dict = dict(zip(range(len(leaves_colors)), leaves_colors.values))
        Z = self.linkage

        # As per https://stackoverflow.com/a/38208611/13168978:
        # notes:
        # * rows in Z correspond to "inverted U" links that connect clusters
        # * rows are ordered by increasing distance
        # * if the colors of the connected clusters match, use that color for link
        link_cols = {}

        nr_leaves = len(Z) + 1
        for i, i12 in enumerate(Z[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > nr_leaves-1 else leaf_colors_dict[x]
                      for x in i12)
            link_cols[i+nr_leaves] = c1 if c1 == c2 else 'k'

        shc_dendro_kwargs.setdefault('truncate_mode', 'level')
        shc_dendro_kwargs.setdefault('p', len(self.levels) // 2 + 1)
        shc_dendro_kwargs.setdefault('link_color_func', lambda x: link_cols[x])
        _ = shc.dendrogram(self.linkage, ax=ax, **shc_dendro_kwargs)

        return fig, ax


    def map_plot(self, *map_plot_args, fig=None, axes=None, **map_plot_kwargs):
        if axes is None:
            fig, axes = plt.subplots(nrows=len(self.levels))
        for clustering, ax in zip(self.levels, axes):
            fig, ax = clustering.map_plot(
                *map_plot_args, fig=fig, ax=ax, **map_plot_kwargs)
        return fig, axes


    def silhouette_plot(self, proj_vectors, metric=None):
        for lvl in self.levels:
            _, _ = lvl.silhouette_plot(proj_vectors, metric=metric)



@dataclass
class Decomposition:
    word_counts_vectors: word_counts.WordCountsVectors
    word_vectors: word_counts.WordVectors
    decomposition: PCA
    proj_vectors: np.ndarray
    word_mask: np.ndarray
    n_components: int = 0
    clusterings: List[Union[Clustering, HierarchicalClustering]] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.n_components = self.proj_vectors.shape[1]
        # To save memory:
        self.word_counts_vectors = self.word_counts_vectors[[0],[0]].copy()
        self.word_vectors = self.word_vectors[[0],[0]].copy()


    def __str__(self):
        self_dict = asdict(self)
        exclude_keys = ['proj_vectors']
        return str({key: value
                    for key, value in self_dict.items()
                    if key not in exclude_keys})


    def __repr__(self):
        self_dict = asdict(self)
        exclude_keys = ['proj_vectors']
        attr_str = ', '.join([f'{key}={getattr(self, key)!r}'
                              for key in self_dict.keys()
                              if key not in exclude_keys])
        return f'Decomposition({attr_str})'


    def save_net(self, metric, net_file_path_fmt, transfo=None):
        net_file_path = gen_net_file_path(
            net_file_path_fmt, self.decomposition, metric, transfo=transfo)
        if transfo is None:
            transfo = lambda x: 1 / x
        dist_vec = scipy.spatial.distance.pdist(self.proj_vectors,
                                                metric=metric)
        sim_mat = scipy.spatial.distance.squareform(transfo(dist_vec))
        edge_list = []
        nr_cells = self.proj_vectors.shape[0]
        for i in range(nr_cells):
            for j in range(i+1, nr_cells):
                edge_list.append((i, j, sim_mat[i, j]))
        # or save directly line by line?
        net_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(net_file_path, 'w') as f:
            f.write('\n'.join([
                ' '.join([str(x) for x in edge])
                for edge in edge_list]))
        return net_file_path


    def explained_var_plot(self, n_components=None, ax=None, lgd_kwargs=None):
        if lgd_kwargs is None:
            lgd_kwargs = {}
        if ax is None:
            fig, ax = plt.subplots(1)
        fig = ax.get_figure()

        var_prop = self.decomposition.explained_variance_ratio_[:n_components]
        n_components = var_prop.size
        var_prop = np.insert(var_prop, 0, 0)
        x_plot = np.arange(0, n_components + 1)

        ax.stairs(var_prop[1:], x_plot, alpha=0.5, fill=True, label="component's")
        y_plot = broken_stick(self.decomposition.n_features_)[:n_components]
        ax.stairs(y_plot, x_plot, label='broken-stick model', color='r', fill=True, alpha=0.5)
        ax.set_xlabel('component rank')
        ax.set_ylabel('by component')
        ax.set_yscale('log')
        ax.set_title('proportion of variance explained')

        ax2 = ax.twinx()
        y_plot = var_prop.cumsum()
        ax2.plot(x_plot, y_plot, ls=':', marker='.', label='cumulative')
        ax2.set_ylabel('cumulative')
        fig.legend(
            title='explained variance', bbox_transform=ax.transAxes, **lgd_kwargs
        )
        return fig, ax


    def map_comp(self, lang, nr_plots=5, cmap='plasma', **plot_kwargs):
        for i in range(nr_plots):
            comp_series = pd.Series(self.proj_vectors[:, i],
                                    index=lang.relevant_cells, name='pca_comp')
            fig, axes = plt.subplots(
                ncols=len(lang.list_cc)+1, figsize=(9, 3),
                gridspec_kw={'width_ratios': lang.width_ratios})
            map_axes = axes[:-1]
            cax = axes[-1]
            vmin = self.proj_vectors[:, i].min()
            vmax = self.proj_vectors[:, i].max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            for ax, cc in zip(map_axes, lang.list_cc):
                mask = lang.cells_geodf.index.str.startswith(cc)
                plot_df = (
                    lang.cells_geodf.loc[mask].join(comp_series, how='inner'))
                plot_df.plot(column='pca_comp', ax=ax, norm=norm, cmap=cmap,
                             **plot_kwargs)
                ax.set_axis_off()

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            _ = fig.colorbar(sm, cax=cax, label='')
            fig.show()


    def add_clustering(self, method, cells_ids, *method_args,
                       append=True, **method_kwargs):
        if hasattr(method, 'fit_predict'):
            method = method(*method_args, **method_kwargs)
            res = method.fit_predict(self.proj_vectors)
        elif callable(method):
            res = method(self.proj_vectors, *method_args, **method_kwargs)
        else:
            raise TypeError('Please provide a callable or an object with a fit_predict method')
        clust = Clustering(
            res, cells_ids, repr(method), method_obj=method,
            method_args=method_args, method_kwargs=method_kwargs)
        cluster_labels = clust.labels.values.astype(int) - 1
        clust.score = silhouette_score(self.proj_vectors, cluster_labels,
                                       metric='euclidean')
        if append or len(self.clusterings) == 0:
            self.clusterings.append(clust)
        else:
            self.clusterings[-1] = clust
        return clust


    def add_scipy_hierarchy(self, cells_ids, **kwargs):
        clustering = HierarchicalClustering.from_scipy_agglo(
            self.proj_vectors, cells_ids, **kwargs)
        metric = kwargs.get('metric', 'euclidean')
        for clust in clustering.levels:
            cluster_labels = clust.labels.values.astype(int) - 1
            clust.score = silhouette_score(self.proj_vectors, cluster_labels,
                                           metric=metric)
        self.clusterings.append(clustering)
        return clustering


    def prep_oslom(self, metric, net_file_path_fmt, transfo=None):
        data_path = self.save_net(metric, net_file_path_fmt, transfo=transfo)
        return OSLOM_DIR, data_path


    def add_oslom_hierarchy(self, metric, net_file_path_fmt, cells_ids,
                            transfo=None, oslom_opt_params=None):
        oslom_net_file_path = gen_net_file_path(
            net_file_path_fmt, self.decomposition, metric, transfo=transfo)
        clustering = HierarchicalClustering.from_oslom_res(
            oslom_net_file_path, cells_ids, metric, transfo=transfo,
            oslom_opt_params=oslom_opt_params)
        self.clusterings.append(clustering)
        return clustering


    def add_sbm_hierarchy(self, state, cells_ids, **sbm_kwargs):
        clustering = HierarchicalClustering.from_sbm_res(
            state, cells_ids, **sbm_kwargs)
        self.clusterings.append(clustering)
        return clustering
