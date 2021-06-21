import json
import pickle
import inspect
from pathlib import Path
from dataclasses import dataclass, field, InitVar, asdict
from typing import List
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as geopd
import ray
from sklearn.decomposition import PCA
import libpysal
import src.utils.geometry as geo
import src.utils.parallel as parallel
import src.utils.paths as paths_utils
import src.visualization.maps as map_viz
import src.data.word_counts as word_counts
import src.data.clustering as data_clustering

@dataclass
class Region:
    cc: str
    lc: str
    readable: str = ''
    xy_proj: str = 'epsg:3857'
    max_place_area: float = 5e9
    cell_size: float = 50e3
    shape_bbox: List[float] = None
    shapefile_name: str = 'CNTR_RG_01M_2016_4326.shp'
    shapefile_col: str = 'FID'
    shape_geodf: geopd.GeoDataFrame = None
    cells_geodf: geopd.GeoDataFrame = None
    cell_counts: pd.DataFrame = None
    region_counts: pd.DataFrame = None
    raw_cell_counts: pd.DataFrame = None
    shapefile_val: str = None


    def __post_init__(self):
        self.shapefile_val = self.shapefile_val or self.cc


    @classmethod
    def from_dict(cls, cc, lc, d):
        return cls(cc=cc, lc=lc, **{
            k: v for k, v in d.items()
            if k in inspect.signature(cls).parameters
        })


    @staticmethod
    def from_global_json(file_path: str, cc: str, lc: str):
        with open(file_path) as f:
            countries_dict = json.load(f)
        d = countries_dict[cc]
        return Region.from_dict(cc, lc, d)


    def get_shape_geodf(self, all_cntr_shapes=None, simplify_tol=1000):
        if self.shape_geodf is None:
            mask = all_cntr_shapes[self.shapefile_col].str.startswith(self.shapefile_val)
            self.shape_geodf = all_cntr_shapes.loc[mask]
            self.shape_geodf = geo.extract_shape(
                self.shape_geodf, self.cc, xy_proj=self.xy_proj,
                bbox=self.shape_bbox, simplify_tol=simplify_tol)
        return self.shape_geodf


    def get_cells_geodf(self, **kwargs):
        if self.cells_geodf is None:
            self.shape_geodf = self.get_shape_geodf(**kwargs)
            _, self.cells_geodf, _, _ = geo.create_grid(
                self.shape_geodf, self.cell_size,
                xy_proj=self.xy_proj, intersect=True)
        return self.cells_geodf


    def read_counts(self, df_name, files_fmt=None):
        if hasattr(self, df_name):
            if getattr(self, df_name) is None:
                parquet_file = str(files_fmt).format(
                    kind=df_name, lc=self.lc, cc=self.cc)
                setattr(self, df_name, pd.read_parquet(parquet_file))
            return getattr(self, df_name)
        else:
            print(f'{df_name} is not a valid name')


    def cleanup(self):
        self.cell_counts = None
        self.region_counts = None
        self.raw_cell_counts = None


@dataclass
class Language:
    lc: str
    readable: str
    list_cc: List[str]
    regions: List[Region]
    _paths: paths_utils.ProjectPaths
    all_cntr_shapes: InitVar[geopd.GeoDataFrame]
    cc: str = None
    latlon_proj: str = 'epsg:4326'
    min_nr_cells: int = 10
    cell_tokens_th: float = 1e4
    cell_tokens_decade_crit: float = 2.
    cells_geodf: geopd.GeoDataFrame = None
    global_counts: pd.DataFrame = None
    raw_cell_counts: pd.DataFrame = None
    words_prior_mask: pd.Series = None
    cell_counts: pd.DataFrame = None
    relevant_cells: pd.Index = None
    word_counts_vectors: np.ndarray = None
    word_vec_var: str = 'normed_freqs'
    word_vectors: np.ndarray = None
    cdf_th: float = 0.99
    width_ratios: np.ndarray = None
    decompositions: List[data_clustering.Decomposition] = field(default_factory=list)
    z_th: float = 10
    p_th: float = 0.01

    def __post_init__(self, all_cntr_shapes):
        self.list_cc, self.regions = zip(*sorted(
            zip(self.list_cc, self.regions), key=lambda t: t[0]))
        self.cc = '-'.join(self.list_cc)
        self.cells_geodf = pd.concat([
            reg.get_cells_geodf(all_cntr_shapes=all_cntr_shapes)
               .to_crs(self.latlon_proj)
            for reg in self.regions
            ])


    @classmethod
    def from_countries_dict(cls, lc, readable, list_cc, countries_dict,
                            all_cntr_shapes, paths, **kwargs):
        list_cc.sort()
        regions = [Region.from_dict(cc, lc, countries_dict[cc])
                   for cc in list_cc]
        return cls(lc, readable, list_cc, regions, paths,
                   all_cntr_shapes, **kwargs)


    def data_file_fmt(self, save_dir):
        params_str = '_'.join(
            [f'{key}={{{key}}}'
             for key in ('lc', 'cc', 'min_nr_cells', 'cell_tokens_decade_crit',
                         'cell_tokens_th', 'cdf_th')])
        save_path_fmt = str(save_dir / f'{{kind}}_{params_str}.{{ext}}')
        return save_path_fmt


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'cc', 'min_nr_cells', 'cell_tokens_decade_crit',
            'cell_tokens_th', 'word_vec_var', 'cdf_th']
        return {attr: getattr(self, attr) for attr in list_attr}


    @property
    def paths(self):
        '''
        Defined as a property so that when we pass paths to eg a child
        Decomposition or Clustering, the paths is already partially formatted
        with the Language attributes.
        '''
        self._paths.partial_format(**self.to_dict())
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p


    def to_pickle(self, save_path_fmt):
        self.cleanup(include_global=False)
        save_path = Path(str(save_path_fmt).format(**self.to_dict()))
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)


    def get_global_counts(self):
        if self.global_counts is None:
            cols = ['count', 'is_proper', 'nr_cells']
            self.global_counts = pd.DataFrame({c: [] for c in cols})
            for reg in self.regions:
                reg_counts = reg.read_counts(
                    'region_counts', files_fmt=self.paths.counts_files_fmt)
                self.global_counts = self.global_counts.add(reg_counts[cols],
                                                            fill_value=0)
            self.global_counts = self.global_counts.sort_values(by='count',
                                                                ascending=False)
        return self.global_counts


    def filter_global_counts(self):
        self.global_counts = self.get_global_counts()
        proper_mask = self.global_counts['is_proper'] == 0
        nr_cells_mask = self.global_counts['nr_cells'] >= self.min_nr_cells
        self.words_prior_mask = proper_mask & nr_cells_mask
        self.global_counts = self.global_counts.loc[self.words_prior_mask]
        return self.global_counts


    def get_raw_cell_counts(self):
        if self.raw_cell_counts is None:
            self.raw_cell_counts = pd.concat([
                reg.read_counts('raw_cell_counts',
                                files_fmt=self.paths.counts_files_fmt)
                for reg in self.regions
                ]).sort_index(axis=0, level=0)
        return self.raw_cell_counts


    def get_cell_counts(self):
        if self.cell_counts is None:
            self.raw_cell_counts = self.get_raw_cell_counts()
            total_nr_tokens = self.raw_cell_counts['count'].sum()
            if self.words_prior_mask is None:
                _ = self.filter_global_counts()
            self.cell_counts = word_counts.filter_part_multidx(
                self.raw_cell_counts, [self.words_prior_mask])

            cell_sum = self.cell_counts.groupby('cell_id')['count'].sum()
            # For countries containing deserts like Australia, geometric mean
            # can be very low, so take at least the default `sum_th`.
            sum_th = max(
                10**(np.log10(cell_sum).mean() - self.cell_tokens_decade_crit),
                self.cell_tokens_th)
            cell_is_relevant = cell_sum > sum_th
            self.relevant_cells = cell_is_relevant.loc[cell_is_relevant].index
            print(f'Keeping {self.relevant_cells.shape[0]} cells out of '
                  f'{cell_is_relevant.shape[0]} with threshold {sum_th:.2e}')
            self.cell_counts = word_counts.filter_part_multidx(
                self.cell_counts, [cell_is_relevant])

            filtered_nr_tokens = self.cell_counts['count'].sum()
            rel_diff = (total_nr_tokens - filtered_nr_tokens) / total_nr_tokens
            print(f'We had {total_nr_tokens:.0f} tokens, and filtering ',
                  f'brought it down to {filtered_nr_tokens:.0f}, so we lost ',
                  f'{100*rel_diff:.3g}%.')

        return self.cell_counts

    
    def save_interim(self):
        save_path_fmt = self.data_file_fmt(self.paths.interim_data)
        self_dict = self.to_dict()
        save_path = save_path_fmt.format(kind='word_counts_vectors',
                                         **self_dict, ext='csv.gz')
        np.savetxt(save_path, self.word_counts_vectors, delimiter=',')
        save_path = save_path_fmt.format(kind='relevant_cells',
                                         **self_dict, ext='csv.gz')
        np.savetxt(save_path, self.relevant_cells.values, delimiter=',', fmt="%s")
        save_path = save_path_fmt.format(kind='global_counts',
                                         **self_dict, ext='parquet')
        self.global_counts.to_parquet(save_path, index=True)


    def load_interim(self):
        save_path_fmt = self.data_file_fmt(self.paths.interim_data)
        self_dict = self.to_dict()
        save_path = save_path_fmt.format(kind='word_counts_vectors',
                                         **self_dict, ext='csv.gz')
        self.word_counts_vectors = np.loadtxt(save_path, delimiter=',')
        save_path = save_path_fmt.format(kind='relevant_cells',
                                         **self_dict, ext='csv.gz')
        self.relevant_cells = pd.Index(
            np.loadtxt(save_path, delimiter=',', dtype='object'),
            name='cell_id')
        save_path = save_path_fmt.format(kind='global_counts',
                                         **self_dict, ext='parquet')
        self.global_counts = pd.read_parquet(save_path)


    def get_width_ratios(self, ratio_lgd=None):
        self.width_ratios = np.ones(len(self.list_cc) + 1)
        for i, reg in enumerate(self.regions):
            min_lon, min_lat, max_lon, max_lat = (
                reg.shape_geodf.geometry.to_crs(self.latlon_proj).total_bounds)
            # For a given longitude extent, the width is maximum the closer to the
            # equator, so the closer the latitude is to 0.
            eq_crossed = min_lat * max_lat < 0
            lat_max_width = min(abs(min_lat), abs(max_lat)) * (1 - int(eq_crossed))
            width = geo.haversine(min_lon, lat_max_width, max_lon, lat_max_width)
            height = geo.haversine(min_lon, min_lat, min_lon, max_lat)
            self.width_ratios[i] = width / height
        if ratio_lgd:
            self.width_ratios[-1] = ratio_lgd
        else:
            self.width_ratios = self.width_ratios[:-1]
        return self.width_ratios


    def get_word_counts_vectors(self):
        if self.word_counts_vectors is None:
            self.cell_counts = self.get_cell_counts()
            counts = self.global_counts['count']
            cdf_mask = (counts / counts.sum()).cumsum() < self.cdf_th
            self.word_counts_vectors = word_counts.to_vectors(self.cell_counts,
                                                              cdf_mask)
        return self.word_counts_vectors


    def set_cdf_th(self, th):
        self.cdf_th = th
        self.word_counts_vectors = None
        self.word_vectors = None
        _ = self.get_word_counts_vectors()


    def get_word_vectors(self):
        self.word_counts_vectors = self.get_word_counts_vectors()
        counts = self.global_counts['count']
        cdf_mask = (counts / counts.sum()).cumsum() < self.cdf_th
        # this is done here and not above because Language can be loaded from
        # interim data
        self.global_counts['cdf_mask'] = cdf_mask
        self.word_vectors = word_counts.vec_to_metric(
            self.word_counts_vectors, self.global_counts.loc[cdf_mask],
            word_vec_var=self.word_vec_var)
        return self.word_vectors


    def set_word_vec_var(self, word_vec_var):
        self.word_vec_var = word_vec_var
        _ = self.get_word_vectors()


    def calc_morans(self, num_cpus=1):
        contiguity = libpysal.weights.Queen.from_dataframe(
            self.cells_geodf.loc[self.relevant_cells])
        contiguity.transform = 'r'
        ray.init(num_cpus=num_cpus)
        num_morans = self.word_vectors.shape[1]
        shared_word_vectors = ray.put(self.word_vectors)
        obj_refs = parallel.split_task(
            data_clustering.chunk_moran, num_cpus,
            list(range(num_morans)), shared_word_vectors, contiguity)
        res = ray.get(obj_refs)
        moran_dict = res[0]
        for m_dict in res[1:]:
            for key, value in m_dict.items():
                moran_dict[key].extend(value)
        ray.shutdown()
        cdf_mask = self.global_counts['cdf_mask']
        words = self.global_counts.loc[cdf_mask].index[:num_morans]
        moran_df = (pd.DataFrame.from_dict(moran_dict)
                                .set_index(words))
        self.global_counts = self.global_counts.join(moran_df)
        return self.global_counts


    def filter_word_vectors(self, z_th=10, p_th=0.01):
        self.z_th = z_th
        self.p_th = p_th
        self.word_vectors = self.get_word_vectors()
        # assumes moran has been done
        is_regional = ((self.global_counts['z_value'] > z_th)
                       & (self.global_counts['p_value'] < p_th))
        cdf_mask = self.global_counts['cdf_mask']
        mask = is_regional.loc[cdf_mask].values
        self.word_vectors = self.word_vectors[:, mask]
        nr_kept = self.word_vectors.shape[1]
        print(f'Keeping {nr_kept} words out of {mask.shape[0]}')


    def make_decomposition(self, **kwargs):
        pca = PCA(**kwargs).fit(self.word_vectors)
        proj_vectors = pca.transform(self.word_vectors)
        decomposition = data_clustering.Decomposition(
            self.word_vec_var, pca, proj_vectors, self.word_vectors.shape[1],
            self.z_th, self.p_th)
        self.decompositions.append(decomposition)
        return decomposition


    def cleanup(self, include_global=False, include_regions=True):
        self.cell_counts = None
        self.raw_cell_counts = None
        # Global counts can be useful to go back to the words, so possibility is
        # left open.
        if include_global:
            self.global_counts = None
        if include_regions:
            for reg in self.regions:
                reg.cleanup()


    def add_cc(self, list_cc, countries_dict):
        # Allow for supplying list of cc containing country codes of already
        # initialised regions, in that case we keep the existing and only add
        # the new ones.
        new_cc = [cc for cc in list_cc if cc not in self.list_cc]
        self.list_cc.extend(new_cc)
        self.regions.extend([Region.from_dict(cc, self.lc, countries_dict[cc])
                             for cc in new_cc])
        self.cleanup(include_global=True, include_regions=False)


    def get_clust_words(self, i_decompo=-1, i_clust=-1, i_lvl=0):
        decomp = self.decompositions[i_decompo]
        clustering = decomp.clusterings[i_clust]
        levels = getattr(clustering, 'levels', [clustering])
        cluster_labels = levels[i_lvl].clusters_series
        is_regional = ((self.global_counts['z_value'] > decomp.z_th)
                       & (self.global_counts['p_value'] < decomp.p_th))
        clust_words = self.global_counts.loc[is_regional].copy()
        for lbl in np.unique(cluster_labels):
            cluster_mask = cluster_labels == lbl
            clust_center = decomp.proj_vectors[cluster_mask, :].mean(axis=0)
            words_cluster = decomp.decomposition.inverse_transform(clust_center)
            clust_words[f'cluster{lbl}'] = words_cluster
        return clust_words


    def get_maps_pos(self, total_width, total_height=None, ratio_lgd=None):
        width_ratios = self.get_width_ratios(ratio_lgd=ratio_lgd)
        return map_viz.position_axes(width_ratios, total_width,
                                     total_height=total_height)


    def map_comp_loading(self, i_decompo=-1, nr_plots=5, cmap='bwr',
                         total_width=178, total_height=None, **plot_kwargs):
        normed_bboxes, (total_width, total_height) = self.get_maps_pos(
            total_width, total_height=total_height, ratio_lgd=1/10)
        proj_vectors = self.decompositions[i_decompo].proj_vectors
        for i in range(nr_plots):
            comp_series = pd.Series(proj_vectors[:, i],
                                    index=self.relevant_cells, name='pca_comp')
            fig, axes = plt.subplots(
                len(self.list_cc)+1,
                figsize=(total_width/10/2.54, total_height/10/2.54))
            map_axes = axes[:-1]
            cax = axes[-1]
            cax.set_position(normed_bboxes[-1])
            vmin = proj_vectors[:, i].min()
            vmax = proj_vectors[:, i].max()
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

            for ax, reg, bbox in zip(map_axes, self.regions, normed_bboxes[:-1]):
                ax.set_position(bbox)
                plot_df = reg.cells_geodf.join(comp_series, how='inner')
                plot_df.plot(column='pca_comp', ax=ax, norm=norm, cmap=cmap,
                             **plot_kwargs)
                reg.shape_geodf.plot(ax=ax, color='none', edgecolor='black')
                ax.set_axis_off()

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            _ = fig.colorbar(sm, cax=cax, label=f'Loading of component {i}')
            fig.show()


    def map_clustering(self, i_decompo=-1, i_clust=-1, total_width=178,
                       cmap=None, show=True, save_path_fmt=None, **kwargs):
        # total width in mm
        normed_bboxes, (total_width, total_height) = self.get_maps_pos(total_width)
        fig_list = []
        axes_list = []
        decomposition = self.decompositions[i_decompo]
        clustering = decomposition.clusterings[i_clust]
        for level in getattr(clustering, 'levels', [clustering]):
            fig, axes = plt.subplots(
                len(self.list_cc),
                figsize=(total_width/10/2.54, total_height/10/2.54))
            if len(self.list_cc) == 1:
                axes = (axes,)

            if level.clusters_series is None:
                level.clusters_series = level.get_clusters_series(self.relevant_cells)

            label_color = level.attr_color_to_labels(cmap=cmap)
            for ax, reg, bbox in zip(axes, self.regions, normed_bboxes):
                ax.set_position(bbox)
                cc_geodf = reg.cells_geodf.join(level.clusters_series, how='inner')
                for label, label_geodf in cc_geodf.groupby('clusters'):
                    # Don't put a cmap in kwargs['plot'] because here we use a
                    # fixed color per cluster.
                    label_geodf.plot(ax=ax, color=label_color[label],
                                    **kwargs.get('plot', {}))
                reg.shape_geodf.plot(ax=ax, color='none', edgecolor='black')
                ax.set_axis_off()

            # The colours will correspond because groupby sorts by the column by
            # which we group, and we sorted the unique labels.
            fig = map_viz.colored_pts_legend(fig, label_color,
                                             **kwargs.get('legend', {}))
            if show:
                fig.show()
            if save_path_fmt:
                fmt_dict = {**self.to_dict(), **asdict(decomposition),
                            **asdict(level)}
                save_path = Path(str(save_path_fmt).format(**fmt_dict))
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)
            fig_list.append(fig)
            axes_list.append(axes)

        return fig_list, axes_list


    def silhouette_plot(self, i_decompo=-1, i_clust=-1, metric=None):
        decomp = self.decompositions[i_decompo]
        clust = decomp.clusterings[i_clust]
        return clust.silhouette_plot(decomp.proj_vectors, metric=metric)
