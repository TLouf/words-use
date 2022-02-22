from __future__ import annotations

import re
import json
import pickle
import inspect
import copy
from pathlib import Path
from dataclasses import dataclass, field, InitVar, asdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
    mongo_coll: str = ''
    year_from: int = 2015
    year_to: int = 2021
    readable: str = ''
    xy_proj: str = 'epsg:3857'
    max_place_area: float = 5e9
    cell_size: float | str = 50e3
    shape_bbox: list[float] | None = None
    shapefile_col: str = 'FID'
    shape_geodf: geopd.GeoDataFrame | None = None
    cells_geodf: geopd.GeoDataFrame | None = None
    cell_counts: pd.DataFrame | None = None
    region_counts: pd.DataFrame | None = None
    raw_cell_counts: pd.DataFrame | None = None
    shapefile_val: str | None = None
    total_bounds: np.ndarray | None = None

    def __post_init__(self):
        self.shapefile_val = self.shapefile_val or self.cc


    def __repr__(self):
        field_dict = self.__dataclass_fields__
        attr_str_components = []
        for key in field_dict.keys():
            field = getattr(self, key)
            size = getattr(field, 'size', 0)
            if isinstance(size, int) and size < 10:
                attr_str_components.append(f'{key}={field!r}')

        attr_str = ', '.join(attr_str_components)
        return f'{self.__class__.__name__}({attr_str})'


    @classmethod
    def from_dict(cls, cc, lc, d , **kwargs):
        return cls(cc=cc, lc=lc, **kwargs, **{
            k: v for k, v in d.items()
            if k in inspect.signature(cls).parameters
        })


    @staticmethod
    def from_global_json(file_path, cc, lc, **kwargs):
        with open(file_path) as f:
            countries_dict = json.load(f)
        d = countries_dict[cc]
        return Region.from_dict(cc, lc, d, **kwargs)


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'cc', 'year_from', 'year_to', 'cell_size',
            'max_place_area', 'xy_proj'
        ]
        return {attr: getattr(self, attr) for attr in list_attr}


    def get_shape_geodf(self, all_cntr_shapes=None, simplify_tol=100):
        if self.shape_geodf is None:
            col = self.shapefile_col
            mask = all_cntr_shapes[col].str.startswith(self.shapefile_val)
            self.shape_geodf = all_cntr_shapes.loc[mask]
            self.shape_geodf = geo.extract_shape(
                self.shape_geodf, self.cc, xy_proj=self.xy_proj,
                bbox=self.shape_bbox, simplify_tol=simplify_tol)
        return self.shape_geodf


    def get_total_bounds(self, crs='epsg:4326'):
        if self.total_bounds is None:
            shape_geodf = self.get_shape_geodf()
            total_bounds = shape_geodf.geometry.to_crs(crs).total_bounds
            self.total_bounds = total_bounds
        return self.total_bounds


    def get_cells_geodf(self, **kwargs):
        if self.cells_geodf is None:
            self.shape_geodf = self.get_shape_geodf(**kwargs)
            _, self.cells_geodf, _, _ = geo.create_grid(
                self.shape_geodf, self.cell_size, self.cc,
                xy_proj=self.xy_proj, intersect=True)
        return self.cells_geodf


    def read_counts(self, df_name, files_fmt=None):
        if hasattr(self, df_name):
            if getattr(self, df_name) is None:
                reg_dict = self.to_dict()
                parquet_file = Path(str(files_fmt).format(kind=df_name, **reg_dict))
                if parquet_file.exists():
                    res = pd.read_parquet(parquet_file)
                else:
                    files_to_read = []
                    reg_dict['year_from'] = '{year_from}'
                    reg_dict['year_to'] = '{year_to}'
                    year_patt = Path(
                        str(files_fmt)
                         .format(kind=df_name, **reg_dict)
                         .replace('{year_from}', '([0-9]{4})')
                         .replace('{year_to}', '([0-9]{4})')
                         .replace('.', '\.')
                    )

                    for f in year_patt.parent.iterdir():
                        match = re.search(str(year_patt), str(f))
                        if match is not None:
                            years = [int(y) for y in match.groups()]
                            # Assumes no overlap
                            if years[0] >= self.year_from and years[1] <= self.year_to:
                                files_to_read.append(f)

                    pbar = tqdm(enumerate(files_to_read), total=len(files_to_read))
                    for i, f in pbar:
                        pbar.set_description(f.name)
                        if i == 0:
                            res = pd.read_parquet(f)
                        else:
                            res = res.add(pd.read_parquet(f), fill_value=0)

                setattr(self, df_name, res)

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
    list_cc: list[str]
    regions: list[Region]
    _paths: paths_utils.ProjectPaths
    all_cntr_shapes: InitVar[geopd.GeoDataFrame] = None
    year_from: int = 2015
    year_to: int = 2021
    cc: str = field(init=False)
    latlon_proj: str = 'epsg:4326'
    min_nr_cells: int = 10
    upper_th: float = 0.4
    cell_tokens_th: float = 1e4
    cell_tokens_decade_crit: float = 2.
    word_tokens_th: float = 0
    max_word_rank: int | None = None
    cells_geodf: geopd.GeoDataFrame = field(init=False)
    global_counts: pd.DataFrame | None = None
    raw_cell_counts: pd.DataFrame | None = None
    words_prior_mask: pd.Series | None = None
    cell_counts: pd.DataFrame | None = None
    relevant_cells: pd.Index | None = None # necessarily sorted given that it's created from a groupby
    word_counts_vectors: word_counts.WordCountsVectors | None = None
    word_vec_var: str = ''
    word_vectors: word_counts.WordVectors | None = None
    cdf_th: float = 0.99
    width_ratios: np.ndarray | None = None
    decompositions: list[data_clustering.Decomposition] = field(default_factory=list)

    def __post_init__(self, all_cntr_shapes):
        self.list_cc, self.regions = zip(*sorted(
            zip(self.list_cc, self.regions), key=lambda t: t[0]))
        self.cc = '-'.join(self.list_cc)
        self.cells_geodf = pd.concat([
            reg.get_cells_geodf(all_cntr_shapes=all_cntr_shapes)
               .to_crs(self.latlon_proj)
            for reg in self.regions
        ]).sort_index()
        # To get the shape anyway when `cells_geodf` has been provided in init.
        for reg in self.regions:
            _ = reg.get_shape_geodf(all_cntr_shapes)


    def __repr__(self):
        field_dict = self.__dataclass_fields__
        persistent_field_keys = [
            key
            for key, value in field_dict.items()
            if not isinstance(value.type, InitVar)
        ]
        attr_str_components = []
        for key in persistent_field_keys:
            field = getattr(self, key)
            size = getattr(field, 'size', 0)
            if isinstance(size, int) and size < 10:
                attr_str_components.append(f'{key}={field!r}')

        attr_str = ', '.join(attr_str_components)
        return f'{self.__class__.__name__}({attr_str})'


    @classmethod
    def from_other(cls, other: Language):
        field_dict = other.__dataclass_fields__
        init_field_keys = [
            key
            for key, value in field_dict.items()
            if value.init and not isinstance(value.type, InitVar)
        ]
        other_attrs = {
            key: getattr(other, key, field_dict[key].default)
            for key in init_field_keys
        }
        return cls(**other_attrs)


    @classmethod
    def from_countries_dict(cls, lc, readable, list_cc, countries_dict,
                            all_cntr_shapes, paths, year_from=2015,
                            year_to=2021, **kwargs):
        list_cc.sort()
        regions = [Region.from_dict(cc, lc, countries_dict[cc],
                                    year_from=year_from, year_to=year_to)
                   for cc in list_cc]
        return cls(
            lc, readable, list_cc, regions, paths, all_cntr_shapes,
            year_from=year_from, year_to=year_to, **kwargs
        )


    def data_file_fmt(self, save_dir, add_keys=None):
        add_keys = add_keys or []
        keys = ['lc', 'cc', 'min_nr_cells', 'cell_tokens_decade_crit',
                'cell_tokens_th', 'cdf_th'] + add_keys
        params_str = '_'.join(
            [f'{key}={{{key}}}' for key in keys])
        save_path_fmt = str(save_dir / f'{{kind}}_{params_str}.{{ext}}')
        return save_path_fmt


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'cc', 'min_nr_cells', 'cell_tokens_decade_crit',
            'cell_tokens_th', 'cdf_th', 'year_from', 'year_to', 'max_word_rank',
            'upper_th',
        ]
        return {attr: getattr(self, attr) for attr in list_attr}


    @property
    def paths(self):
        '''
        Defined as a property so that when we pass paths to eg a child
        Decomposition or Clustering, the paths is already partially formatted
        with the Language attributes.
        '''
        self._paths = paths_utils.ProjectPaths()
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
            pbar = tqdm(self.regions)
            for reg in pbar:
                pbar.set_description(reg.cc)
                reg_counts = reg.read_counts(
                    'region_counts', files_fmt=self.paths.counts_files_fmt
                )
                reg_counts['is_proper'] = (
                    reg_counts['count_upper'] / reg_counts['count'] > self.upper_th
                )
                self.global_counts = self.global_counts.add(
                    reg_counts[cols], fill_value=0
                )
        self.global_counts = self.global_counts.sort_values(
            by='count', ascending=False
        )
        return self.global_counts


    def filter_global_counts(self, mask=None, invert=False, filter_col=None):
        global_counts = self.get_global_counts()

        if mask is None:
            proper_mask = global_counts['is_proper'] == 0
            nr_cells_mask = global_counts['nr_cells'] >= self.min_nr_cells
            words_mask = proper_mask & nr_cells_mask
        elif isinstance(mask, pd.Series):
            words_mask = mask
        elif isinstance(mask, pd.Index):
            words_mask = pd.Series(False, index=global_counts.index)
            intersect_index = global_counts.index.intersection(mask)
            words_mask.loc[intersect_index] = True
        else:
            raise TypeError('mask is of wrong type')

        if invert:
            # useful when wish to exclude words in an Index given by `mask`.
            words_mask = ~words_mask

        # TODO: filter_col useful?
        if filter_col is None:
            self.words_prior_mask = words_mask
            global_counts = global_counts.loc[self.words_prior_mask].copy()
            if self.max_word_rank is not None:
                self.max_word_rank = (
                    words_mask.values
                    & (words_mask.reset_index().index < self.max_word_rank)
                ).sum()

        else:
            global_counts[filter_col] = words_mask

        self.global_counts = global_counts
        return global_counts


    def make_cell_counts_mask(self, mask=None, invert=False):
        '''
        Have this mask to select words from `raw_cell_counts` to keep in
        `cell_counts`, while not deleting anything from `global_counts`, as
        opposed to how `filter_global_counts` works.
        '''
        # mask appplided to raw_cell_counts to get cell_counts
        self.global_counts['cell_counts_mask'] = False
        col = self.global_counts.columns.get_loc('cell_counts_mask')
        self.global_counts.iloc[:self.max_word_rank, col] = True
        word_tokens_mask = self.global_counts['count'] < self.word_tokens_th
        self.global_counts.loc[word_tokens_mask, 'cell_counts_mask'] = False
        if mask is not None:
            if isinstance(mask, pd.Series):
                words_mask = mask
            elif isinstance(mask, pd.Index):
                words_mask = pd.Series(False, index=self.global_counts.index)
                intersect_index = self.global_counts.index.intersection(mask)
                words_mask.loc[intersect_index] = True

            if invert:
                # Useful when wish to exclude words in an Index
                words_mask = ~words_mask

            self.global_counts['cell_counts_mask'] = (
                self.global_counts['cell_counts_mask'] & words_mask
            )


    def get_raw_cell_counts(self):
        if self.raw_cell_counts is None:
            to_concat = []
            pbar = tqdm(self.regions)
            for reg in pbar:
                pbar.set_description(reg.cc)
                to_concat.append(
                    reg.read_counts(
                        'raw_cell_counts', files_fmt=self.paths.counts_files_fmt
                    )
                )
            self.raw_cell_counts = pd.concat(to_concat).sort_index(axis=0, level=0)
        return self.raw_cell_counts


    def get_cell_counts(self):
        if self.cell_counts is None:
            raw_cell_counts = self.get_raw_cell_counts()
            total_nr_tokens = raw_cell_counts['count'].sum()
            if self.words_prior_mask is None:
                _ = self.filter_global_counts()
            if 'cell_counts_mask' not in self.global_counts.columns:
                self.make_cell_counts_mask()
            cell_counts = word_counts.filter_part_multidx(
                raw_cell_counts, [self.global_counts['cell_counts_mask']]
            )

            # Reindex to have all cells (useful when self.cell_tokens_th == 0).
            cell_sum = (raw_cell_counts.groupby('cell_id')['count']
                                       .sum()
                                       .reindex(self.cells_geodf.index)
                                       .fillna(0))
            sum_th = self.cell_tokens_th
            cell_is_relevant = cell_sum >= sum_th
            self.relevant_cells = cell_is_relevant.loc[cell_is_relevant].index
            print(f'Keeping {self.relevant_cells.shape[0]} cells out of '
                  f'{cell_is_relevant.shape[0]} with threshold {sum_th:.2e}')
            cell_counts = word_counts.filter_part_multidx(
                cell_counts, [cell_is_relevant])

            filtered_nr_tokens = cell_counts['count'].sum()
            rel_diff = (total_nr_tokens - filtered_nr_tokens) / total_nr_tokens
            print(f'We had {total_nr_tokens:.0f} tokens, and filtering ',
                  f'brought it down to {filtered_nr_tokens:.0f}, so we lost ',
                  f'{100*rel_diff:.3g}%.')
            self.cell_counts = cell_counts
        return self.cell_counts


    def save_interim(self, add_keys=None):
        save_path_fmt = self.data_file_fmt(self.paths.interim_data,
                                           add_keys=add_keys)
        self_dict = self.to_dict()
        save_path = save_path_fmt.format(kind='word_counts_vectors',
                                         **self_dict, ext='csv.gz')
        np.savetxt(save_path, self.word_counts_vectors,
                   delimiter=',', fmt='%i')
        save_path = save_path_fmt.format(
            kind=f'word_vectors_word_vec_var={self.word_counts_vectors.word_vec_var}',
            **self_dict, ext='csv.gz')
        np.savetxt(save_path, self.word_vectors, delimiter=',')
        save_path = save_path_fmt.format(kind='relevant_cells',
                                         **self_dict, ext='csv.gz')
        np.savetxt(save_path, self.relevant_cells.values,
                   delimiter=',', fmt="%s")
        save_path = save_path_fmt.format(kind='global_counts',
                                         **self_dict, ext='parquet')
        self.global_counts.to_parquet(save_path, index=True)


    def load_interim(self, add_keys=None):
        save_path_fmt = self.data_file_fmt(self.paths.interim_data,
                                           add_keys=add_keys)
        self_dict = self.to_dict()
        save_path = save_path_fmt.format(kind='word_counts_vectors',
                                         **self_dict, ext='csv.gz')
        word_counts_vectors = np.loadtxt(save_path, delimiter=',', dtype=int)
        self.word_counts_vectors = word_counts.WordCountsVector(array=word_counts_vectors)
        save_path = save_path_fmt.format(
            kind=f'word_vectors_word_vec_var={self.word_vec_var}',
            **self_dict, ext='csv.gz')
        if Path(save_path).exists():
            np.loadtxt(save_path, delimiter=',')
        else:
            print(f'word_vectors_word_vec_var={self.word_vec_var} is not saved')
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
            min_lon, min_lat, max_lon, max_lat = reg.get_total_bounds()
            # For a given longitude extent, the width is maximum the closer to the
            # equator, so the closer the latitude is to 0.
            eq_not_crossed = int(min_lat * max_lat > 0)
            lat_max_width = min(abs(min_lat), abs(max_lat)) * eq_not_crossed
            width = geo.haversine(min_lon, lat_max_width,
                                  max_lon, lat_max_width)
            height = geo.haversine(min_lon, min_lat, min_lon, max_lat)
            self.width_ratios[i] = width / height
        if ratio_lgd:
            self.width_ratios[-1] = ratio_lgd
        else:
            self.width_ratios = self.width_ratios[:-1]
        return self.width_ratios


    def get_word_counts_vectors(self, **kwargs):
        word_mask_col = kwargs.get('word_mask_col')
        if word_mask_col is not None:
            if kwargs.get('cell_sums') is None:
                cell_sums = (self.raw_cell_counts.groupby('cell_id')['count']
                                                 .sum()
                                                 .reindex(self.relevant_cells)
                                                 .fillna(0)
                                                 .values)
                kwargs['cell_sums'] = cell_sums

        if self.word_counts_vectors is None:
            self.word_counts_vectors = word_counts.WordCountsVectors.from_lang(self, **kwargs)
        return self.word_counts_vectors


    def set_cdf_th(self, th):
        #TODO remove
        self.cdf_th = th
        self.word_counts_vectors = None
        self.word_vectors = None
        _ = self.get_word_counts_vectors()


    def get_word_vectors(self, **kwargs):
        if self.word_vectors is None:
            self.word_vectors = word_counts.WordVectors.from_lang(self, **kwargs)
        return self.word_vectors


    def set_word_vec_var(self, word_vec_var):
        #TODO remove
        if word_vec_var != self.word_vec_var:
            self.word_vec_var = word_vec_var
            self.word_vectors = None
        _ = self.get_word_vectors(word_vec_var=word_vec_var)


    def calc_morans(self, num_cpus=1):
        contiguity = libpysal.weights.Queen.from_dataframe(
            self.cells_geodf.loc[self.relevant_cells])
        contiguity.transform = 'r'
        ray.init(num_cpus=num_cpus)
        num_morans = self.word_vectors.shape[1]
        shared_word_vectors = ray.put(self.word_vectors)
        obj_refs = parallel.split_task(
            data_clustering.chunk_moran, list(range(num_morans)),
            shared_word_vectors, contiguity, num_cpus=num_cpus)
        res = ray.get(obj_refs)
        moran_dict = res[0]
        for m_dict in res[1:]:
            for key, value in m_dict.items():
                moran_dict[key].extend(value)
        ray.shutdown()
        cell_counts_mask = self.global_counts['cell_counts_mask']
        words = self.global_counts.loc[cell_counts_mask].index[:num_morans]
        moran_df = pd.DataFrame.from_dict(moran_dict).set_index(words)
        self.global_counts = self.global_counts.join(moran_df)
        return self.global_counts


    def filter_word_vectors(self, z_th=10, p_th=0.01, var_th=None):
        word_vectors = self.get_word_vectors()
        word_vectors.z_th = z_th
        word_vectors.p_th = p_th
        word_vectors.var_th = var_th
        word_vectors = word_vectors.filter(self)
        self.word_vectors = word_vectors


    def make_decomposition(self, from_other: Language | None = None, **kwargs):
        word_mask = self.global_counts['cell_counts_mask']
        word_vectors = self.word_vectors
        if from_other is None:
            pca = PCA(**kwargs).fit(word_vectors)

            if kwargs.get('n_components') is None:
                # If number of components is not specified, select them using the
                # broken stick rule.
                var_pca = pca.explained_variance_ratio_
                var_broken_stick = data_clustering.broken_stick(word_vectors.shape[1])
                # We keep components until they don't explain more than what would be
                # expected from a random partition of the variance into a number of
                # parts equal to the number of words.
                size = min(*word_vectors.shape)
                n_components = np.argmin(var_pca[:size] > var_broken_stick[:size]) - 1
                print(n_components)
                pca = data_clustering.select_components(pca, n_components)
                print(pca.n_components_)
        else:
            # recalculate everyhting or adapt/mask?
            other_decomp = from_other.decompositions[-1]
            pca = copy.deepcopy(other_decomp.decomposition)
            other_word_mask = from_other.global_counts['cell_counts_mask'].rename('other_word_mask')
            iloc = np.arange(word_mask.shape[0])
            joined_masks = (
                other_word_mask.to_frame()
                 .join(word_mask.to_frame().assign(iloc=iloc), how='left')
                 .fillna({'cell_counts_mask': False, 'other_word_mask': False}))
            joined_masks['comb'] = (
                joined_masks['cell_counts_mask'] & joined_masks['other_word_mask']
            )
            word_mask = joined_masks['comb'].rename('cell_counts_mask')
            joined_masks = joined_masks.reset_index()

            # other_relevant_cells = from_other.relevant_cells
            # TODO: solve word shape mismatch
            # select_idc = (joined_masks.loc[joined_masks['cell_counts_mask'], 'iloc']
            select_idc = (joined_masks.loc[joined_masks['comb'], 'iloc']
                          .sort_values()
                          .values
                          .astype(int))
            # select_idc = (joined_masks.loc[joined_masks['cell_counts_mask']]
                        #   .sort_values(by='iloc')['iloc'])
            # word_vectors = word_vectors[cell_mask, word_mask]
            word_vectors = word_vectors[:, select_idc]

            idc_word_mask = joined_masks.loc[joined_masks['other_word_mask'], 'comb']
            pca.components_ = pca.components_[:, idc_word_mask.values]
            # double argsort() to keep order but cast from whatever range of
            # values to 0...n, with n  the length of the array.
            reorder_idc = joined_masks.loc[joined_masks['comb'], 'iloc'].values.astype(int).argsort().argsort()
            n_features = reorder_idc.shape[0]
            # reorder_idc = np.arange(n_features)
            pca.components_ = pca.components_[:, reorder_idc]
            pca.n_features_ = n_features
            pca.n_features_in_ = n_features
            pca.mean_ = np.mean(word_vectors, axis=0)

        print(word_vectors.shape)
        proj_vectors = pca.transform(word_vectors)
        print(proj_vectors.shape)
        decomposition = data_clustering.Decomposition(
            self.word_counts_vectors, word_vectors, pca, proj_vectors, word_mask,
        )
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
        cluster_labels = levels[i_lvl].labels
        unique_labels = np.unique(cluster_labels)
        is_regional = self.global_counts['is_regional']
        clust_words = self.global_counts.loc[is_regional].copy()

        for lbl in unique_labels:
            cluster_mask = cluster_labels == lbl
            clust_center = decomp.proj_vectors[cluster_mask, :].mean(axis=0)
            # We reproject the cluster's center to word space
            # use approx inverse_transform (very bad for low variance ratio threshold in
            # pca) or original word_vectors?
            words_cluster = decomp.decomposition.inverse_transform(clust_center)
            clust_words[f'cluster{lbl}'] = words_cluster

        for lbl in np.unique(cluster_labels):
            other_clust_cols = [
                col
                for col in clust_words.columns
                if col.startswith('cluster') and col != f'cluster{lbl}'
            ]
            # Dist to closest cluster for every word.
            clust_words[f'dist{lbl}'] = np.min(
                [
                    (clust_words[f'cluster{lbl}'] - clust_words[col]).values**2
                    for col in other_clust_cols
                ],
                axis=0
            )
        return clust_words


    def get_maps_pos(self, total_width, total_height=None, ratio_lgd=None):
        width_ratios = self.get_width_ratios(ratio_lgd=ratio_lgd)
        return map_viz.position_axes(
            width_ratios, total_width, total_height=total_height, ratio_lgd=ratio_lgd
        )


    def map_continuous_choro(
        self, z_plot, normed_bboxes: bool | np.ndarray = True,
        total_width=178, total_height=None, axes=None, cax=None,
        cbar_kwargs=None, **choro_kwargs
    ):
        if normed_bboxes is True:
            # calculate optimal position
            normed_bboxes, (total_width, total_height) = self.get_maps_pos(
                total_width, total_height=total_height, ratio_lgd=1/10
            )

        if axes is None:
            figsize = (total_width/10/2.54, total_height/10/2.54)
            _, axes = plt.subplots(len(self.regions) + 1, figsize=figsize)
            cax = axes[-1]
            axes = axes[:-1]
        
        plot_series = pd.Series(z_plot, index=self.relevant_cells, name='z')

        fig, axes = map_viz.choropleth(
            plot_series, self.regions, axes=axes, cax=cax,
            cbar_kwargs=cbar_kwargs, **choro_kwargs
        )

        # if normed_bboxes set to False, don't position the axes
        if not normed_bboxes is False:
            for ax, bbox in zip(np.append(axes, cax), normed_bboxes):
                ax.set_position(bbox)

        return fig, axes


    def map_word(self, word, vcenter=0, vmin=None, vmax=None, cmap='bwr',
                 cbar_kwargs=None, **plot_kwargs):
        cbar_label = f'{self.word_vectors.word_vec_var} of {word}'
        is_regional = self.global_counts['is_regional']
        word_idx = self.global_counts.loc[is_regional].index.get_loc(word)
        z_plot = self.word_vectors[:, word_idx]

        fig, axes = self.map_continuous_choro(
            z_plot, cmap=cmap, vcenter=vcenter, vmin=vmin, vmax=vmax,
            cbar_label=cbar_label, cbar_kwargs=cbar_kwargs, **plot_kwargs
        )
        return fig, axes


    def map_comp(self, i_decompo=-1, comps=None, cmap='bwr',
                 total_width=178, total_height=None, save_path_fmt='',
                 normed_bboxes=True, cbar_kwargs=None, **plot_kwargs):
        if normed_bboxes is True:
            normed_bboxes, (total_width, total_height) = self.get_maps_pos(
                total_width, total_height=total_height, ratio_lgd=1/10
            )

        decomp = self.decompositions[i_decompo]
        proj_vectors = decomp.proj_vectors
        if comps is None:
            comps = range(proj_vectors.shape[1])
        for i in comps:
            cbar_label = f'Component {i+1}'
            z_plot = proj_vectors[:, i]
            save_path = Path(str(save_path_fmt).format(
                **self.to_dict(),
                **decomp.word_counts_vectors.to_dict(),
                **decomp.word_vectors.to_dict(),
                component=i+1,
                **asdict(decomp),
            ))
            _, _ = self.map_continuous_choro(
                z_plot, normed_bboxes=normed_bboxes, total_width=total_width,
                total_height=total_height, cmap=cmap, cbar_label=cbar_label,
                vcenter=0, save_path=save_path, cbar_kwargs=cbar_kwargs, **plot_kwargs)


    def map_clustering(self, i_decompo=-1, i_clust=-1, total_width=178,
                       total_height=None, cmap=None, show=True,
                       save_path_fmt=None, levels_subset=None, **kwargs):
        # total width in mm
        normed_bboxes, (total_width, total_height) = self.get_maps_pos(
            total_width, total_height=total_height
        )
        figsize = (total_width/10/2.54, total_height/10/2.54)
        fig_list = []
        axes_list = []
        decomposition = self.decompositions[i_decompo]
        clustering = decomposition.clusterings[i_clust]
        plot_levels = getattr(clustering, 'levels', [clustering])
        if levels_subset is not None:
            plot_levels = [plot_levels[lvl] for lvl in levels_subset]

        for level in plot_levels:
            if save_path_fmt:
                fmt_dict = {**self.to_dict(),
                            **asdict(decomposition),
                            **decomposition.word_counts_vectors.to_dict(),
                            **decomposition.word_vectors.to_dict(),
                            **asdict(level)}
                save_path = Path(str(save_path_fmt).format(**fmt_dict))
            else:
                save_path = None

            fig, axes = map_viz.cluster_level(
                level, self.regions, figsize=figsize, cmap=cmap,
                save_path=save_path, show=show, **kwargs
            )
            for ax, bbox in zip(axes, normed_bboxes):
                ax.set_position(bbox)
            fig_list.append(fig)
            axes_list.append(axes)

        return fig_list, axes_list


    def silhouette_plot(self, i_decompo=-1, i_clust=-1, metric=None):
        decomp = self.decompositions[i_decompo]
        clust = decomp.clusterings[i_clust]
        return clust.silhouette_plot(decomp.proj_vectors, metric=metric)
