import os
import re
from pathlib import Path
from dataclasses import dataclass
from string import Formatter
from dotenv import load_dotenv
load_dotenv()


def yield_kw_from_fmt_str(fmt_str):
    '''
    Yield all keywords from a format string.
    '''
    for _, fn, _, _ in Formatter().parse(fmt_str):
        if fn is not None:
            yield fn


def yield_paramed_matches(file_path_format, params_dict):
    '''
    Generator of named matches of the parameterised format `file_path_format`
    for every parameter not fixed in `params_dict`.
    '''
    fname = file_path_format.name
    # Create a dictionary with keys corresponding to each parameter of the file
    # format, and the values being either the one found in `params_dict`, or a
    # regex named capture group of the parameter.
    pformat = {fn: params_dict.get(fn, f'(?P<{fn}>.+)')
               for fn in yield_kw_from_fmt_str(fname)}
    file_pattern = re.compile(fname.replace('.', r'\.').format(**pformat))
    for f in file_path_format.parent.iterdir():
        match = re.search(file_pattern, f.name)
        if match is not None:
            yield match


def partial_format(fmt_str, **kwargs):
    all_kw = list(yield_kw_from_fmt_str(fmt_str))
    fmt_dict = {**{kw: f'{{{kw}}}' for kw in all_kw}, **kwargs}
    return fmt_str.format(**fmt_dict)


def format_path(path_fmt, **kwargs):
    '''
    Utility to apply string formatting to a Path.
    '''
    return Path(str(path_fmt).format(**kwargs))


@dataclass
class ProjectPaths:
    '''
    Dataclass containing all the paths used throughout the project. Defining
    this class allows us to define these only once, ensuring consistency.
    '''
    source_data: Path = Path(os.environ['DATA_DIR'])
    proj: Path = Path(os.environ['PROJ_DIR'])
    oslom: Path = Path(os.environ['OSLOM_DIR'])
    source_fname_fmt: str = '{kind}_{from}_{to}_{cc}.json.gz'
    counts_fname_fmt: str = '{kind}_lang={lc}_cc={cc}.parquet'
    cluster_fig_fname_fmt: str = (
        'clusters_method={method_repr}{kwargs_str}_word_vec_var={word_vec_var}'
        '_decomposition={decomposition}.pdf')
    net_fname_fmt: str = (
        'net_metric={metric}_transfo={transfo_str}_word_vec_var={word_vec_var}'
        '_decomposition={decomposition}.dat')
    source_fmt: Path = None
    proj_data: Path = None
    ext_data: Path = None
    raw_data: Path = None
    interim_data: Path = None
    processed_data: Path = None
    counts_files_fmt: Path = None
    shp_file_fmt: Path = None
    figs: Path = None
    cluster_fig_fmt: Path = None

    def __post_init__(self):
        self.source_fmt = self.source_data / self.source_fname_fmt
        self.proj_data = self.proj / 'data'
        self.ext_data = self.proj_data / 'external'
        self.raw_data = self.proj_data / 'raw'
        self.interim_data = self.proj_data / 'interim'
        self.processed_data = self.proj_data / 'processed'
        self.counts_files_fmt = self.raw_data / self.counts_fname_fmt
        self.shp_file_fmt = self.ext_data / '{0}.shp' / '{0}.shp'
        self.figs = self.proj / 'reports' / 'figures'
        self.cluster_fig_fmt = (self.figs / '{lc}' / '{cc}' /
                                self.cluster_fig_fname_fmt)
        self.net_fmt = (self.processed_data / '{lc}' / '{cc}' /
                        self.net_fname_fmt)


    def partial_format(self, **kwargs):
        self.cluster_fig_fmt = Path(partial_format(str(self.cluster_fig_fmt),
                                                   **kwargs))
        self.net_fmt = Path(partial_format(str(self.net_fmt), **kwargs))
