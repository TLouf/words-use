import sys
import json
from pathlib import Path
import logging
import logging.config
import geopandas as geopd
import src.data.us_specific as us_data
import src.utils.paths as paths_utils
from src.dialects import Region
from dotenv import load_dotenv
load_dotenv()

from cell_counts_from_mongo import main


LOGGER = logging.getLogger(__name__)
# load config from file
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)


if __name__ == '__main__':
    paths = paths_utils.ProjectPaths()
    ext_data_path = paths.ext_data
    with open(ext_data_path / 'countries.json') as f:
        countries_dict = json.load(f)
    all_cntr_shapes = geopd.read_file(paths.ext_data
                                      / 'CNTR_RG_01M_2016_4326.shp'
                                      / 'CNTR_RG_01M_2016_4326.shp')
    
    lang = sys.argv[1]
    year_from = int(sys.argv[2])
    year_to = int(sys.argv[3])
    num_cpus = int(sys.argv[4])

    list_cc = [key for key, value in countries_dict.items()
               if lang in value['local_langs'] and key != 'ES']
    print(list_cc)
    if year_from < 2015 or year_to > 2021:
        raise ValueError('year range must be comprised between 2015 and 2021')

    for cc in list_cc:
        if cc == 'US':
            cnty_fpath = str(paths.shp_file_fmt).format('cb_2018_us_county_5m')
            state_fpath = str(paths.shp_file_fmt).format('cb_2018_us_state_5m')
            us_dict = countries_dict['US']
            us_dict['shape_geodf'] = us_data.get_states_geodf(
                state_fpath, xy_proj=countries_dict['US']['xy_proj'])
            us_dict['cells_geodf'] = us_data.get_counties_geodf(
                cnty_fpath, us_dict['shape_geodf'],
                xy_proj=countries_dict['US']['xy_proj'])
            us_dict['cell_size'] = 'county'
        
        reg_dict = countries_dict[cc]
        filter_cell_counts_kwargs = {
        # 'min_nr_cells': 3,
        # 'sum_th': 1e4,
        # 'cell_tokens_decade_crit': 2,
        # 'upper_th': 0.4
    }
        reg = Region.from_dict(cc, lang, reg_dict)
        _ = reg.get_shape_geodf(all_cntr_shapes=all_cntr_shapes)
        _ = reg.get_cells_geodf()

        res_fpath_format = Path(str(paths.counts_files_fmt).format(
            lc=lang, cc=cc, kind='{kind}', year_from='{year_from}',
            year_to='{year_to}', cell_size=reg.cell_size
        ))
        years = range(year_from, year_to + 1)
        
        main(reg, num_cpus, years, res_fpath_format,
            **{'filter': filter_cell_counts_kwargs})
