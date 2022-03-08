import sys
import re
import json
from pathlib import Path
import logging
import logging.config
import datetime
import dateutil
import ray
import geopandas as geopd
import querier
import src.data.access as data_access
import src.data.us_specific as us_data
import src.utils.paths as paths_utils
import src.utils.geometry as geo_utils
import src.utils.places_to_cells as places_to_cells
import src.utils.parallel as parallel
import src.utils.paths as paths_utils
import src.data.word_counts as word_counts
from src.utils.dialects import Region 
from dotenv import load_dotenv
load_dotenv()

LOGGER = logging.getLogger(__name__)
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)

@ray.remote
def remote_chunk_process(
    db, colls, tweets_filter, reg, places_geodf, cells_in_places, lang
):
    # Since Python logger module creates a singleton logger per process, loggers
    # should be configured on per task/actor basis.
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    tweets_df = data_access.tweets_from_mongo(
        db, tweets_filter, colls, add_cols={'source': {'field': 'source', 'dtype': 'string'}}
    )
    LOGGER.info(f'retrieved {tweets_df.shape[0]} tweets from {colls} of {db}')
    cell_counts = word_counts.get_cell_word_counts(
        tweets_df, reg.cells_geodf, places_geodf, cells_in_places, lang
    )
    LOGGER.info('done')
    return cell_counts


@ray.remote(num_returns=2)
def get_places_and_intersect(db, places_filter, reg, tweets_filter, tweets_colls):
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    raw_places_geodf = data_access.places_from_mongo(
        db, places_filter, tweets_filter=tweets_filter, tweets_colls=tweets_colls
    )
    LOGGER.info(f'retrieved {raw_places_geodf.shape[0]} places')
    places_geodf = geo_utils.make_places_geodf(
        raw_places_geodf, reg.shape_geodf, reg.cc, xy_proj=reg.xy_proj
    )
    LOGGER.info(f'filtered down to {places_geodf.shape[0]} places')
    max_area_mask = places_geodf['area'] < reg.max_place_area
    is_poi = places_geodf['area'] == 0
    relevant_bbox = max_area_mask & (~is_poi)
    cells_in_places = places_to_cells.get_intersect(
        reg.cells_geodf, places_geodf.loc[relevant_bbox]
    )
    LOGGER.info(f'computed cells_in_places of size {cells_in_places.shape[0]}')
    return places_geodf, cells_in_places


def get_dt_range_counts(
    db, reg, pre_filter, start, end, bot_ids, places_geodf, cells_in_places, chunksize=1e6
):
    filter = pre_filter.copy()

    cell_counts_refs = []
    raw_cell_counts_ref = []

    chunk_filters = data_access.dt_chunk_filters_mongo(
        db, reg.mongo_coll, filter, start, end, chunksize=chunksize
    )
    LOGGER.info(f'got chunks')
    # Moved this below chunk_filters computation because it dramatically slows down
    # count_entries to add the bot filter.
    filter.none_of('user.id', bot_ids)

    for i, chunk_f in enumerate(chunk_filters):
        LOGGER.info(f'- started chunk {i}')
        tweets_filter = querier.Filter({**filter, **chunk_f})
        cell_counts_refs.append(remote_chunk_process.remote(
            db, reg.mongo_coll, tweets_filter, reg, places_geodf, cells_in_places, reg.lc
        ))
        ready, not_ready = ray.wait(
            cell_counts_refs, num_returns=len(cell_counts_refs), timeout=0.
        )

        idle_cpus = ray.available_resources().get('CPU', 0)
        if  idle_cpus >= 1:
            LOGGER.info(f'{idle_cpus} CPUs are idle')

        if len(ready) > 2*num_cpus:
            LOGGER.info(f'combining {len(ready)} elements')
            comb_list = ray.get(ready) + raw_cell_counts_ref
            raw_cell_counts_ref = parallel.fast_combine(
                word_counts.combine_cell_counts, comb_list
            )
            for r in ready:
                cell_counts_refs.remove(r)

        if len(not_ready) > 1.5*num_cpus:
            LOGGER.info(f'waiting at chunk {i}, pending jobs: {len(not_ready)}')
            _, _ = ray.wait(not_ready, num_returns=1)
            LOGGER.info('waited')

    comb_list = cell_counts_refs + raw_cell_counts_ref
    LOGGER.info(f"** combining for {start.date()} - {end.date()} **")
    raw_cell_counts_ref = parallel.fast_combine(word_counts.combine_cell_counts,
                                                comb_list)
    raw_cell_counts = ray.get(raw_cell_counts_ref)[0]

    region_counts, raw_cell_counts = word_counts.get_reg_counts(raw_cell_counts)
    raw_cell_counts = word_counts.agg_by_lower(raw_cell_counts)
    return region_counts, raw_cell_counts


def main(reg, num_cpus, years, res_fpath_format, **kwargs):
    db_name_fmt = 'twitter_{year}'
    ray.init(num_cpus=num_cpus)
    LOGGER.info(ray.cluster_resources())

    for year in years:
        db = db_name_fmt.format(year=year)

        first_tweets_filter = querier.Filter()
        first_tweets_filter.equals('place.country_code', reg.cc)

        places_filter = querier.Filter()
        places_filter.equals('country_code', reg.cc)
        places_geodf, cells_in_places = get_places_and_intersect.remote(
            db, places_filter, reg, first_tweets_filter, reg.mongo_coll
        )

        bot_ids = data_access.get_bot_ids(
            db, reg.mongo_coll, first_tweets_filter, max_hourly_rate=10
        )
        print(f'{len(bot_ids)} bots detected')

        if '{month}' in str(res_fpath_format):
            month_td = dateutil.relativedelta.relativedelta(months=1)
            year_dt = datetime.datetime(year, 1, 1)
            dt_ranges = [
                (year_dt + i * month_td, year_dt + (i+1) * month_td)
                for i in range(12)
            ]
        else:
            start = datetime.datetime(year, 1, 1)
            end = datetime.datetime(year+1, 1, 1)
            dt_ranges = [(start, end)]

        for start, end in dt_ranges:
            region_counts, raw_cell_counts = get_dt_range_counts(
                db, reg, first_tweets_filter, start, end, bot_ids, places_geodf,
                cells_in_places, chunksize=1e6
            )

            month = start.month
            save_path = Path(str(res_fpath_format).format(
                kind='raw_cell_counts', year_from=year, year_to=year, year=year, month=month
            ))
            raw_cell_counts.to_parquet(save_path, index=True)
            save_path = Path(str(res_fpath_format).format(
                kind='region_counts', year_from=year, year_to=year, year=year, month=month
            ))
            region_counts.to_parquet(save_path, index=True)
            LOGGER.info(f'** {start.date()} - {end.date()} done **')

    ray.shutdown()


if __name__ == '__main__':
    paths = paths_utils.ProjectPaths()
    ext_data_path = paths.ext_data
    with open(ext_data_path / 'countries.json') as f:
        countries_dict = json.load(f)
    all_cntr_shapes = geopd.read_file(
        str(paths.shp_file_fmt).format('CNTR_RG_01M_2016_4326')
    )

    lang = sys.argv[1]
    cc = sys.argv[2]
    year_from = int(sys.argv[3])
    year_to = int(sys.argv[4])
    num_cpus = int(sys.argv[5])

    if year_from < 2015 or year_to > 2021:
        raise ValueError('year range must be comprised between 2015 and 2021')

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
    }
    reg = Region.from_dict(cc, lang, reg_dict)
    _ = reg.get_shape_geodf(all_cntr_shapes=all_cntr_shapes)
    _ = reg.get_cells_geodf()

    res_fpath_format = Path(str(paths.monthly_counts_files_fmt).format(
        lc=lang, cc=cc, kind='{kind}', year='{year}',
        month='{month}', cell_size=reg.cell_size
    ))
    years = range(year_from, year_to + 1)
    
    main(reg, num_cpus, years, res_fpath_format,
         **{'filter': filter_cell_counts_kwargs})
