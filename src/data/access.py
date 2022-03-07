'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column. Cannot use Modin or Dask because of gzip
compression, Parquet data files would be ideal.
'''
from __future__ import annotations
from functools import reduce
from math import ceil
import datetime
import zipfile
import gzip
import bz2
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
import geopandas as geopd
import bson
import querier as qr

import src.utils.geometry as geo_utils


LOGGER = logging.getLogger(__name__)


def yield_tweets_access(file_path, size=1e9):
    '''
    Yield access to the data in `file_path`, either a list of lines of the data
    itself read with `read_json_bytes` or the chunk information to read data
    with `read_json_bytes`.
    '''
    if file_path.stat().st_size > 5e9:
        # When files are too large, we prefer to directly readlines to
        # avoid seeking as in `read_json_bytes`. This adds overhead but it's
        # worth it when files are large enough.
        for f in yield_archive(file_path):
            while True:
                lines = f.read(int(size))
                lines += f.readline()
                if len(lines) > 0:
                    yield lines
                else:
                    break
    else:
        for chunk_start, chunk_size in chunkify(file_path, size=size):
            yield (file_path, chunk_start, chunk_size)


# Better to separate generators (functions with yield) and regular functions
# (terminating with return).
def return_json(file_path, compression='infer'):
    '''
    Returns a DataFrame from a local or remote json file. Not recommended for
    large data files.
    '''
    data = pd.read_json(file_path, lines=True, compression=compression)
    return data


def yield_json(file_path, chunk_size=1000, compression='infer'):
    '''
    Yields a JsonReader from a local or remote json file, reading it it chunks.
    This is more suitable to larger files than `return_json`, however it can't
    be parallelized because it would involve a generator of file handles, which
    can't be serialized so this can't be used with `multiprocessing`.
    '''
    data = pd.read_json(file_path, lines=True, chunksize=chunk_size,
                        compression=compression)
    for raw_df in data:
        yield raw_df


def yield_compressed_file(file_path):
    '''
    Yields a gzip file handler.
    '''
    suffix = Path(file_path).suffix
    if suffix == '.gz':
        module = gzip
    elif suffix == '.bz2':
        module = bz2
    else:
        raise NotImplementedError(f'{suffix} not supported')
    with module.open(file_path, 'rb') as unzipped_f:
        yield unzipped_f


def yield_zip(zip_path, file_name):
    '''
    Yields a file handler of file `file_name` from the zip at `zip_path`.
    '''
    with zipfile.ZipFile(zip_path) as zip_dir:
        with zip_dir.open(file_name) as unzipped_f:
            yield unzipped_f


def yield_archive(file_path):
    '''
    Yields a file handler of a compressed file.
    '''
    if isinstance(file_path, tuple):
        return yield_zip(*file_path)
    else:
        return yield_compressed_file(file_path)


def read_json_bytes(file_path, chunk_start, chunk_size):
    '''
    Reads a DataFrame from the json file in 'file_path', starting at the byte
    'chunk_start' and reading 'chunk_size' bytes.
    '''
    for f in yield_archive(file_path):
        # The following is extremely costly on large files, prefer
        # `read_json_lines` when files are of several GB.
        f.seek(chunk_start)
        lines = f.read(chunk_size)
        df = pd.read_json(lines, lines=True)
        nr_tweets = len(df)
        LOGGER.info(f'{chunk_size*10**-6:.4g}MB read, {nr_tweets} tweets '
                    'unpacked.')
        return df


def read_json_lines(lines):
    '''
    Reads a DataFrame from the string or bytes `lines`.
    '''
    df = pd.read_json(lines, lines=True)
    nr_tweets = len(df)
    LOGGER.info(f'{nr_tweets} tweets unpacked')
    return df


def read_json_wrapper(df_access):
    # If the file was not too large,
    if isinstance(df_access, tuple):
        df = read_json_bytes(*df_access)
    else:
        df = read_json_lines(df_access)
    return df


def chunkify(file_path, size=5e8):
    '''
    Generator going through a json file located in 'file_path', and yielding the
    chunk start and size of (approximate byte) size 'size'. Since we want to
    read lines of data, the function ensures that the end of the chunk
    'chunk_end' is at the end of a line.
    '''
    for f in yield_archive(file_path):
        chunk_end = f.tell()
        while True:
            chunk_start = chunk_end
            # Seek 'size' bytes ahead, relatively to where we are now (second
            # argument = 1)
            f.seek(int(size), 1)
            # Read a line at this point, that is, read until a '\n' is
            # encountered:
            f.readline()
            # Put the end of the chunk at the end of this line:
            chunk_end = f.tell()
            # If the end of the file is reached, f.tell() returns
            # the last byte, even if we keep seeking forward.
            yield chunk_start, chunk_end-chunk_start
            # Because of readline, we'll always read some bytes more than
            # 'size', if it's not the case it means we've reached the end of the
            # file.
            if chunk_end - chunk_start < size:
                break


def places_from_mongo(db, filter, add_fields=None, tweets_filter=None, tweets_colls=None):
    '''
    Return the GeoDataFrame of places in database `db` matching `filter`.
    '''
    if add_fields is None:
        add_fields = []
    default_fields = ['id', 'name', 'place_type', 'bounding_box.coordinates']
    all_fields = default_fields + add_fields

    with qr.Connection(db) as con:

        if 'places' not in con.list_available_collections():
            if tweets_colls is None or tweets_filter is None:
                raise ValueError(
                    f'There is no places collection in {db}, specify tweet '
                    'filters and collections from which to retrieve them.'
                )
            return places_from_mongo_tweets(db, tweets_colls, tweets_filter, add_fields=add_fields)

        places = con['places'].extract(filter, fields=all_fields)

        all_fields.remove('bounding_box.coordinates')
        places_dict = {key: [] for key in all_fields}
        places_dict['geometry'] = []

        for p in places:
            if p.get('bounding_box') is None:
                continue

            for f in all_fields:
                places_dict[f].append(p[f])

            bbox = p['bounding_box']['coordinates'][0]
            geo, _ = geo_utils.geo_from_bbox(bbox)
            places_dict['geometry'].append(geo)

    raw_places_gdf = geopd.GeoDataFrame(places_dict, crs='epsg:4326').set_index('id', drop=False)
    return raw_places_gdf


def places_from_mongo_tweets(db, colls: str | list, tweets_filter, add_fields=None):
    '''
    When no 'places' collection
    '''
    if add_fields is None:
        add_fields = []
    if isinstance(colls, str):
        colls = [colls]

    with qr.Connection(db) as con:
        agg_dict = {
            "name": ("place.name", "first"),
            "type": ("place.place_type", "first"),
            "nr_tweets": ("place.id", "count"),
            "bbox": ("place.bounding_box.coordinates", "first"),
            **{field: (f"place.{field}", "first") for field in add_fields}
        }
        geodf_dicts = []

        for coll in colls:
            places = con[coll].groupby(
                'place.id', pre_filter=tweets_filter, allowDiskUse=True
            ).agg(**agg_dict)

            for p in tqdm(places):
                p['id'] = p.pop('_id')
                bbox = p.pop('bbox')[0]
                p['geometry'], _ = geo_utils.geo_from_bbox(bbox)
                geodf_dicts.append(p)

    raw_places_gdf = (geopd.GeoDataFrame.from_dict(geodf_dicts)
                                        .drop_duplicates(subset='id')
                                        .set_index('id', drop=False)
                                        .set_crs('epsg:4326'))
    return raw_places_gdf


def tweets_from_mongo(db, filter, colls: str | list, add_cols=None):
    '''
    Return the DataFrame of tweets in the collections `colls` of the database
    `db` matching `filter`.
    '''
    if add_cols is None:
        add_cols = {}
    default_cols = {
        'id': {'field': 'id', 'dtype': 'string'},
        'text': {'field': ['extended_tweet.full_text', 'text'], 'dtype': 'string'},
        'coordinates': {'field': 'coordinates.coordinates', 'dtype': 'object'},
        'place_id': {'field': 'place.id', 'dtype': 'string'},
    }
    cols_dict = {**default_cols, **add_cols}
    fields_to_extract = []
    for d in cols_dict.values():
        if isinstance(d['field'], str):
            d['field'] = [d['field']]
        fields_to_extract.extend(d['field'])

    with qr.Connection(db) as con:
        tweets = con[colls].extract(filter, fields=fields_to_extract)

        assign_dict = {
            col: [f.split('.') for f in col_dict['field']]
            for col, col_dict in cols_dict.items()
        }
        tweets_dict = {key: [] for key in cols_dict.keys()}

        last_id = ''
        for t in tweets:
            if t['id'] != last_id:
                last_id = t['id']
                # In the result, nested fields will actually be contained in a nested
                # dict.
                for col, fields in assign_dict.items():
                    for f in fields:
                        value = t.get(f[0])

                        if len(f) > 1 and value is not None:
                            for part in f[1:]:
                                value = value.get(part)

                        if value is not None:
                            break

                    tweets_dict[col].append(value)

    dtypes = {col: d.get('dtype') for col, d in cols_dict.items()}
    tweets_df = pd.DataFrame(tweets_dict).astype(dtypes)
    return tweets_df


def geo_within_bbox_filter(minx, miny, maxx, maxy):
    '''
    From bounds defining a bounding box, return a Mongo filter of geometries
    within this bounding box.
    '''
    return geo_within_filter([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny]
    ])

def geo_within_filter(polygon_coords):
    '''
    From a list of coordinates defining a polygon `polygon_coords`, return a
    Mongo filter of geometries within this polygon.
    '''
    # Close the polygon if not done
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    return {
        '$geoWithin': {
            '$geometry': {
                'coordinates': [polygon_coords],
                'type': 'Polygon'
            }
        }
    }


def get_all_types_mongo_id(id: int | str | bson.ObjectId):
    # only conversion that's not clear how to perform is str, objectid -> int, but very
    # few int "_id" so should be ok to discard them.
    types = [id]
    if isinstance(id, bson.ObjectId):
        types.append(str(id))
    elif isinstance(id, str):
        types.append(bson.ObjectId(id.zfill(24)))
    elif isinstance(id, int):
        types.append(str(id).zfill(24))
        types.append(bson.ObjectId(types[-1]))
    return types


def chunk_filters_mongo(db, colls: str | list, filter, chunksize=1e6):
    '''
    Returns a list of filters of ID ranges that split the elements of collection
    `coll` of MongoDB database `db` matching `filter` into chunks of size
    `chunksize`. From (as I'm writing, not-yet-released) dask-mongo.
    '''
    with qr.Connection(db) as con:
        nrows = con[colls].count_entries(filter)
        npartitions = int(ceil(nrows / chunksize))
        LOGGER.info(
            f'{nrows:.3e} tweets matching the filter in collection {colls} of '
            f'DB {db}, dividing in {npartitions} chunks...'
        )
        # TODO: same but with "groupBy": "$created_at"?? may be slower though?
        partitions_ids = list(
            con[colls].aggregate(
                [
                    {"$match": filter},
                    {"$bucketAuto": {"groupBy": "$_id", "buckets": npartitions}},
                ],
                allowDiskUse=True,
            )
        )

    chunk_filters = []
    greater_fun = lambda id: qr.Filter().greater_or_equals("_id", id)
    less_fun = lambda id: qr.Filter().less_than("_id", id)

    for i, partition in enumerate(partitions_ids):
        # As the type of the "_id" field is not consistent throughout the database, for
        # each bound we have to allow comparison with all possible types. So what we do
        # below is to get chunk_filter = (_id > lower bound for one type at least) and
        # (_id <(=) upper bound for one type at least).
        or_operator = lambda f1, f2: f1 | f2

        min_id = partition["_id"]["min"]
        all_types_min_id = get_all_types_mongo_id(min_id)
        min_id_f = reduce(or_operator, map(greater_fun, all_types_min_id))

        max_id = partition["_id"]["max"]
        all_types_max_id = get_all_types_mongo_id(max_id)
        if i == npartitions - 1:
            less_fun = lambda id: qr.Filter().less_or_equals("_id", id)
        max_id_f = reduce(or_operator, map(less_fun, all_types_max_id))

        chunk_filters.append(min_id_f & max_id_f)

    return chunk_filters


def dt_chunk_filters_mongo(db, colls: str | list, filter, start, end, chunksize=1e6):
    '''
    Returns a list of filters of datetime ranges that split the elements of collection
    `coll` of MongoDB database `db` matching `filter` into chunks of equal time delta.
    Has the advantage of being faster than `chunk_filters_mongo`, to the cost of having
    chunks of unequal sizes.
    '''
    with qr.Connection(db) as con:
        nrows = con[colls].count_entries(filter)
        npartitions = int(ceil(nrows / chunksize))
        LOGGER.info(
            f'{nrows:.3e} tweets matching the filter in collection {colls} of '
            f'DB {db}, dividing in {npartitions} chunks...'
        )

    dt_step = (end - start) / npartitions
    dt_bounds = [start + i*dt_step for i in range(npartitions+1)]

    chunk_filters = []
    for i in range(npartitions):
        f = qr.Filter()
        f.greater_or_equals('created_at', dt_bounds[i])
        f.less_than('created_at', dt_bounds[i+1])
        chunk_filters.append(f)

    return chunk_filters
