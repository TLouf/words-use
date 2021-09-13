'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column. Cannot use Modin or Dask because of gzip
compression, Parquet data files would be ideal.
'''
from math import ceil
import gzip
import logging
from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as geopd
import querier

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
        for f in yield_gzip(file_path):
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


def yield_gzip(file_path):
    '''
    Yields a gzip file handler from a remote or local directory.
    '''
    with gzip.open(file_path, 'rb') as unzipped_f:
        yield unzipped_f


def read_json_bytes(file_path, chunk_start, chunk_size):
    '''
    Reads a DataFrame from the json file in 'file_path', starting at the byte
    'chunk_start' and reading 'chunk_size' bytes.
    '''
    for f in yield_gzip(file_path):
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
    for f in yield_gzip(file_path):
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


def places_from_mongo(db, filter, add_fields=None):
    '''
    Return the GeoDataFrame of places in database `db` matching `filter`.
    '''
    if add_fields is None:
        add_fields = []
    default_fields = ['id', 'name', 'place_type', 'bounding_box.coordinates']
    all_fields = default_fields + add_fields
    with querier.Connection(db) as con:
        places = con.extract(
            filter,
            fields=all_fields,
            collections_subset=['places']
        )

        all_fields.remove('bounding_box.coordinates')
        places_dict = {key: [] for key in all_fields}
        places_dict['geometry'] = []

        for p in places:

            for f in all_fields:
                places_dict[f].append(p[f])

            bbox = p['bounding_box']['coordinates'][0]
            if bbox[0] == bbox[1]:
                geo = Point(bbox[0])
            else:
                geo = Polygon(bbox)
            places_dict['geometry'].append(geo)

    raw_places_gdf = geopd.GeoDataFrame(places_dict, crs='epsg:4326').set_index('id')
    return raw_places_gdf


def tweets_from_mongo(db, filter, colls, add_fields=None):
    '''
    Return the DataFrame of tweets in the collections `colls` of the database
    `db` matching `filter`.
    '''
    if add_fields is None:
        add_fields = {}
    default_fields = {
        'text': 'text',
        'coordinates.coordinates': 'coordinates',
        'place.id': 'place_id'
    }
    fields_to_cols = {**default_fields, **add_fields}
    all_fields = list(fields_to_cols.keys())

    with querier.Connection(db) as con:
        tweets = con.extract(
            filter,
            fields=all_fields,
            collections_subset=colls
        )

        all_fields = [f.split('.') for f in all_fields]
        col_names = fields_to_cols.values()
        tweets_dict = {key: [] for key in col_names}

        for t in tweets:
            for field, col in zip(all_fields, col_names):
                value = t.get(field[0])

                if len(field) > 1 and value is not None:
                    for part in field[1:]:
                        value = value.get(part)

                tweets_dict[col].append(value)

    tweets_df = pd.DataFrame(tweets_dict).astype({'text': 'string', 'place_id': 'string'})
    return tweets_df


def geo_within_filter(minx, miny, maxx, maxy):
    '''
    From bounds defining a bounding box, return a Mongo filter of geometries
    within this bounding box.
    '''
    return {
        '$geoWithin': {
            '$geometry': {
                'coordinates': [[
                    [minx, miny],
                    [minx, maxy],
                    [maxx, maxy],
                    [maxx, miny],
                    [minx, miny]
                ]],
                'type': 'Polygon'
            }
        }
    }


def chunk_filters_mongo(db, coll, filter, chunksize=1e6):
    '''
    Returns a list of filters of ID ranges that split the elements of collection
    `coll` of MongoDB database `db` matching `filter` into chunks of size
    `chunnksize`. From (as I'm writing, not-yet-released) dask-mongo.
    '''
    match = filter._query
    with querier.Connection(db) as con:
        nrows = con.count_entries(filter, collection=coll)
        npartitions = int(ceil(nrows / chunksize))
        partitions_ids = list(
            con._db[coll].aggregate(
                [
                    {"$match": match},
                    {"$bucketAuto": {"groupBy": "$_id", "buckets": npartitions}},
                ],
                allowDiskUse=True,
            )
        )

    chunk_filters = [
        {
            "_id": {
                "$gte": partition["_id"]["min"],
                "$lte" if i == npartitions - 1 else "$lt": partition["_id"]["max"]
            }
        }
        for i, partition in enumerate(partitions_ids)
    ]
    
    return chunk_filters
