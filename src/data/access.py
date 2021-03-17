'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column. Cannot use Modin or Dask because of gzip
compression, Parquet data files would be ideal.
'''
import gzip
import logging
import pandas as pd

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
