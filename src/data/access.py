'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column.
'''
import gzip
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)


def yield_tweets_access(tweets_files_paths, tweets_res=None, size=1e9):
    '''
    Yields what we call an access to a tweets' DataFrame, which can either be
    the DataFrame directly if a list `tweets_res` of them is supplied, or the
    arguments of `read_json_wrapper`. The functions applied in a loop started
    from this generator then must have as an argument a "get_df" function to
    finally get a DataFrame (see more detail in comments below).
    Unfortunately we can't make this "get_df" function part of the yield here,
    as the function needs to be pickable (so declared globally) for later use in
    a multiprocessing context.
    '''
    if tweets_res is None:
        # Here get_df = lambda x: read_json_wrapper(*x).
        for file_path in tweets_files_paths:
            for chunk_start, chunk_size in chunkify(file_path, size=size):
                yield (file_path, chunk_start, chunk_size)
    else:
        # In this case get_df = lambda x: x is to be used
        for tweets_df in tweets_res:
            yield tweets_df


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


def read_json_wrapper(file_path, chunk_start, chunk_size):
    '''
    Reads a DataFrame from the json file in 'file_path', starting at the byte
    'chunk_start' and reading 'chunk_size' bytes.
    '''
    for f in yield_gzip(file_path):
        f.seek(chunk_start)
        lines = f.read(chunk_size)
        raw_tweets_df = pd.read_json(lines, lines=True)
        nr_tweets = len(raw_tweets_df)
        LOGGER.info(f'{chunk_size*10**-6:.4g}MB read, {nr_tweets} tweets '
                    'unpacked.')
        return raw_tweets_df


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

