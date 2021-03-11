import geopandas as geopd


def get_intersect(cells_geodf, raw_places_geodf):
    '''
    Get the area of the intersection between the cells in cells_df and the
    places in places_geodf.
    '''
    places_geodf = raw_places_geodf.copy()
    places_geodf['place_id'] = places_geodf.index
    cells_in_places = geopd.overlay(
        places_geodf, cells_geodf, how='intersection')
    cells_in_places['area_intersect'] = cells_in_places.geometry.area
    cells_in_places['ratio'] = (cells_in_places['area_intersect']
                                / cells_in_places['area'])
    return cells_in_places.set_index(['place_id', 'cell_id'])['ratio']


def intersect_to_cells(cells_in_places, cells_df, count_cols):
    '''
    Scales all the counts in the columns `count_cols` by the area of
    intersection of the cells with the places, which must have been computed
    prior to calling this function, in the 'area_intersect' column of
    `cells_in_places`. Then the resulting scaled counts are summed by cell.
    '''
    for col in count_cols:
        cells_in_places[col] = (
            cells_in_places[col] * cells_in_places['ratio'])
        cells_counts = cells_in_places.groupby('cell_id')[col].sum()
        cells_df = cells_df.join(cells_counts, how='left')
    return cells_df
