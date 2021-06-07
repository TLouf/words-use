import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, box
import geopandas as geopd

def haversine(lon1, lat1, lon2, lat2):
    R = 6367
    lon1 = lon1 * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lon2 = lon2 * np.pi / 180
    lat2 = lat2 * np.pi / 180

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d


def gen_quadrants(left_x, right_x, bot_y, top_y, places_geodf, sub_cells_list,
                  min_size=1000, max_nr_places=10):
    '''
    Recursively splits each quadrant of the cell defined by the coordinates
    (`left_x`, `right_x`, `bot_y`, `top_y`), as long as there are more than
    `max_nr_places` contained within a subcell, stopping at a minimum cell size
    of `min_size`
    '''
    mid_x = (left_x + right_x) / 2
    mid_y = (bot_y + top_y) / 2
    for quadrant in [(left_x, mid_x, bot_y, mid_y),
                     (left_x, mid_x, mid_y, top_y),
                     (mid_x, right_x, bot_y, mid_y),
                     (mid_x, right_x, mid_y, top_y)]:
        nr_contained_places = places_geodf.loc[
            (places_geodf['x_c'] < quadrant[1])
            & (places_geodf['x_c'] > quadrant[0])
            & (places_geodf['y_c'] < quadrant[3])
            & (places_geodf['y_c'] > quadrant[2])].shape[0]
        if (nr_contained_places > max_nr_places
                and quadrant[1] - quadrant[0] > 2*min_size):
            gen_quadrants(*quadrant, places_geodf, sub_cells_list)
        else:
            sub_cells_list.append(quadrant)


def create_grid(shape_df, cell_size, xy_proj='epsg:3857', intersect=False,
                places_geodf=None, max_nr_places=None):
    '''
    Creates a square grid over a given shape.
    `shape_df` (GeoDataFrame): single line GeoDataFrame containing the shape on
    which the grid is to be created, in lat,lon coordinates.
    `cell_size` (int): size of the sides of the square cells which constitute
    the grid, in meters.
    `intersect` (bool): determines whether the function computes
    `cells_in_shape_df`, the intersection of the created grid with the shape of
    the area of interest, so that the grid only covers the shape.
    If `places_geodf` is given, cells are split in 4 when they countain more
    than `max_nr_places` places from this data frame.
    '''
    if places_geodf is not None:
        places_geodf['x_c'], places_geodf['y_c'] = zip(
            *places_geodf.geometry.apply(lambda geo: geo.centroid.coords[0]))
        if max_nr_places is None:
            max_nr_places = places_geodf.shape[0] // 100
    # We have a one line dataframe, we'll take only the geoseries with the
    # geometry, create a new dataframe out of it with the bounds of this series.
    # Then the values attribute is a one line matrix, so we take the first
    # line and get the bounds' in the format (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = shape_df['geometry'].total_bounds
    # We want to cover at least the whole shape of the area, because if we want
    # to restrict just to the shape we can then intersect the grid with its
    # shape. Hence the x,ymax+cell_size in the arange.
    x_grid = np.arange(x_min, x_max+cell_size, cell_size)
    y_grid = np.arange(y_min, y_max+cell_size, cell_size)
    Nx = len(x_grid)
    Ny = len(y_grid)
    # We want (x1 x1 ...(Ny times) x1 x2 x2 ...  xNx) and (y1 y2 ... yNy ...(Nx
    # times)), so we use repeat and tile:
    x_grid_re = np.repeat(x_grid, Ny)
    y_grid_re = np.tile(y_grid, Nx)
    # So then (x_grid_re[i], y_grid_re[i]) for all i are all the edges in the
    # grid
    cells_list = []
    for i in range(Nx-1):
        left_x = x_grid_re[i*Ny]
        right_x = x_grid_re[(i+1)*Ny]
        for j in range(Ny-1):
            bot_y = y_grid_re[j]
            top_y = y_grid_re[j+1]
            sub_cells_list = [(left_x, right_x, bot_y, top_y)]
            if places_geodf is not None:
                nr_contained_places = places_geodf.loc[
                    (places_geodf['x_c'] < right_x)
                    & (places_geodf['x_c'] > left_x)
                    & (places_geodf['y_c'] < top_y)
                    & (places_geodf['y_c'] > bot_y)].shape[0]
                if nr_contained_places > max_nr_places:
                    sub_cells_list = []
                    gen_quadrants(left_x, right_x, bot_y, top_y, places_geodf,
                                  sub_cells_list, max_nr_places=max_nr_places,
                                  min_size=cell_size/10)

            for (sub_left_x, sub_right_x, sub_bot_y, sub_top_y) in sub_cells_list:
                # The Polygon closes itself, so no need to repeat the first
                # point at the end.
                cells_list.append(Polygon([
                    (sub_left_x, sub_top_y), (sub_right_x, sub_top_y),
                    (sub_right_x, sub_bot_y), (sub_left_x, sub_bot_y)]))

    cells_df = geopd.GeoDataFrame(cells_list, crs=xy_proj, columns=['geometry'])
    # Prefix `cc` to the index to keep a unique index when mixing data from
    # several regions.
    cells_df.index = shape_df.cc + '.' + cells_df.index.astype(str)
    cells_df['cell_id'] = cells_df.index
    if intersect:
        # cells_in_shape_df = geopd.overlay(
        #     cells_df, shape_df[['geometry']], how='intersection')
        cells_in_shape_df = geopd.clip(cells_df, shape_df)
        cells_in_shape_df.index = cells_in_shape_df['cell_id']
        cells_in_shape_df.cell_size = cell_size
    else:
        cells_in_shape_df = None

    cells_df.cell_size = cell_size
    return cells_df, cells_in_shape_df, Nx-1, Ny-1


def extract_shape(raw_shape_df, cc, bbox=None, latlon_proj='epsg:4326',
                  min_area=None, simplify_tol=None, xy_proj='epsg:3857'):
    '''
    Extracts the shape of the area of interest. If bbox is provided, in the
    format [min_lon, min_lat, max_lon, max_lat], only keep the intersection of
    the shape with this bounding box. Then the shape we extract is simplified to
    accelerate later computations, first by removing irrelevant polygons inside
    the shape (if it's comprised of more than one), and then simplifying the
    contours.
    '''
    shape_df = raw_shape_df.copy()
    if bbox:
        bbox_geodf = geopd.GeoDataFrame(geometry=[box(*bbox)], crs=latlon_proj)
        shape_df = geopd.clip(shape_df, bbox_geodf)
    shape_df = shape_df.to_crs(xy_proj)
    shapely_geo = shape_df.geometry.iloc[0]
    if min_area is None or simplify_tol is None:
        area_bounds = shapely_geo.bounds
        # Get an upper limit of the distance that can be travelled inside the
        # area
        max_distance = np.sqrt((area_bounds[0]-area_bounds[2])**2
                               + (area_bounds[1]-area_bounds[3])**2)
        if simplify_tol is None:
            simplify_tol = max_distance / 1000

    if isinstance(shapely_geo, MultiPolygon):
        if min_area is None:
            min_area = max_distance**2 / 1000
        # We delete the polygons in the multipolygon which are too small and
        # just complicate the shape needlessly.
        shape_df.geometry.iloc[0] = MultiPolygon([poly for poly in shapely_geo
                                                  if poly.area > min_area])
    # We also simplify by a given tolerance (max distance a point can be moved),
    # this could be a parameter in countries.json if needed
    shape_df.geometry = shape_df.simplify(simplify_tol)
    shape_df.cc = cc
    return shape_df


def geo_from_bbox(bbox):
    '''
    From a bounding box dictionary `bbox`, which is of the form
    {'coordinates': [[[x,y] at top right, [x,y] at top left, ...]]}, returns the
    geometry and its area. If the four coordinates are actually the same, they
    define a null-area Polygon, or rather something better defined as a Point.
    '''
    geo = Polygon(bbox)
    area = geo.area
    if area == 0:
        geo = Point(bbox[0])
    return geo, area


def make_places_geodf(raw_places_df, shape_df, latlon_proj='epsg:4326',
                      xy_proj='epsg:3857'):
    '''
    Constructs a GeoDataFrame with all the places in `raw_places_df` which have
    their centroid within `shape_df`, and calculates their area within the shape
    in squared meters.
    '''
    # Very few places have a null bounding box, 2 of them in Canada for
    # instance, so we get rid of those.
    places_df = (raw_places_df.rename(columns={'bounding_box': 'bbox'})
                              .dropna(subset=['bbox'])
                              .copy())
    places_df['bbox'] = places_df['bbox'].apply(
        lambda bbox: bbox['coordinates'][0])
    shape_in_latlon = shape_df[['geometry']].to_crs(latlon_proj)
    shape_min_lon, shape_min_lat, shape_max_lon, shape_max_lat = (
        shape_in_latlon['geometry'].iloc[0].bounds)
    # We will first filter out places which are outside of or bigger than
    # the shape we're considering. This may seem redundant with what
    # follows, but this part is much faster to run, and allows the following
    # operations to run much faster, as the size of `places_df` can be
    # greatly reduced (for instance when considering a state of the US,
    # which has more than a million entries in `places_df`).
    places_df['min_lon'] = places_df['bbox'].apply(lambda x: x[0][0])
    places_df['min_lat'] = places_df['bbox'].apply(lambda x: x[0][1])
    places_df['max_lon'] = places_df['bbox'].apply(lambda x: x[2][0])
    places_df['max_lat'] = places_df['bbox'].apply(lambda x: x[2][1])
    is_bot_left_in_shape = ((places_df['min_lon'] <= shape_max_lon)
                            & (places_df['min_lon'] >= shape_min_lon)
                            & (places_df['min_lat'] <= shape_max_lat)
                            & (places_df['min_lat'] >= shape_min_lat))
    is_top_right_in_shape = ((places_df['max_lon'] <= shape_max_lon)
                             & (places_df['max_lon'] >= shape_min_lon)
                             & (places_df['max_lat'] <= shape_max_lat)
                             & (places_df['max_lat'] >= shape_min_lat))
    # Add this mask to keep country when there are no other used bbox places,
    # for small countries where we keep a single cell we need to keep it, and
    # for larger countries it will be discarded with the `max_place_area` filter
    # anyway.
    is_country = places_df['place_type'] == 'country'
    shape_mask = is_bot_left_in_shape | is_top_right_in_shape | is_country
    places_df = places_df.loc[shape_mask]
    # The area obtained here is in degree**2 and thus doesn't mean much, we just
    # get it at this point to differentiate points from polygons.
    places_df['geometry'], places_df['area'] = [
        pd.Series(x, index=places_df.index)
        for x in zip(*places_df['bbox'].apply(geo_from_bbox))]
    places_geodf = geopd.GeoDataFrame(
        places_df, crs=latlon_proj, geometry=places_df['geometry'])
    places_geodf = places_geodf.set_index('id', drop=False)
    # We then get the places' centroids to check that they are within our area
    # (useful when we're only interested in the region of a country).
    places_centroids = geopd.GeoDataFrame(
        geometry=places_geodf[['geometry']].centroid, crs=latlon_proj)
    places_in_shape = geopd.sjoin(places_centroids, shape_in_latlon,
                                  op='within', rsuffix='shape')
    places_geodf = places_geodf.join(places_in_shape['index_shape'],
                                     how='inner')
    places_geodf = places_geodf.to_crs(xy_proj)
    # Since the places' bbox can stretch outside of the whole shape, we need to
    # take the intersection between the two. However, we only use the overlay to
    # calculate the area, so that places_to_cells distributes the whole
    # population of a place to the different cells within it. However we don't
    # need the actual geometry from the intersection, which is more complex and
    # thus slows down computations later on.
    poly_mask = places_geodf['area'] > 0
    if places_geodf.loc[poly_mask].shape[0] > 0:
        polygons_in_shape = geopd.overlay(
            shape_df[['geometry']], places_geodf.loc[poly_mask],
            how='intersection')
        polygons_in_shape = polygons_in_shape.set_index('id')
        places_geodf.loc[poly_mask, 'area'] = polygons_in_shape.area
    places_geodf = places_geodf.drop(
        columns=['bbox', 'id', 'index_shape'])
    places_geodf.index.name = 'place_id'
    places_geodf.cc = shape_df.cc
    return places_geodf


def d_matrix_from_cells(cell_plot_df):
    '''
    Generates a matrix of distances between pairs of cell centroids, for every
    combination of cells found in `cell_plot_df`.
    '''
    n_cells = cell_plot_df.shape[0]
    d_matrix = np.zeros((n_cells, n_cells))
    cells_centroids = cell_plot_df.geometry.centroid.values
    for i in range(n_cells):
        d_matrix[i, i:] = cells_centroids[i:].distance(cells_centroids[i])
        d_matrix[i:, i] = d_matrix[i, i:]
    return d_matrix


def calc_shape_dims(shape_df, latlon_proj='epsg:4326'):
    '''
    Calculate the width and height in meters of the bounding box of the shapes
    contained within `shape_df`.
    '''
    min_lon, min_lat, max_lon, max_lat = (
        shape_df.geometry.to_crs(latlon_proj).total_bounds)
    # For a given longitude extent, the width is maximum the closer to the
    # equator, so the closer the latitude is to 0.
    eq_crossed = min_lat * max_lat < 0
    lat_max_width = min(abs(min_lat), abs(max_lat)) * (1 - int(eq_crossed))
    width = haversine(min_lon, lat_max_width, max_lon, lat_max_width)
    height = haversine(min_lon, min_lat, min_lon, max_lat)
    return width, height
