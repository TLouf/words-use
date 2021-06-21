import os
from pathlib import Path
import subprocess
import IPython.display
import numpy as np
import plotly.graph_objects as go
import plotly.colors
import plotly.offline
from shapely.geometry import MultiPoint
import src.visualization.maps as map_viz

def plot_interactive(fig, mapbox_style='stamen-toner', mapbox_zoom=10,
                     plotly_renderer='iframe_connected',
                     save_path=None, show=False):
    '''
    Utility to plot an interactive map, given the plot data and layout given in
    the plotly Figure instance `fig`.
    '''
    fig.update_layout(
        mapbox_style=mapbox_style, mapbox_zoom=mapbox_zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0})
    use_iframe_renderer = plotly_renderer.startswith('iframe')
    if save_path:
        # Path objects are not yet supported by plotly, so first cast to str.
        plotly.offline.plot(fig, filename=str(save_path), auto_open=False)
    if show:
        # With the 'iframe' renderer, a standalone HTML is created in a new
        # folder iframe_figures/, and the files are named according to the cell
        # number. Thus, many files can be created while testing out this
        # function, so to avoid this we simply use the previously saved HTML
        # file (so use_iframe_renderer should imply save_path), which has a
        # name we chose.
        if use_iframe_renderer:
            pwd_path = Path(os.environ['PWD'])
            rel_path = save_path.resolve().relative_to(pwd_path)
            # Print the location of the file to have a clickable hyperlink to
            # open it in a new tab from a notebook.
            print(get_jpt_address() + 'files/' + str(rel_path))
            IPython.display.display(IPython.display.IFrame(
                src=save_path, width=900, height=600))
        else:
            fig.show(renderer=plotly_renderer, width=900, height=600,
                     config={'modeBarButtonsToAdd': ['zoomInMapbox',
                                                     'zoomOutMapbox']})

    return fig


def cells(cell_plot_df, metric_col,
          colorscale='Plasma', latlon_proj='epsg:4326',
          alpha=0.8, **plot_interactive_kwargs):
    '''
    Plots an interactive Choropleth map with Plotly.
    The map layer on top of which this data is shown is provided by mapbox (see
    https://plot.ly/python/mapbox-layers/#base-maps-in-layoutmapboxstyle for
    possible values of 'mapbox_style').
    Plotly proposes different renderers, described at:
    https://plot.ly/python/renderers/#the-builtin-renderers.
    The geometry column of cell_plot_df must contain only valid geometries:
    just one null value will prevent the choropleth from being plotted.
    '''
    start_point = MultiPoint(
        cell_plot_df['geometry'].to_crs(latlon_proj).centroid.values).centroid
    layout = go.Layout(
        mapbox_center={"lat": start_point.y, "lon": start_point.x})

    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon.
    geo_dict = cell_plot_df.to_crs(latlon_proj).geometry.__geo_interface__
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=cell_plot_df.index.values,
        hoverinfo='skip',
        colorscale=colorscale,
        marker_opacity=alpha,
        marker_line_width=0.1)

    data = [go.Choroplethmapbox(**choropleth_dict,
                                z=cell_plot_df[metric_col],
                                visible=True)]

    fig = go.Figure(data=data, layout=layout)
    fig = plot_interactive(fig, **plot_interactive_kwargs)
    return fig


def clusters(geodf, cluster_data, valid_cnt,
             colorscale=None, latlon_proj='epsg:4326',
             alpha=0.8, **plot_interactive_kwargs):
    '''
    Plots an interactive choropleth map of clusters, with a clickable legend to
    show/hide clusters.
    '''
    cell_plot_df = map_viz.prep_cluster_plot(geodf.loc[valid_cnt], cluster_data)
    geometry = cell_plot_df.to_crs(latlon_proj).geometry
    start_point = MultiPoint(geometry.centroid.values).centroid
    layout = go.Layout(
        mapbox_center={"lat": start_point.y, "lon": start_point.x},
        legend={'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(255, 255, 255, 0.7)',
                'bordercolor': 'rgb(0, 0, 0)', 'borderwidth': 1})

    if np.all(cell_plot_df['label'].str.isnumeric()):
        all_clusters = sorted(cell_plot_df['label'].unique(), key=int)
    else:
        all_clusters = sorted(cell_plot_df['label'].unique())
    data = []
    choropleth_dict = dict(
        showlegend=True,
        showscale=False,
        marker_opacity=alpha,
        marker_line_width=0.1)

    if colorscale is None:
        if 'homeless' in all_clusters:
            # We generate one color less and assign grey to homeless counties.
            colorscale = (map_viz.gen_distinct_colors(len(all_clusters) - 1)
                          + [(0.5, 0.5, 0.5)])
        else:
            colorscale = map_viz.gen_distinct_colors(len(all_clusters))
        colorscale = [f'rgb{c}' for c in  colorscale]

    # Each cluster is plotted in a separate trace.
    for i, cluster in enumerate(all_clusters):
        mask = cell_plot_df['label'] == cluster
        # We extract only the geometries within the cluster to generate separate
        # but non-duplicate GeoJSON data.
        geo_dict = geometry.loc[mask].__geo_interface__
        # We select a single color from the scale, and apply to this trace a
        # colorscale that keeps this color on its whole range. Then we also
        # assign `z` a constant numerical value. The cluster will thus be
        # coloured in this colour both on the map and in the legend.
        c = colorscale[i]
        data.append(go.Choroplethmapbox(
            **choropleth_dict,
            geojson=geo_dict,
            locations=mask.loc[mask].index.values,
            z=[i] * mask.sum(),
            colorscale=((0.0, c), (1.0, c)),
            name=str(cluster)))

    fig = go.Figure(data=data, layout=layout)
    fig = plot_interactive(fig, **plot_interactive_kwargs)
    return fig


def get_jpt_address():
    '''
    Get the currently running Jupyter's server address, or at least the first
    one found, if any.
    '''
    # from jupyter_server import serverapp
    # serverapp.list_running_servers()
    env_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
    serv_list = subprocess.run([env_path + '/bin/jupyter', 'server', 'list'],
                               capture_output=True, check=True)
    all_lines = serv_list.stdout.decode().split('\n')
    for line in all_lines[1:]:
        address, jpt_wd = line.split(' :: ')
        if jpt_wd == os.environ['PWD']:
            return address

    return ''
