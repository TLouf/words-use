from itertools import chain
import re
import subprocess
import numpy as np
import esda

def gen_oslom_res_path(data_path, oslom_opt_params, suffix=''):
    '''
    Generate a path where to save the results of OSLOM, based on the data file
    on which it will be run and with which parameters.
    '''
    parent_path = data_path.parent
    data_filename = data_path.parts[-1]
    data_str = data_filename.split('.')[0]
    params_str = ''.join([s.replace(' ', '=') for s in oslom_opt_params])
    oslom_res_path = parent_path / f"res_{data_str}{params_str}{suffix}"
    return oslom_res_path


def run_oslom(oslom_dir, data_path, res_path, oslom_opt_params=None,
              directional=False, silent=False):
    '''
    Run the compiled OSLOM located in `oslom_dir` on the net<ork data contained
    in `data_path`, saving the results in `res_path`.
    '''
    if oslom_opt_params is None:
        oslom_opt_params = []
    dir_prefix = (not directional) * 'un'
    oslom_exec_path = oslom_dir / f'oslom_{dir_prefix}dir'
    cmd_list = [str(oslom_exec_path), '-f',
                str(data_path), '-o', str(res_path), '-w'] + oslom_opt_params
    run_kwargs = {'check': True}
    if silent:
        run_kwargs['stdout'] = subprocess.DEVNULL
    p = subprocess.run(cmd_list, **run_kwargs)
    return p


def read_oslom_res(oslom_res_path):
    '''
    Read the tp files containing the results of an OSLOM run into a dictionary
    `cluster_dict` containing, for each returned level, a dictionary with the
    cluster labels as keys and the lists of contained counties as values. The
    key -1 is reserved for singletons.
    '''
    cluster_dict = {}
    for file_path in chain(oslom_res_path.glob('tp[1-9]'),
                           oslom_res_path.glob('tp')):
        lvl_match = re.search("tp([1-9]+)", str(file_path))
        if lvl_match is None:
            lvl = 0
        else:
            lvl = int(lvl_match.groups()[0])
        cluster_dict[lvl] = {-1: []}
        print(f'- level {lvl}')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        pattern = "#module ([0-9]+) size: ([0-9]+) bs: (.+)"
        for i in range(len(lines) // 2):
            mod_line = lines[2*i]
            ids_line = lines[2*i+1]
            matches = re.search(pattern, mod_line).groups()
            clust_nr = int(matches[0])
            mod_cnties = ids_line.split(' ')
            if len(mod_cnties[:-1]) > 1:
                print(f'cluster {clust_nr}, size: {matches[1]}, p-value: ',
                      matches[2])
                cluster_dict[lvl][clust_nr] = [int(c) for c in mod_cnties[:-1]]
            # else it's noise, and there's just one county in the list
            else:
                cnt_id = int(mod_cnties[0])
                cluster_dict[lvl][-1].append(cnt_id)
    return cluster_dict


def get_clusters_agg(cutree):
    '''
    From a matrix (n_samples x n_levels), returns a matrix (n_levels x
    max_nr_clusters) giving the assignment of the lowest level's clusters at
    higher levels, thus showing which clusters get aggregated with which at each
    aggregation step.
    '''
    n_lvls = cutree.shape[1]
    levels_x_clust = np.zeros((n_lvls, n_lvls + 1)).astype(int)
    levels_x_clust[-1, :] = np.arange(0, n_lvls + 1)
    for i in range(n_lvls-2, -1, -1):
        lvl_clusts = cutree[:, i]
        lower_lvl_clusts = cutree[:, i+1]
        # For every cluster in the lower level,
        for clust in np.unique(lower_lvl_clusts):
            # we select the higher level cluster to which it belongs. Because we
            # started from the less aggregated level, all members of that
            # cluster will belong to the same cluster in the more aggregated
            # level, so we take the higher level cluster of the first one.
            agg_lvl = lvl_clusts[lower_lvl_clusts == clust][0]
            levels_x_clust[i, :][levels_x_clust[i+1, :] == clust] = agg_lvl
    levels_x_clust += 1
    return levels_x_clust


def chunk_moran(list_i, word_vectors, contiguity):
    moran_dict = {'I': [], 'z_value': [], 'p_value': []}
    for i in list_i:
        y = word_vectors[:, i]
        mi = esda.moran.Moran(y, contiguity)
        moran_dict['I'].append(mi.I)
        moran_dict['z_value'].append(mi.z_sim)
        moran_dict['p_value'].append(mi.p_sim)
    return moran_dict
