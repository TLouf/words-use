from itertools import chain
import re
import subprocess

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
    for file_path in chain(oslom_res_path.glob('tp'),
                           oslom_res_path.glob('tp[1-9]')):
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
