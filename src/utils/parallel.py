import numpy as np
import ray

def split_task(fun, num_cpus, list_iter, *fun_args, **fun_kwargs):
    '''
    Split the task of applying `fun` on every element of `list_iter` in
    `num_cpus` bunches and run the task in parallel using this many CPUs.
    '''
    bounds = np.linspace(0, len(list_iter), num_cpus+1).astype(int)
    sub_list_iter = [list_iter[bounds[i]:bounds[i+1]] for i in range(num_cpus)]

    obj_refs = [remote_fun.remote(fun, sub_list, *fun_args, **fun_kwargs)
                for sub_list in sub_list_iter]
    return obj_refs


def fast_combine(combine_fun, list_elems):
    '''
    Parallelize a function to combine with `combine_fun` elements of
    `raw_list_elems` in an optimal way. First combine base elements two by two,
    then the results of these combinations two by two, etc recursively until
    there's nothing left to combine.
    '''
    while len(list_elems) > 1:
        list_elems = list_elems[2:] + [combine_remote.remote(combine_fun,
                                                             list_elems[0],
                                                             list_elems[1])]

    return list_elems


@ray.remote
def combine_remote(combine_fun, x, y):
    return combine_fun(x, y)


@ray.remote
def remote_fun(fun, sub_list, *fun_args, **fun_kwargs):
    return fun(sub_list, *fun_args, **fun_kwargs)
