import numpy as np
import ray

def split_task(fun, num_cpus, list_iter, *fun_args, **fun_kwargs):
    '''
    Split the task of applying `fun` on every element of `list_iter` in
    `num_cpus` bunches and run the task in parallel using this many CPUs.
    '''
    bounds = np.linspace(0, len(list_iter), num_cpus+1).astype(int)
    sub_list_iter = [list_iter[bounds[i]:bounds[i+1]] for i in range(num_cpus)]

    @ray.remote
    def remote_fun(sub_list, *fun_args, **fun_kwargs):
        return fun(sub_list, *fun_args, **fun_kwargs)

    obj_refs = [remote_fun.remote(sub_list, *fun_args, **fun_kwargs)
                for sub_list in sub_list_iter]
    res = ray.get(obj_refs)
    return res
