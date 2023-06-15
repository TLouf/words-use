import datetime
import os
from pathlib import Path

import numpy as np
import paramiko
import ray
from dotenv import load_dotenv

import src.utils.paths as paths_utils

load_dotenv()


def split_task(fun, list_iter, *fun_args, num_cpus=None, **fun_kwargs):
    """
    Split the task of applying `fun` on every element of `list_iter` in
    `num_cpus` bunches and run the task in parallel using this many CPUs.
    """
    if num_cpus is None:
        num_cpus = int(ray.available_resources()["CPU"])

    bounds = np.linspace(0, len(list_iter), num_cpus + 1).astype(int)
    sub_list_iter = [list_iter[bounds[i] : bounds[i + 1]] for i in range(num_cpus)]

    obj_refs = [
        remote_fun.remote(fun, sub_list, *fun_args, **fun_kwargs)
        for sub_list in sub_list_iter
    ]
    return obj_refs


def fast_combine(combine_fun, list_elems):
    """
    Parallelize a function to combine with `combine_fun` elements of
    `raw_list_elems` in an optimal way. First combine base elements two by two,
    then the results of these combinations two by two, etc recursively until
    there's nothing left to combine.
    """
    while len(list_elems) > 1:
        list_elems = list_elems[2:] + [
            combine_remote.remote(combine_fun, list_elems[0], list_elems[1])
        ]

    return list_elems


@ray.remote
def combine_remote(combine_fun, x, y):
    return combine_fun(x, y)


@ray.remote
def remote_fun(fun, sub_list, *fun_args, **fun_kwargs):
    return fun(sub_list, *fun_args, **fun_kwargs)


def nured_run(
    script_fname,
    args,
    allocated_time,
    waiting_time=0,
    executable=None,
    custom_log_file=True,
    ssh_domain="nuredduna2020",
    username_key="IFISC_USERNAME",
    run_dir="scripts",
    proj_dir=None,
):
    if isinstance(args, str):
        args = [args]

    run_from_dir = Path(proj_dir or os.environ["PROJ_DIR"]) / run_dir

    if executable is None:
        # This assumes we're on the server.
        venv_path = os.environ.get("CONDA_PREFIX", os.environ.get("VIRTUAL_ENV"))
        executable = Path(venv_path) / "bin" / "python"

    sbatch_dir = "/common/slurm/bin"
    base_cmd = f"export PATH=$PATH:{sbatch_dir};cd {run_from_dir};"

    py_file_path = Path(script_fname)
    if custom_log_file:
        logs_dir = run_from_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        time_str = datetime.datetime.now().isoformat(timespec='milliseconds')
        log_file_fmt = f"{py_file_path.stem}_{time_str}{{i}}{{type}}.log"
    else:
        log_str = ""

    whole_cmd = base_cmd
    for i, a in enumerate(args):
        if custom_log_file:
            iter_log_file = logs_dir / log_file_fmt.format(type="_out", i=i)
            iter_err_log_file = logs_dir / log_file_fmt.format(type="_err", i=i)
            log_str = f" -o {iter_log_file} -e {iter_err_log_file}"

        whole_cmd += (
            f"/usr/local/bin/run -t {allocated_time}{log_str}"
            # Because this part below is enclosed in quotes, optional arguments can be
            # used, otherwise they are interpreted as `run`'s optional arguments.
            f' "{executable} {py_file_path} {a}";'
        )

        if waiting_time:
            whole_cmd += f"sleep {waiting_time};"

    print(whole_cmd)

    # Get username fron environment variable in case it was not configured in
    # `~/.ssh/.config`.
    ssh_username = os.environ.get(username_key)
    with paramiko.client.SSHClient() as ssh_client:
        ssh_client.load_system_host_keys()
        ssh_client.connect(ssh_domain, username=ssh_username)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command(whole_cmd)
        print(ssh_stderr.readlines())
        print(ssh_stdout.readlines())
