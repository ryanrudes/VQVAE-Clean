import os

import torch
from torch import distributed as dist
from torch import multiprocessing as mp

import distributed as dist_fn

# This is an original copy of licensed source code that has
# been released into the public domain by its author(s)
#
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    return port


def launch(fn, n_gpu_per_machine, n_machine=1, machine_rank=0, dist_url=None, args=()):
    world_size = n_machine * n_gpu_per_machine

    if world_size > 1:
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"

        if dist_url == "auto":
            if n_machine != 1:
                raise ValueError('dist_url="auto" not supported in multi-machine jobs')

            port = find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        if n_machine > 1 and dist_url.startswith("file://"):
            raise ValueError(
                "file:// is not a reliable init method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            distributed_worker,
            nprocs=n_gpu_per_machine,
            args=(fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args),
            daemon=False,
        )

    else:
        fn(*args)


def distributed_worker(
    local_rank, fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args
):
    if not torch.cuda.is_available():
        raise OSError("CUDA is not available. Please check your environments")

    global_rank = machine_rank * n_gpu_per_machine + local_rank

    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )

    except Exception:
        raise OSError("failed to initialize NCCL groups")

    dist_fn.synchronize()

    if n_gpu_per_machine > torch.cuda.device_count():
        raise ValueError(
            f"specified n_gpu_per_machine larger than available device ({torch.cuda.device_count()})"
        )

    torch.cuda.set_device(local_rank)

    if dist_fn.LOCAL_PROCESS_GROUP is not None:
        raise ValueError("torch.distributed.LOCAL_PROCESS_GROUP is not None")

    n_machine = world_size // n_gpu_per_machine

    for i in range(n_machine):
        ranks_on_i = list(range(i * n_gpu_per_machine, (i + 1) * n_gpu_per_machine))
        pg = dist.new_group(ranks_on_i)

        if i == machine_rank:
            dist_fn.distributed.LOCAL_PROCESS_GROUP = pg

    fn(*args)
