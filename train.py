import distributed as dist
from vqvae import VQVAE
from torch import optim
import sys
import os

class Args:
    n_gpu = 1
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    dist_url = f'tcp://127.0.0.1:{port}'
    size = 256
    epoch = 560
    lr = 3e-4
    batch_size = 128
    num_workers = 4
    normalize = True
    optimizer = optim.Adam
    sched = ''
    path = ''

args = Args()
model = VQVAE()

dist.launch(model.train, args.n_gpu, 1, 0, args.dist_url, args = (args,))
