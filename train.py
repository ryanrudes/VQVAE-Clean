import distributed as dist
from vqvae import VQVAE

def train(args):
    model = VQVAE()
    dist.launch(model.run, args.n_gpu, 1, 0, args.dist_url, args = (args,))
