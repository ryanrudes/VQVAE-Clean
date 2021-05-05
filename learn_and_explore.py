from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from goexplore.utils import *
from collections import deque
from vqvae import VQVAE
from rich import print
from time import time
import random
import os

from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from torch import optim
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def encode_fn():
    model = VQVAE().to(device)

    def encode(x):
        model.eval()

        with torch.no_grad():
            x = cv2.resize(x, (160, 160), interpolation = cv2.INTER_AREA)
            x = transform(x)
            x = x.unsqueeze(0)
            x = x.to(device)

            _, _, _, id_t, id_b = model.encode(x)

        id_t = id_t.cpu().numpy()
        # id_b = id_b.cpu().numpy()

        model.train()

        return id_t

    return model, encode

# The below method is not used unless specified in goexplore.initialize
# To use this method, also ensure that encode() returns both `id_t` and `id_b`
def hashfn(x):
    """Hash function for cells based on both encoder reps"""
    return hash(x[0].data.tobytes() + x[1].data.tobytes())

def data_stream():
    global updates

    replay = deque(maxlen = 20000)

    while True:
        observations = goexplore.run(return_states = True)

        for x in observations:
            x = cv2.resize(x, (160, 160), interpolation = cv2.INTER_AREA)
            x = transform(x)
            replay.append(x)

        """
        Determine `repeat` somehow. I've tried various approaches to balancing
        training and exploration, but haven't found one that seems significantly
        superior to the others yet. Some I've tried are:
         - Computing `repeat` based upon the encoder perplexity
         - Computing `repeat` based on variance across observations
         - Computing `repeat` through an asymptotic function
        """

        for i in range(int(repeat)):
            for x in random.choices(replay, k = BATCH_SIZE):
                x = x.to(device)
                yield x, 0

            updates += 1

class ObservationDataset(IterableDataset):
    def __init__(self):
        self.stream = data_stream()

    def __iter__(self):
        return self.stream

updates = 0
BATCH_SIZE = 128

env = MontezumaRevenge()
goexplore = GoExplore(env)
model, cellfn = encode_fn()
goexplore.initialize(cellfn = cellfn, mode = 'ram', saveobs = True)

dataset = ObservationDataset()
loader = DataLoader(dataset, batch_size = BATCH_SIZE)

optimizer = optim.Adam(model.parameters(), lr = 3e-4)
scheduler = None

start = str(time())
run_path = os.path.join('runs', start)
sample_path = os.path.join(run_path, 'sample')
checkpoint_path = os.path.join(run_path, 'checkpoint')
os.mkdir(run_path)
os.mkdir(sample_path)
os.mkdir(checkpoint_path)

start = time()
for i, (recon_loss, latent_loss, avg_mse, lr) in enumerate(model.train_epoch(0, loader, optimizer, scheduler, device, sample_path)):
    if goexplore.iterations % 100 == 0 or updates % 100 == 0:
        now = time()
        duration = now - start
        start = now
        print (f'updates: {updates}; mse: {recon_loss:.5f}; latent: {latent_loss:.5f}; avg mse: {avg_mse:.5f}; lr: {lr:.5f}; duration: {duration:.5f}; perplexity (top): {model.perplexity_t:.5f}; perplexity (bottom): {model.perplexity_b:.5f}')
        print (goexplore.report())
        print ()
        goexplore.refresh()

    if goexplore.iterations % 1000 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'vqvae_%s.pt' % i))
