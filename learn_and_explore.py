from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from goexplore.utils import *
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

            _, _, _, encoded, _ = model.encode(x)

        encoded = encoded.cpu().numpy()[0]
        model.train()
        return encoded

    return model, encode

def data_stream():
    global updates
    iteration = 0
    a = 0 # mean_samps
    FRAMES = 500_000
    UPDATES = 200_000
    repeat_obs = 16
    while True:
        iteration += 1
        observations = goexplore.run(return_states = True, return_traj = True)

        a += (len(observations) - a) / iteration # mean_samps
        n = FRAMES / (goexplore.frames / iteration) # total_iterations
        m = (2 * UPDATES - 2 * a * n) / (n ** 2 - n)
        y = int(a - m * iteration)
        updates += y

        for i in range(y):
            random.shuffle(observations)
            for observation in observations:
                x = cv2.resize(observation, (160, 160), interpolation = cv2.INTER_AREA)
                x = transform(x)
                x = x.to(device)
                yield x, 0

        if iteration % 10 == 0:
            goexplore.refresh()

class ObservationDataset(IterableDataset):
    def __init__(self):
        self.stream = data_stream()

    def __iter__(self):
        return self.stream

updates = 0

env = Qbert()
goexplore = GoExplore(env)
model, cellfn = encode_fn()
goexplore.initialize(cellfn = cellfn, mode = 'trajectory')

dataset = ObservationDataset()
loader = DataLoader(dataset, batch_size = 128)

optimizer = optim.Adam(model.parameters(), lr = 3e-4)
scheduler = None

start = str(time())
run_path = os.path.join('runs', start)
sample_path = os.path.join(run_path, 'sample')
checkpoint_path = os.path.join(run_path, 'checkpoint')
os.mkdir(run_path)
os.mkdir(sample_path)
os.mkdir(checkpoint_path)

for i, (recon_loss, latent_loss, avg_mse, lr) in enumerate(model.train_epoch(0, loader, optimizer, scheduler, device, sample_path)):
    print (f'updates: {updates}; mse: {recon_loss:.5f}; latent: {latent_loss:.5f}; avg mse: {avg_mse:.5f}; lr: {lr:.5f}')
    print (goexplore.report())
    if i % 1000 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'vqvae_%s.pt' % i))

"""
200,000 batches over 500,000 steps
"""
