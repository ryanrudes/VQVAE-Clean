from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from vqvae import VQVAE
from rich import print
import torch

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_fn(PATH):
    print ('Loading model...')
    model = VQVAE()
    saved = torch.load(PATH, map_location = torch.device(device))['model']
    model.load_state_dict(saved)
    model.eval()

    def encode(observation):
        x = cv2.resize(observation, (196, 148), interpolation = cv2.INTER_AREA)
        x = x / 255.0
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)
        x = x.unsqueeze(0)
        x = x.to(device)

        _, _, _, indices, _ = model.encode(x)
        encoded = indices.cpu().numpy()[0]

        return encoded

    return encode

print ('Creating environment...')
env = MontezumaRevenge()

print ('Starting algorithm...')
goexplore = GoExplore(env)
goexplore.initialize(method = 'ram', cellfn = encode_fn(PATH))

while goexplore.highscore == 0:
    goexplore.run(render = True)
    print(goexplore.report() + ', ' + goexplore.status())
