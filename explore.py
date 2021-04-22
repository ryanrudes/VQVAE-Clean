from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from tqdm import tqdm

env = Qbert()
goexplore = GoExplore(env, return_via = 'trajectory')

goexplore.initialize()

while True:
    checkpoint_reached = goexplore.run(render = True) # goexplore.iterations % 100 == 0)
    if checkpoint_reached:
        # print (goexplore.report())
        print (goexplore.status())
