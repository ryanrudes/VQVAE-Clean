from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from tqdm import tqdm

iterations = 1000000

env = Pong()
goexplore = GoExplore(env)

goexplore.initialize(method = 'trajectory')
goexplore.run_for(iterations)
