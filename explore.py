from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from tqdm import tqdm

iterations = 1000

env = Qbert()
goexplore = GoExplore(env, method = 'trajectory', repeat = 0.9)

goexplore.initialize()
goexplore.run_for(iterations)
