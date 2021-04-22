from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from tqdm import tqdm

env = Qbert()
goexplore = GoExplore(env)

goexplore.initialize()

for iteration in range(1000):
    goexplore.run(render = True)
    print (goexplore.report())
