from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from tqdm import tqdm

env = MontezumaRevenge()
goexplore = GoExplore(env)

goexplore.initialize()

while True:
    goexplore.run(render = goexplore.iterations % 100 == 0)
    print (goexplore.report())
