from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from goexplore.utils import *

from multiprocessing import Pool
from rich import print
import cv2

iterations = 100000

env = MontezumaRevenge()
goexplore = GoExplore(env)

width = 11
height = 8
interpolation = cv2.INTER_AREA
grayscale = True
intensities = 8

cellfn = makecellfn(width, height, interpolation, grayscale, intensities)
goexplore.initialize(method = 'ram', cellfn = cellfn)

while goexplore.highscore == 0:
    goexplore.run(render = True)
    print(goexplore.report() + ', ' + goexplore.status())
