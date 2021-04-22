from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from goexplore.utils import *
import cv2

iterations = 1000

env = Qbert()
goexplore = GoExplore(env)

interpolation = cv2.INTER_AREA
grayscale = True
intensities = 8

for width in range(1, 12):
    for height in range(1, 12):
        cellfn = makecellfn(width, height, interpolation, grayscale, intensities)
        goexplore.initialize(method = 'ram', cellfn = cellfn)
        goexplore.run_for(iterations, verbose = 2)
