from goexplore.algorithm import GoExplore
from goexplore.wrappers import *
from goexplore.utils import *
import cv2

iterations = 1000000

env = MontezumaRevenge()
goexplore = GoExplore(env)

width = 11
height = 8
interpolation = cv2.INTER_AREA
grayscale = False
intensities = 8

cellfn = makecellfn(width, height, interpolation, grayscale, intensities)
goexplore.initialize(method = 'trajectory', cellfn = cellfn)
goexplore.run_for(iterations)
