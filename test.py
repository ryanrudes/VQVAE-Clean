from experiment import Experiment
from goexplore.wrappers import *
from goexplore.utils import *
from rich.progress import *
import os

def renderfn(iterations):
    return False

EXPERIMENTS = 20
DURATION = 10000
UNITS = 'frames'

RENDERFN = renderfn
CELLFN = cellfn
HASHFN = hashfn

REPEAT = 0.95
NSTEPS = 100

VERBOSE = 0
SEED = 42

METHOD = 'ram'

address = input('Enter your email address: ')

def callback(goexplore, experiment, root):
    path = os.path.join(root, 'experiments')
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, str(experiment))
    goexplore.save(path)
    return {'cells': goexplore.archivesize()}

record = ['highscore', 'frames', 'iterations']

env = Qbert()

experiment = Experiment(env,
                        units    = UNITS,
                        cellfn   = CELLFN,
                        hashfn   = HASHFN,
                        repeat   = REPEAT,
                        nsteps   = NSTEPS,
                        seed     = SEED,
                        method   = METHOD,
                        verbose  = VERBOSE,
                        renderfn = RENDERFN)

experiment.run(DURATION, EXPERIMENTS,
               sendmail = True,
               address  = address,
               record   = record,
               callback = callback)
