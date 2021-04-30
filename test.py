from experiment import Experiment
from goexplore.wrappers import *
from goexplore.utils import *
from rich.progress import *
import json
import os

def renderfn(iterations):
    return False

EXPERIMENTS = 20
DURATION = 300000
UNITS = 'frames'

RENDERFN = renderfn
CELLFN = cellfn
HASHFN = hashfn

REPEAT = 0.95
NSTEPS = 100

VERBOSE = 0
SEED = 42

METHOD = 'ram'

def callback(goexplore, experiment, root):
    path = os.path.join(root, 'experiments')
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, str(experiment))
    goexplore.save(path)
    return {'cells': goexplore.archivesize()}

record = ['highscore', 'frames', 'iterations']

for name, env in name2env.items():
    env = env()

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
                   # sendmail = False,
                   # address  = address,
                   # password = password,
                   showinfo = False,
                   record   = record,
                   callback = callback,
                   title    = f'{name} downscaled,n={EXPERIMENTS},t={DURATION}')
