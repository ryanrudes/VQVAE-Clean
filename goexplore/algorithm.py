from .termination import *
from .exceptions import *
from .wrappers import *
from .archive import *
from .weights import *
from .config import *
from .powers import *
from .utils import *
from .cell import *
from .tree import *

from rich.traceback import install
from rich.console import Console
from rich.progress import *
install()

from collections import defaultdict
from sys import getsizeof
from time import sleep
import numpy as np
import inspect
import marshal
import tarfile
import shutil
import types
import json
import gzip
import sys
import os
import io

class GoExplore:
    metadata = {'method': ['ram', 'trajectory']}

    def __init__(self, env, hashseed=42):
        self.env = env
        self.report = lambda: 'Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d' % (self.iterations, len(self.record), self.frames, self.highscore)
        self.status = lambda delimiter=' ', separator=True: 'Archive: %s, Trajectory: %s' % (prettysize(self.record, delimiter=delimiter, separator=separator, sizefn=getsizeof), prettysize(self.trajectory, delimiter=delimiter, separator=separator))
        self.setseed(hashseed)

    @property
    def __dict__(self):
        return {
            'env': self.env.env_id,
            'seed': self.seed,
            'method': self.method,
            'repeat': self.repeat,
            'nsteps': self.nsteps,
            'highscore': self.highscore,
            'frames': self.frames,
            'iterations': self.iterations,
            'cellfn': inspect.getsource(self.cellfn),
            'hashfn': inspect.getsource(self.hashfn),
        }

    def setseed(self, seed):
        self.hashseed = seed
        if not os.environ.get('PYTHONHASHSEED'):
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.execv(sys.executable, ['python3'] + sys.argv)

    def archivesize(self):
        return len(self.record)

    def log(self, verbose, console=Console()):
        if verbose == 1:
            console.print(self.report())
        elif verbose == 2:
            console.print(self.report() + ', ' + self.status(delimeter, separator))

    def ram(self):
        return self.env.env.clone_full_state()

    def restore(self, cell):
        ram, reward, length = cell.choose()
        self.reward = reward
        self.length = length
        self.trajectory.set(cell.node)
        self.env.reset()

        if self.method == 'ram':
            self.env.env.restore_full_state(ram)
        else:
            trajectory = self.trajectory.get_trajectory()
            while trajectory:
                action = trajectory.pop()
                observation, reward, terminal, info = self.env.step(action)
                yield observation

    def random(self):
        return self.env.action_space.sample()

    def getstate(self):
        return (
            self.ram() if self.method == 'ram' else None,
            self.reward,
            self.length,
        )

    def initialize(self,
                   cellfn=cellfn,
                   hashfn=hashfn,
                   repeat=0.95,
                   nsteps=100,
                   seed=42,
                   method='ram',
                   **kwargs):
        self.cellfn = cellfn
        self.hashfn = hashfn
        self.repeat = repeat
        self.nsteps = nsteps
        self.method = method
        self.seed = seed

        ensure_type(repeat, float, 'repeat', 'action repeat probability')
        ensure_range(repeat, float, 'repeat', 'action repeat probability', 0, 1)

        ensure_type(nsteps, int, 'nsteps', 'max explore duration')
        ensure_range(nsteps, int, 'nsteps', 'max explore duration', minn=1)
        ensure_from(method, self.metadata['method'], 'method', 'return method')

        self.env.seed(seed)
        observation = self.env.reset()

        cell = self.cellfn(observation)
        code = self.hashfn(cell)

        self.record = Archive()
        self.reward = 0
        self.action = 0
        self.length = 0
        self.frames = 0
        self.highscore = 0
        self.discovered = 0
        self.iterations = 0
        self.trajectory = LinkedTree()

        cell = self.record[code]

        cell.node = self.trajectory.node
        self.trajectory.node.assign(code)
        cell.visit()
        cell.setstate(self.getstate())
        self.restore_code = code

    def update(self, cell):
        new = cell.visit()
        return new or self.reward > cell.reward or self.reward == cell.reward and self.length < cell.length

    def act(self, render=False):
        if np.random.random() > self.repeat:
            self.action = self.random()

        observation, reward, terminal, info = self.env.step(self.action)
        self.reward += reward
        self.length += 1
        self.frames += 1
        self.highscore = max(self.highscore, self.reward)

        if render:
            self.env.render()

        if terminal:
            return observation, reward, terminal, info

        cell = self.cellfn(observation)
        code = self.hashfn(cell)
        cell = self.record[code]

        self.trajectory.act(self.action)

        if self.update(cell):
            if hasattr(cell, 'code'):
                cell.node.remove()

            cell.node = self.trajectory.node
            self.trajectory.node.assign(code)

            cell.setstate(self.getstate())
            self.discovered += 1

        return observation, reward, terminal, info

    def run(self,
            render=False,
            debug=False,
            delay=0.01,
            return_states=False,
            max_frames=np.inf):
        self.discovered = 0

        if return_states:
            observations = []

        for i in range(self.nsteps):
            observation, reward, terminal, info = self.act(render)

            if return_states:
                observations.append(observation)

            if terminal or self.frames == max_frames:
                break

            if debug:
                sleep(delay)

        if self.discovered:
            self.record[self.restore_code].led_to_improvement()

        self.iterations += 1

        codes = [*self.record]
        probs = np.array([cell.score for cell in self.record.values()])
        probs = probs / probs.sum()

        restore_code = np.random.choice(codes, p = probs)
        restore_cell = self.record[restore_code]

        traj = list(self.restore(restore_cell))
        self.restore_code = restore_code

        if return_states:
            return observations + traj

    def run_for(self,
                duration,
                verbose=1,
                units='iterations',
                renderfn=lambda iteration: False,
                delimeter=' ',
                separator=True,
                debug=False,
                delay=0.01,
                desc='Running',
                **kwargs):
        ensure_type(verbose, int, 'verbose', 'logging verbosity')
        ensure_range(verbose, int, 'verbose', 'logging verbosity', 0, 2)
        ensure_from(units, ['iterations', 'frames'], 'units', 'units of duration')

        progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "{task.completed} of {task.total} %s" % units
        )

        if units == 'iterations':
            max_frames = np.inf
        else:
            max_frames = duration

        with progress:
            task = progress.add_task(desc, total = duration)
            iteration = 0

            while not progress.finished:
                render = renderfn(iteration)
                self.run(render, debug=debug, delay=delay, max_frames=max_frames)

                if units == 'iterations':
                    progress.advance(task)
                else:
                    progress.update(task, completed = self.frames, refresh = True)

                self.log(verbose, progress.console)
                iteration += 1

    def save(self, path):
        try:
            # Make the save directory if it does not already exist
            if not os.path.exists(path):
                os.mkdir(path)

            # Write PYTHONHASHSEED to file
            with open(os.path.join(path, '.PYTHONHASHSEED'), 'w') as f:
                f.write(str(self.hashseed))

            states = os.path.join(path, 'ram')
            trajectories = os.path.join(path, 'trajectory')

            # Make directories
            os.mkdir(states)
            os.mkdir(trajectories)

            # Save emulator ram states and action trajectories to compressed .npy.gz archives
            # Collect archive data
            data = {}
            for code, cell in self.record.items():
                with gzip.GzipFile(os.path.join(states, f'{code}.npy.gz'), 'w') as f:
                    np.save(f, cell.ram)

                with gzip.GzipFile(os.path.join(trajectories, f'{code}.npy.gz'), 'w') as f:
                    np.save(f, np.array(self.trajectory.get_trajectory(cell.node)))
                data[code] = cell.save()

            # Save archive data
            with open(os.path.join(path, 'exploration.json'), 'w') as f:
                json.dump(data, f)

            # Save vars
            with open(os.path.join(path, 'vars.json'), 'w') as f:
                json.dump(vars(self), f)

            # Make .tar.gz archives for directories
            shutil.make_archive(os.path.join(path, 'ram'), 'gztar', states)
            shutil.make_archive(os.path.join(path, 'trajectory'), 'gztar', trajectories)

            # Remove directories
            shutil.rmtree(states)
            shutil.rmtree(trajectories)

            return True
        except:
            return False

    def load(self, path, replace=True):
        if replace:
            self.record = Archive()
            self.trajectory = LinkedTree()

        try:
            # Load archive data
            with open(os.path.join(path, 'exploration.json'), 'r') as f:
                archive = json.load(f)

            # Load vars
            with open(os.path.join(path, 'vars.json'), 'r') as f:
                meta = json.load(f)

                self.cellfn = types.FunctionType(marshal.loads(meta['cellfn']), globals(), 'cellfn')
                self.hashfn = types.FunctionType(marshal.loads(meta['hashfn']), globals(), 'hashfn')

                self.repeat = meta['repeat']
                self.nsteps = meta['nsteps']
                self.frames = meta['frames']
                self.method = meta['method']

                self.iterations = meta['iterations']
                self.highscore = meta['highscore']

                if self.env.env_id != meta['env']:
                    self.close()
                    self.env = name2env[meta['env']]

                self.env.seed(meta['seed'])
                self.seed = meta['seed']

            # Update archive with file data
            for code, info in archive.items():
                self.record[int(code)].load(info)

            # Update PYTHONHASHSEED
            with open(os.path.join(path, '.PYTHONHASHSEED'), 'r') as f:
                self.setseed(int(f.read()))

            # Load emulator ram states to their corresponding cells
            with tarfile.open(os.path.join(path, 'ram.tar.gz'), 'r:gz') as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is not None:
                        code = int(member.name[2:-7])
                        data = gzip.decompress(f.read())
                        stream = io.BytesIO(data)
                        ram = np.load(stream)
                        self.record[code].ram = ram

            # Build trajectory tree
            with tarfile.open(os.path.join(path, 'trajectory.tar.gz'), 'r:gz') as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is not None:
                        code = int(member.name[2:-7])
                        data = gzip.decompress(f.read())
                        stream = io.BytesIO(data)
                        trajectory = np.load(stream)
                        self.record[code].length = len(trajectory)
                        self.record[code].node = self.trajectory.add(trajectory, code)

            return True
        except:
            return False

    def close(self):
        self.env.close()
