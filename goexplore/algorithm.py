from .termination import *
from .wrappers import *
from .weights import *
from .config import *
from .powers import *
from .utils import *
from .cell import *
from .tree import *

from collections import defaultdict
import numpy as np

class GoExplore:
    metadata = {'return_via': ['ram', 'trajectory']}

    def __init__(self,
                 env,
                 cellfn=cellfn,
                 hashfn=hashfn,
                 repeat=0.95,
                 nsteps=100,
                 return_via='ram'):
        self.env = env
        self.cellfn = cellfn
        self.hashfn = hashfn
        self.repeat = repeat
        self.nsteps = nsteps
        self.report = lambda: "Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d" % (self.iterations, len(self.record), self.frames, self.highscore)
        self.status = lambda delimiter=' ', separator=True: "Archive: %s, Trajectory: %s" % (prettysize(self.record, delimiter=delimiter, separator=separator), prettysize(self.trajectory, delimiter=delimiter, separator=separator))

        if return_via in self.metadata['return_via']:
            self.return_via = return_via
        else:
            raise ValueError('Expected return method `return_via` to be one of %s, but found %s' % ('/'.join(self.metadata['return_via']), return_via))

    def ram(self):
        return self.env.env.clone_full_state()

    def restore(self, cell):
        ram, reward, length = cell.choose()
        self.reward = reward
        self.length = length
        self.trajectory.set(cell.node)
        self.env.reset()

        if self.return_via == 'ram':
            self.env.env.restore_full_state(ram)
        else:
            trajectory = self.trajectory.get_trajectory()
            while trajectory:
                action = trajectory.pop()
                self.env.step(action)

    def random(self):
        return self.env.action_space.sample()

    def getstate(self):
        return (
            self.ram() if self.return_via == 'ram' else None,
            self.reward,
            self.length,
        )

    def initialize(self):
        observation = self.env.reset()

        cell = self.cellfn(observation)
        code = self.hashfn(cell)

        self.record = defaultdict(Cell)
        self.reward = 0
        self.action = 0
        self.length = 0
        self.frames = 0
        self.highscore = 0
        self.discovered = 0
        self.iterations = 0
        self.trajectory = LinkedTree(code)

        cell = self.record[code]

        cell.node = self.trajectory.node
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

        new_highscore = False
        if self.reward > self.highscore:
            self.highscore = self.reward
            new_highscore = True

        if render:
            self.env.render()

        if terminal:
            return True, new_highscore

        cell = self.cellfn(observation)
        code = self.hashfn(cell)
        cell = self.record[code]

        self.trajectory.act(self.action, code)

        if self.update(cell):
            cell.node = self.trajectory.node
            cell.setstate(self.getstate())
            self.discovered += 1

        return False, new_highscore

    def run(self, render=False):
        self.discovered = 0
        checkpoint_reached = False

        for i in range(self.nsteps):
            terminal, new_highscore = self.act(render)
            checkpoint_reached |= new_highscore
            if terminal:
                break

        if self.discovered:
            self.record[self.restore_code].lead_to_improvement()

        self.iterations += 1

        codes = [*self.record]
        probs = np.array([cell.score for cell in self.record.values()])
        probs = probs / probs.sum()

        restore_code = np.random.choice(codes, p = probs)
        restore_cell = self.record[restore_code]

        self.restore(restore_cell)
        self.restore_code = restore_code

        return checkpoint_reached
