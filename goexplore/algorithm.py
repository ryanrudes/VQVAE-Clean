from .termination import *
from .wrappers import *
from .weights import *
from .config import *
from .powers import *
from .utils import *
from .cell import *

from collections import defaultdict
import numpy as np

class GoExplore:
    def __init__(self,
                 env,
                 cellfn=cellfn,
                 hashfn=hashfn,
                 repeat=0.95,
                 nsteps=100):
        self.env = env
        self.cellfn = cellfn
        self.hashfn = hashfn
        self.repeat = repeat
        self.nsteps = nsteps
        self.report = lambda: "Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d" % (self.iterations, len(self.record), self.frames, self.highscore)

    def ram(self):
        return self.env.env.clone_full_state()

    def restore(self, cell):
        ram, reward, length = cell.choose()
        self.env.env.restore_full_state(ram)
        self.reward = reward
        self.length = length

    def random(self):
        return self.env.action_space.sample()

    def getstate(self):
        return (
            self.ram(),
            self.reward,
            self.length,
        )

    def initialize(self):
        self.record = defaultdict(Cell)
        self.reward = 0
        self.action = 0
        self.length = 0
        self.frames = 0
        self.highscore = 0
        self.discovered = 0
        self.iterations = 0

        observation = self.env.reset()

        cell = self.cellfn(observation)
        code = self.hashfn(cell)
        cell = self.record[code]

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
            return True

        cell = self.cellfn(observation)
        code = self.hashfn(cell)
        cell = self.record[code]

        if self.update(cell):
            cell.setstate(self.getstate())
            self.discovered += 1

        return False

    def run(self, render=False):
        self.discovered = 0

        for i in range(self.nsteps):
            terminal = self.act(render)
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
