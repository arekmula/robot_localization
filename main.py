#!/usr/bin/env python

"""code template"""

import random
import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from graphics import *
from gridutil import *

import agents


class LocWorldEnv:
    actions = "turnleft turnright forward".split()

    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls
        self.action_sensors = []
        self.locations = {*locations(self.size)}.difference(self.walls)
        self.eps_perc = eps_perc
        self.eps_move = eps_move
        self.reset()

    def reset(self):
        self.agentLoc = random.choice(list(self.locations))
        self.agentDir = random.choice(['N', 'E', 'S', 'W'])

    def getPercept(self):
        p = self.action_sensors
        self.action_sensors = []
        rel_dirs = {'fwd': 0, 'right': 1, 'bckwd': 2, 'left': 3}
        for rel_dir, incr in rel_dirs.items():
            nh = nextLoc(self.agentLoc, nextDirection(self.agentDir, incr))
            prob = 0.0 + self.eps_perc
            if (not legalLoc(nh, self.size)) or nh in self.walls:
                prob = 1.0 - self.eps_perc
            if random.random() < prob:
                p.append(rel_dir)

        return p

    def doAction(self, action):
        points = -1
        if action == "turnleft":
            if random.random() < self.eps_move:
                # small chance that the agent will not turn
                print('Robot did not turn')
            else:
                self.agentDir = leftTurn(self.agentDir)
        elif action == "turnright":
            if random.random() < self.eps_move:
                # small chance that the agent will not turn
                print('Robot did not turn')
            else:
                self.agentDir = rightTurn(self.agentDir)
        elif action == "forward":
            if random.random() < self.eps_move:
                # small chance that the agent will not move
                print('Robot did not move')
                loc = self.agentLoc
            else:
                # normal forward move
                loc = nextLoc(self.agentLoc, self.agentDir)
            if legalLoc(loc, self.size) and loc not in self.walls:
                self.agentLoc = loc
            else:
                self.action_sensors.append("bump")
        return points  # cost/benefit of action

    def finished(self):
        return False


class LocView:
    # LocView shows a view of a LocWorldEnv. Just hand it an env, and
    #   a window will pop up.

    Size = .2
    Points = {'N': (0, -Size, 0, Size), 'E': (-Size, 0, Size, 0),
              'S': (0, Size, 0, -Size), 'W': (Size, 0, -Size, 0)}

    color = "black"

    def __init__(self, state, height=800, title="Loc World"):
        xySize = state.size
        win = self.win = GraphWin(title, 1.33 * height, height, autoflush=False)
        win.setBackground("gray99")
        win.setCoords(-.5, -.5, 1.33 * xySize - .5, xySize - .5)
        cells = self.cells = {}
        self.dir_cells = {}
        for x in range(xySize):
            for y in range(xySize):
                cells[(x, y)] = Rectangle(Point(x - .5, y - .5), Point(x + .5, y + .5))
                cells[(x, y)].setWidth(2)
                cells[(x, y)].draw(win)
                for dir in DIRECTIONS:
                    if dir == 'N':
                        self.dir_cells[(x, y, dir)] = Circle(Point(x, y + .25), .15)
                    elif dir == 'E':
                        self.dir_cells[(x, y, dir)] = Circle(Point(x + .25, y), .15)
                    elif dir == 'S':
                        self.dir_cells[(x, y, dir)] = Circle(Point(x, y - .25), .15)
                    elif dir == 'W':
                        self.dir_cells[(x, y, dir)] = Circle(Point(x - .25, y), .15)
                    self.dir_cells[(x, y, dir)].setWidth(1)
                    self.dir_cells[(x, y, dir)].draw(win)
        self.agt = None
        self.arrow = None
        ccenter = 1.167 * (xySize - .5)
        # self.time = Text(Point(ccenter, (xySize - 1) * .75), "Time").draw(win)
        # self.time.setSize(36)
        # self.setTimeColor("black")

        self.agentName = Text(Point(ccenter, (xySize - 1) * .5), "").draw(win)
        self.agentName.setSize(20)
        self.agentName.setFill("Orange")

        self.info = Text(Point(ccenter, (xySize - 1) * .25), "").draw(win)
        self.info.setSize(20)
        self.info.setFace("courier")

        self.update(state)

    def setAgent(self, name):
        self.agentName.setText(name)

    # def setTime(self, seconds):
    #     self.time.setText(str(seconds))

    def setInfo(self, info):
        self.info.setText(info)

    def update(self, state, P=None):
        # View state in exiting window
        for loc, cell in self.cells.items():
            if loc in state.walls:
                cell.setFill("black")
            else:
                cell.setFill("white")
                if P is not None:
                    for i, dir in enumerate(DIRECTIONS):
                        c = int(round(P[loc[0], loc[1], i] * 255))
                        self.dir_cells[(loc[0], loc[1], dir)].setFill('#ff%02x%02x' % (255 - c, 255 - c))
        if self.agt:
            self.agt.undraw()
        if state.agentLoc:
            self.agt = self.drawArrow(state.agentLoc, state.agentDir, 5, self.color)

    def drawArrow(self, loc, heading, width, color):
        x, y = loc
        dx0, dy0, dx1, dy1 = self.Points[heading]
        p1 = Point(x + dx0, y + dy0)
        p2 = Point(x + dx1, y + dy1)
        a = Line(p1, p2)
        a.setWidth(width)
        a.setArrow('last')
        a.setFill(color)
        a.draw(self.win)
        return a

    def pause(self):
        self.win.getMouse()

    # def setTimeColor(self, c):
    #     self.time.setTextColor(c)

    def close(self):
        self.win.close()


def main():
    random.seed(13)
    # rate of executing actions
    rate = 1
    # chance that perception will be wrong
    eps_perc = 0.1
    # chance that the agent will not move forward despite the command
    eps_move = 0.05
    # number of actions to execute
    n_steps = 40
    # size of the environment
    env_size = 16
    # map of the environment: 1 - wall, 0 - free
    map = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # build the list of walls locations
    walls = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                walls.append((j, env_size - i - 1))

    # create the environment and viewer
    env = LocWorldEnv(env_size, walls, eps_perc, eps_move)
    view = LocView(env)

    # create the agent
    agent = agents.prob.LocAgent(env.size, env.walls, eps_perc, eps_move)
    for t in range(n_steps):
        print('step %d' % t)

        percept = env.getPercept()
        action = agent(percept)
        # get what the agent thinks of the environment
        P = agent.getPosterior()

        print('Percept: ', percept)
        print('Action ', action)

        view.update(env, P)
        update(rate)
        # uncomment to pause before action
        view.pause()

        env.doAction(action)

    # pause until mouse clicked
    view.pause()


if __name__ == '__main__':
    main()
