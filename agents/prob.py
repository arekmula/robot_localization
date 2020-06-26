# prob.py
# This is

import random
import numpy as np

from gridutil import *

best_turn = {('N', 'E'): 'turnright',
             ('N', 'S'): 'turnright',
             ('N', 'W'): 'turnleft',
             ('E', 'S'): 'turnright',
             ('E', 'W'): 'turnright',
             ('E', 'N'): 'turnleft',
             ('S', 'W'): 'turnright',
             ('S', 'N'): 'turnright',
             ('S', 'E'): 'turnleft',
             ('W', 'N'): 'turnright',
             ('W', 'E'): 'turnright',
             ('W', 'S'): 'turnleft'}


class LocAgent:

    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls
        # list of valid locations
        self.locations = list({*locations(self.size)}.difference(self.walls))
        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.eps_perc = eps_perc
        self.eps_move = eps_move

        # previous action
        self.prev_action = None

        # neighbours for each direction in percepts order (forward, right, backward, left)
        self.directions = {'N': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                           'E': [(1, 0), (0, -1), (-1, 0), (0, 1)],
                           'S': [(0, -1), (-1, 0), (0, 1), (1, 0)],
                           'W': [(-1, 0), (0, 1), (1, 0), (0, -1)]}


        # starting direction of robot
        self.dir = None

        # Transition Factor for each location.
        self.T = np.zeros((len(self.locations), len(self.locations)), float)
        np.fill_diagonal(self.T, 1)  # fill diagonal with initial probabilty that robot is there

        # Sensor factor for each location. Each location contains four possible directions
        self.sensor = np.ones((len(self.locations), 4, 1), float)


        self.P = None

    def __call__(self, percept, realLoc):
        # TODO: delete realLoc as it is help variable

        # update posterior
        # TODO PUT YOUR CODE HERE

        self.updateSensorFactor(percept, realLoc)


        # -----------------------

        action = 'forward'
        # TODO CHANGE THIS HEURISTICS TO SPEED UP CONVERGENCE
        # if there is a wall ahead then lets turn
        if 'fwd' in percept:
            # higher chance of turning left to avoid getting stuck in one location
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.8, 0.2])
        else:
            # prefer moving forward to explore
            action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.8, 0.1, 0.1])

        self.prev_action = action

        return action


    def updateSensorFactor(self, percept, realLoc):
        """
        This function updates sensor factor for each possible location and direction in this location.

        For example if we are in location (loc[0], loc[1]) and we are considering EAST direction and FORWARD percept
        then we have to check if there's wall in (loc[0]+1, loc[1]), as FORWARD in this case means EAST

        For example if we are in location (loc[0], loc[1]) and we are considering SOUTH direction and BACKWARD percept
        then we have to check if there's wall in (loc[0], loc[1]+1), as BACKWARD in this case means NORTH
        """
        # TODO: Remember to delete realLoc as it is used only for debugging.

        # reset sensor factor before updating it
        self.sensor[self.sensor>0] = 1

        for loc_idx, loc in enumerate(self.locations):  # loop over each location

            for dir_idx, neigh in enumerate(self.directions.values()):  # loop over each direction
                # for current considered direction check if there's wall in percept direction.

                if 'fwd' in percept:
                    if (loc[0] + neigh[0][0], loc[1] + neigh[0][1]) not in self.locations:
                        # if percept was correct (Sensor detected wall in this direction and it is there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9
                    else:
                        # if percept was NOT correct (Sensor detected wall in this direction, but it is NOT there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                else:
                    if (loc[0] + neigh[0][0], loc[1] + neigh[0][1]) not in self.locations:
                        # if lack of percept in this direction was NOT correct
                        # (Sensor didn't detect wall in this direction, but the wall is there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                    else:
                        # if lack of percept in this direction was correct
                        # (Sensor didn't detect wall in this direction and the wall is NOT there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9

                if 'right' in percept:
                    if (loc[0] + neigh[1][0], loc[1] + neigh[1][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                else:
                    if (loc[0] + neigh[1][0], loc[1] + neigh[1][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9

                if 'bckwd' in percept:
                    if (loc[0] + neigh[2][0], loc[1] + neigh[2][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                else:
                    if (loc[0] + neigh[2][0], loc[1] + neigh[2][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9

                if 'left' in percept:
                    if (loc[0] + neigh[3][0], loc[1] + neigh[3][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                else:
                    if (loc[0] + neigh[3][0], loc[1] + neigh[3][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.1
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0.9


        # TODO: delete realLoc usage
        realLoc_idx = self.loc_to_idx[realLoc]
        print(percept)
        print(self.sensor[realLoc_idx])





    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)

        # put probabilities in the array
        # TODO PUT YOUR CODE HERE


        # -----------------------

        return P_arr

    def forward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
