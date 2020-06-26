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

        # forward neighbour for each direction (N, E, S, W)
        self.forward_neighbours = [(0, 1), (1, 0), (0, -1), (-1, 0)]



        # starting direction of robot
        self.dir = None

        # Transition Factor for each location.
        self.T = np.zeros((len(self.locations), len(self.locations)), float)
        np.fill_diagonal(self.T, 1)  # fill diagonal with initial probabilty that robot is there

        # probabilities of correct and failed move of robot on given action
        self.MOVE_CORRECT = 0.95
        self.MOVE_FAILED = 0.05

        # Direction Factor for each location
        # N E S W
        self.D = np.zeros((len(self.directions), len(self.directions)), float)
        np.fill_diagonal(self.D, 1)  # fill diagonal with initial probabilty that robot has this direction

        # Sensor factor for each location. Each location contains four possible directions
        self.sensor = np.ones((len(self.locations), len(self.directions), 1), float)

        # probabilities of correct and false values returned by sensor
        self.SENS_CORRECT = 0.9
        self.SENS_FALSE = 0.1

        # uniform posterior over valid locations
        prob_loc = 1.0/len(self.locations)
        # uniform posterior over valid directions
        prob_dir = 1.0/len(self.directions)

        self.P_loc = prob_loc * np.ones((len(self.locations),1), np.float)
        self.P_dir = prob_dir * np.ones((len(self.directions), 1), np.float)


    def __call__(self, percept, realLoc):
        # TODO: delete realLoc as it is help variable

        # update posterior
        # TODO PUT YOUR CODE HERE

        self.updateSensorFactor(percept, realLoc)
        self.updateTransitionFactor()


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
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT
                    else:
                        # if percept was NOT correct (Sensor detected wall in this direction, but it is NOT there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                else:
                    if (loc[0] + neigh[0][0], loc[1] + neigh[0][1]) not in self.locations:
                        # if lack of percept in this direction was NOT correct
                        # (Sensor didn't detect wall in this direction, but the wall is there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                    else:
                        # if lack of percept in this direction was correct
                        # (Sensor didn't detect wall in this direction and the wall is NOT there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT

                if 'right' in percept:
                    if (loc[0] + neigh[1][0], loc[1] + neigh[1][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                else:
                    if (loc[0] + neigh[1][0], loc[1] + neigh[1][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT

                if 'bckwd' in percept:
                    if (loc[0] + neigh[2][0], loc[1] + neigh[2][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                else:
                    if (loc[0] + neigh[2][0], loc[1] + neigh[2][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT

                if 'left' in percept:
                    if (loc[0] + neigh[3][0], loc[1] + neigh[3][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                else:
                    if (loc[0] + neigh[3][0], loc[1] + neigh[3][1]) not in self.locations:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                    else:
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT


        # TODO: delete realLoc usage
        realLoc_idx = self.loc_to_idx[realLoc]
        # print(percept)
        # print(self.sensor[realLoc_idx])


    def updateTransitionFactor(self):
        # if previous action was turn right or left then robot stayed in same position
        if self.prev_action == 'turnright' or self.prev_action == 'turnleft':
            # set to zero whole Transition factor and then fill diagonal with 1
            self.T[self.T > 0] = 0
            np.fill_diagonal(self.T, 1)
        # else if previous action was forward then robot moved to new location
        else:
            for loc_idx, loc in enumerate(self.locations):  # loop over each locations
                for neigh in self.forward_neighbours:  # loop over each forward location in each direction
                    new_loc = (loc[0] + neigh[0], loc[1] + neigh[1])  # forward location in considered direction

                    # check if forward location in considered direction is not wall
                    if new_loc in self.locations:
                        new_loc_idx = self.loc_to_idx[new_loc]  # find new location index
                        self.T[loc_idx, :] = 0  # set whole row to 0 before modyfing it

                        # probability that robot stayed in current location even though forward was last action
                        self.T[loc_idx, loc_idx] = self.MOVE_FAILED
                        # probability that robot moved to new location
                        self.T[loc_idx, new_loc_idx] = self.MOVE_CORRECT
                    else:
                        # if forward location in considered direction is wall
                        # that means that robot stayed in last location
                        self.T[loc_idx, :] = 0
                        self.T[loc_idx, loc_idx] = 1

        # print(self.T)



    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)

        # put probabilities in the array
        # TODO PUT YOUR CODE HERE
        for loc_idx, loc in enumerate(self.locations):
            P_arr[loc[0], loc[1]] = self.P_loc[loc_idx]
        for dir_idx, dir_loc in enumerate(self.directions.values()):
            P_arr[:, :, dir_idx] = self.P_dir[dir_idx]

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
