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

        # previous action
        self.prev_action = None

        # neighbours for each direction in percepts order (forward, right, backward, left)
        self.directions = {'N': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                           'E': [(1, 0), (0, -1), (-1, 0), (0, 1)],
                           'S': [(0, -1), (-1, 0), (0, 1), (1, 0)],
                           'W': [(-1, 0), (0, 1), (1, 0), (0, -1)]}

        # forward neighbour for each direction (N, E, S, W)
        self.forward_neighbours = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Transition Factor for each location.
        self.T = np.zeros((len(self.locations)*len(self.directions), len(self.locations)*len(self.directions)),
                          float)
        # fill diagonal with initial probabilty that robot is there and has this direction
        np.fill_diagonal(self.T, 1)

        # probabilities of correct and failed move of robot on given action
        self.MOVE_CORRECT = 1-eps_move
        self.MOVE_FAILED = eps_move

        # Sensor factor for each location. Each location contains four possible directions
        self.sensor = np.ones((len(self.locations), len(self.directions)), float)

        # probabilities of correct and false values returned by sensor
        self.SENS_CORRECT = 1-eps_perc
        self.SENS_FALSE = eps_perc
        self.SENS_BUMP = 1

        # uniform posterior over valid locations and directions
        prob_loc = 1.0/(len(self.locations)*len(self.directions))
        self.P = prob_loc * np.ones((len(self.locations) * len(self.directions), 1), np.float)

    def __call__(self, percept):

        # update posterior
        print(f"\n\n\nPrevious action: {self.prev_action}")
        print(f"Current percept: {percept}")
        self.update_sensor_factor(percept)
        self.update_transition_factor()
        self.update_posterior()
        action = self.heuristic(percept)

        return action

    def heuristic(self, percept):
        """
        Returns action that drives robot in a corner while touching wall, which give us more information about
        location probability.

        When we reach 85% confidence of robot location, then robot moves in a random way, not focusing on exploring
        the world
        """

        # find index of most probable location and direction
        loc_and_dir_idx = np.argmax(self.P[:, 0])
        # calculate index of location
        loc_idx = int(np.floor(loc_and_dir_idx/len(self.directions)))
        # calculate index of direction
        dir_idx = int(loc_and_dir_idx % len(self.directions))
        orientations = ['N', 'E', 'S', 'W']

        print(f"Most probable location: {self.locations[loc_idx]}  {orientations[dir_idx]}")
        print(f"Probability of robot being in this location: {round(self.P[loc_and_dir_idx, 0], 3)}")

        action = 'forward'

        # if we are not sure where robot is, plan robot move in a way that explore the world
        if self.P[loc_and_dir_idx, 0] < 0.85:
            if percept is not None:
                if 'fwd' in percept:
                    # if there's wall in front and on the left then turn right
                    if 'left' in percept and 'right' not in percept:
                        action = 'turnright'
                    # if there's wall in front and on the right then turn left
                    elif 'left' not in percept and 'right' in percept:
                        action = 'turnleft'
                    # if there's wall only in front then turn left or right
                    # to force robot to move while touching wall
                    else:
                        action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
                # force robot to move while touching wall
                elif 'right' in percept or 'left' in percept:
                    action = 'forward'
                # if there's wall in our back then turn right or turn left to touch wall
                else:
                    action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
            # if there's no percepts force robot to move forward
            else:
                print("NO PERCEPTS")
                action = 'forward'
        # heuristic when we are sure where robot is. Some random moves
        else:
            print("JUST MOVE")
            # if there is a wall ahead then lets turn
            if 'fwd' in percept:
                if 'left' in percept and 'right' not in percept:
                    action = 'turnright'
                elif 'left' not in percept and 'right' in percept:
                    action = 'turnleft'
                else:
                    action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
            else:
                # prefer moving forward to explore
                action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.95, 0.025, 0.025])

        self.prev_action = action

        return action


    def update_sensor_factor(self, percept):
        """
        This function updates sensor factor for each possible location and direction in this location.
        Checks how many percepts are correct for each direction in each location. Sensor might return false values
        sometimes so we have to consider that.

        For example if we are in location (loc[0], loc[1]) and we are considering EAST direction and FORWARD percept
        then we have to check if there's wall in (loc[0]+1, loc[1]), as FORWARD in this case means EAST

        For example if we are in location (loc[0], loc[1]) and we are considering SOUTH direction and BACKWARD percept
        then we have to check if there's wall in (loc[0], loc[1]+1), as BACKWARD in this case means NORTH

        """

        # reset sensor factor before updating it
        self.sensor[self.sensor > 0] = 1

        for loc_idx, loc in enumerate(self.locations):  # loop over each location

            for dir_idx, neigh in enumerate(self.directions.values()):  # loop over each direction
                # for current considered direction check if there's wall in percept direction.

                if 'fwd' in percept:
                    if (loc[0] + neigh[0][0], loc[1] + neigh[0][1]) not in self.locations:
                        if 'bump' in percept:
                            # if bump was detected, that means that this sensor reading was 100% correct
                            self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_BUMP
                        else:
                            # if percept was correct (Sensor detected wall in this direction and it is there)
                            self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_CORRECT
                    else:
                        # if percept was NOT correct (Sensor detected wall in this direction, but it is NOT there)
                        self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * self.SENS_FALSE
                else:
                    if (loc[0] + neigh[0][0], loc[1] + neigh[0][1]) not in self.locations:
                        if 'bump' in percept:
                            # if bump was detected and sensor returned nothing in forward direction that means that this
                            # sensor reading is 100% false
                            self.sensor[loc_idx, dir_idx] = self.sensor[loc_idx, dir_idx] * 0
                        else:
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

    def update_transition_factor(self):
        """
        Updates transition factor based on previous action.

        If robot turned then robot stayed in same position and changed its direction. For example if robot was facing
        North direction and previous action was turn right robot is facing EAST now. That means that we have to 'pass'
        probability from NORTH to EAST in each location with slight probability that robot failed its last action.
        And this is happening for each direction in each location (EAST -> SOUTH etc. for turn right
        and EAST->NORTH etc. for turn left).

        If robot moved forward then robot changed position and had same direction. For example if robot was in location
        (5, 9) and last action was forward we have to check if there is wall in each direction ( [5, 10] -> N,
        [6,9] -> E, [5, 8] -> S, [4, 9] -> W) and update transition factor based on this information and slight chance
        that robot failed its last move.
        """
        # if previous action was turn right then robot stayed in same position but changed its direction
        if self.prev_action == 'turnright':
            # set to zero whole Transition factor and then fill diagonal with 1
            self.T[self.T > 0] = 0
            np.fill_diagonal(self.T, 1)

            for loc_idx, loc in enumerate(self.locations):
                loc_idx_N = loc_idx * len(self.directions)  # index of NORTH direction for current location

                for dir_idx, direct in enumerate(self.directions):
                    # calculate index for each direction in location
                    loc_idx_D = loc_idx_N + dir_idx  # each location has 4 possible directions

                    # calculate index for right direction in compare to last direction
                    new_dir_idx_T = loc_idx_D + 1

                    # transition from W to N direction
                    if new_dir_idx_T > loc_idx_N + 3:
                        new_dir_idx_T = loc_idx_N

                    # set values for new direction and last direction
                    self.T[loc_idx_D, new_dir_idx_T] = self.MOVE_CORRECT
                    self.T[loc_idx_D, loc_idx_D] = self.MOVE_FAILED

        # if previous action was turn left then robot stayed in same position but changed its direction
        elif self.prev_action == 'turnleft':
            # set to zero whole Transition factor and then fill diagonal with 1
            self.T[self.T > 0] = 0
            np.fill_diagonal(self.T, 1)

            for loc_idx, loc in enumerate(self.locations):
                loc_idx_N = loc_idx * len(self.directions)  # index of NORTH direction for current location

                for dir_idx, direct in enumerate(self.directions):

                    # calculate index for each direction in location
                    loc_idx_D = loc_idx_N + dir_idx  # each location has 4 possible directions

                    # calculate index for left direction in compare to last direction
                    new_dir_idx_T = loc_idx_D - 1

                    # transition from N to E direction
                    if new_dir_idx_T < loc_idx_N:
                        new_dir_idx_T = loc_idx_N + 3

                    # set values for new direction and last direction
                    self.T[loc_idx_D, new_dir_idx_T] = self.MOVE_CORRECT
                    self.T[loc_idx_D, loc_idx_D] = self.MOVE_FAILED

        # else if previous action was forward then robot moved to new location and saved its direction
        else:
            for loc_idx, loc in enumerate(self.locations):
                for dir_idx, neigh in enumerate(self.forward_neighbours):
                    new_loc = (loc[0] + neigh[0], loc[1] + neigh[1])

                    # calculate index of location with direction in T matrix
                    loc_idx_D = loc_idx * len(self.directions) + dir_idx  # each location has 4 possible directions

                    if new_loc in self.locations:
                        new_loc_idx = self.loc_to_idx[new_loc] * len(self.directions) + dir_idx
                        self.T[loc_idx_D, :] = 0  # set whole row to 0 before modyfing it

                        # probability that robot stayed in current location even though forward was last action
                        self.T[loc_idx_D, loc_idx_D] = self.MOVE_FAILED
                        # probability that robot moved to new location
                        self.T[loc_idx_D, new_loc_idx] = self.MOVE_CORRECT
                    else:
                        # if forward location in considered direction is wall
                        # that means that robot stayed in last location
                        self.T[loc_idx_D, :] = 0
                        self.T[loc_idx_D, loc_idx_D] = 1

    def update_posterior(self):
        """
        Updates posterior for each location and directions in this location.
        Based on data from sensor and transitions of robot.
        """
        # reshape sensor array to match transition array shape
        sensor_reshaped = self.sensor.reshape([len(self.locations)*len(self.directions), 1])
        # transpose transition factor
        self.T = self.T.transpose()
        # update posterior
        self.P = sensor_reshaped * self.T.dot(self.P)
        # normalize posterior so its sum = 1
        self.P = self.P / self.P.sum(axis=0, keepdims=1)

    def get_posterior(self):
        """
        returns posterior of each location and directions in this location in array form
        """
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)

        for loc_idx, loc in enumerate(self.locations):
            # calculate index of location with NORTH direction
            loc_idx_N = loc_idx*len(self.directions)
            # get probabilities for all directions in current location
            P_arr[loc[0], loc[1], :] = self.P[loc_idx_N:loc_idx_N + 4, 0]
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
