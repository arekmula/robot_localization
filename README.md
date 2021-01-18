# Robot localization

![Final](https://i.imgur.com/4BcXcRd.png)

# Project goals:
- write a code that estimates probability distribution of robot's localization.
- modify robot's heuristic to speed up estimation of robot's localization. Robot should choose actions that give him more information about environment.

# World assumptions:
- Robot doesn't know it's orientation
- Robot isn't rotating and moving forward correctly everytime. There's a chance (**0.05**) that robot will stay in the same position when the last command was 'forward' or that robot will not rotate when the last command was 'turnleft' or 'turnright'.
- Robot's sensors give us information when obstacles were detected. But they **aren't perfect** and give us wrong information. There's a chance (**0.1**) that sensors will give us wrong information - detects an obstacle when it is not there or not detect an obstacle when it is there.

# Project requirements:
- Python 3.6 (might as well work with all of Python3 versions)
- listed in `requirements.txt`

# How it works?

### Updating sensor data
For each possible location and each possible direction we have to check if there are obstacles around current considered location and how those obstacles match with percepts returned by sensor. 

For example let's assume that sensor returned obstacles on **right** and **left**.
Let's consider location and direction given on image below.
- right percept match with wall on the right (0.9 chance of correct sensor data)
- left percept doesn't match, as there's no wall on the left (0.1 chance of wrong sensor data)
- forward and back percepts match with environment, as there's no walls in the back and in front (0.9 chance of correct sensor data)
This gives us 0.9 * 0.1 * 0.9 * 0.9 = 0,0729 probability that sensor data was read in this location with this orientation

Let's consider that robot is in location from image but this time is facing North. All percepts don't match with environment. This gives us 0.1 * 0.1 * 0.1 * 0.1 = 0,0001 probability that sensor data was read in this location with this orientation

![Image](https://i.imgur.com/A2OyF5k.png)


### Updating transitions
Turning: 
For example if robot was facing *North* direction and previous action was turn right robot is facing *East* now. That means that we have to *"pass"* probability from *North to East* in each location with slight probability (**0.05**) that robot failed its last action. And this is happening for each direction in each location (EAST -> SOUTH etc. for turn right and EAST->NORTH etc. for turn left).

Moving:
If the last action was *forward* then we have to check if there's wall in front of robot for each location and each direction. If there's no wall in front of robot for current considered location and direction that means that robot moved forward with **0.95** chance and stayed in last position with **0.05** chance. If there's wall in front of robot for current considered location and direction that means that robot stayed in last position.

### Updating posterior for each location and direction
At the beginning every location and direction has the same probability that robot is there.
When robot moved data from sensor and data from transitions are multiplied by probability from previous step for each location and direction.

### Heuristics
Heuristic forces robot to go into corner while sticking wall. When we reach 85% confidence of robot location then robot moves in a random way.

