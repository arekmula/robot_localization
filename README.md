# lokalizacja

Project goals:
- write a code that estimates probability distribution of robot's localization.
- modify robot's heuristic to speed up estimation of robot's localization. Robot should choose actions that give him more information about environment.

World assumptions:
- Robot doesn't know it's location
- Robot isn't rotating and moving forward correctly everytime. There's a chance (0.05) that robot will stay in the same position when the last command was 'forward' or that robot will not rotate when the last command was 'turnleft' or 'turnright'.
- Robot's sensors give us information when obstacles were detected. But they aren't perfect and give us wrong information. There's a chance (0.1) that sensors will give us wrong information - detects an obstacle when it is not there.

Tutaj umieść swój raport.
