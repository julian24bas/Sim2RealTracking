# Tracking Ball
run soccer.py

## Keys to initialize field
* Press 's' to set field.
* Press 'g' to set goal.
* Press 'h' to edit HSV-threshhold. When done press 'h' again.
* Press 'q' to quit program.

## Functions of soccer.py to get values for robot
```
get_ball_state()
```
returns list with five elements containing x-pixel-coordinate of ball,
y-pixel-coordinate of ball, bool of ball moving, bool if goal was scored since
last call of function, bool of ball visible (bool: 1.0 for True, 0.0 for False)

```
get_local_ball_position()
```
returns list with two elements containing x-world-coordinate of ball,
y-world-coordinate of ball

```
get_goal_transform()
```
returns list with three elements containing x-world-coordinate of goal,
y-world-coordinate of goal, rotation of goal around z-axis