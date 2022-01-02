# Local Navigation using RL

## Trained model
See how video 

## Static path generation
* Model
	* Actor critic method 
	* The networks are not shared. Both processes data from Conv nets to linear layers
	* Actor outputs in discrete space in which it chooses one of the pre-generated spline curve(polynomial spirals). Each spirals are constained to have zero derivate and curvature at the end points which means that any series of choices will produce double differentiable curve. The plot below shows this with y vs x displacement
	![img](../images/paths.png)


* Environment 
	* Waypoint stored in carla is used to generate road view of the world
	* Goal point is chosen randomly in the road ahead
	* Reward of -10 is given if agent crosses boundary. +10 if reaches goal point and the game ends 
	* Sparse reward : To deal with this I generated a reward vector(R) at each location on the road that points in the road direction(defined by waypoint) and has magnitude depending on the distance from the center of the lane. Agent is rewarded when it tries to make progress. The reward generated is integral(<R, ds>). Since R magnitude is greatest at center of lane, this means that agent is incentivized to stay in the middle. If this generated reward is negative then agent recieves -10 for moving in opposite direction and game ends.
	<img src="../images/Rwpaths.png" alt="magnitude" width="800" height="480"/>


* Training
	* 
