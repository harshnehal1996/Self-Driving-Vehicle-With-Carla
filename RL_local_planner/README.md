# Local Navigation using RL

## Trained model
See how video 

## Static path generation
* Model
	* Actor critic method 
	* Each network has its own network starting from few Conv nets to linear layers
	* Actor outputs in discrete space in which it chooses one of the pre-generated spline curve(polynomial spirals). Each spirals are constained to have zero derivate and curvature at the end points which means that any series of choices will produce double differentiable curve. The plot below shows this with y vs x displacement
	![img](../images/paths.png)


* Environment 
	* Waypoint stored in carla is used to generate road view of the world
	* Goal point is chosen randomly in the road ahead
	* Reward of -10 is given if agent crosses boundary. +10 if reaches goal point
	* Sparse reward
* Training
	* 
