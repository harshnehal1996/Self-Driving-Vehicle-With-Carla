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
	* Waypoint stored in carla is used to generate road view of the world(mapGenerator.py)
	* Goal point is chosen randomly in the road ahead
	* Reward of -10 is given if agent crosses boundary, +10 if reaches goal point and the game ends. Small penalty for choosing higher curvature curves.
	* Sparse reward : To deal with this I generated a reward vector(R) at each location on the road that points in the road direction(defined by waypoint) and has magnitude depending on the distance from the center of the lane. Agent is rewarded when it tries to make progress. The reward generated is integral(<R, ds>), where ds is agent displacement vec. Since R magnitude is greatest at center of lane, this means that agent is incentivized to stay in the middle. If this generated reward is negative then agent recieves -10 for moving in opposite direction and game ends.
	<p align="center"><img src="../images/Rwpaths.png" alt="magnitude" width="750" height="440"/></p>


* Training
	* Pixel wise features/embedding is created and fed to ConvNet : features are {cos_yaw, sin_yaw, goal_point, reward_vec_x, reward_vec_y}
	* After giving the entire map at once, the model didn't worked, possibly because of too many parameters. There was also too many straight road examples in the game which was causing it to overfit and only do straight line movements. To deal with this I used
		* Only local 128x128 map window with the player at the center when training. This reduces the number of conv layers and dense layer width.
		* I saved the environment state for trajectories where the model has previously failed. Instead of completely randomly sampling a newgame, now I also sampled and resumed from these failed trajectories which contained more curved road cases.
	* Some Results : Red is start, green is end
	<p align="center"><img src="../images/paths_1.png" alt="magnitude" width="450" height="420"/><img src="../images/paths_2.png" alt="magnitude" width="450" height="420"/></p>

## Dynamic Navigation
### Short Summary of Methods Tried
Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above Deriving from the feature representation from the static RL work above 


### Offline Model
















