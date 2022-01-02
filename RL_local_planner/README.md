# Local Navigation using RL

## Trained model
See how video 

## Static path generation
* **Model**
	* Actor critic method 
	* The networks are not shared. Both processes data from Conv nets to linear layers
	* Actor outputs in discrete space in which it chooses one of the pre-generated spline curve(polynomial spirals). Each spirals are constained to have zero derivative and curvature at the end points which means that any series of choices will produce double differentiable curve. The plot below shows the spirals with y vs x displacement
	![img](../images/paths.png)


* **Environment** 
	* Waypoint stored in carla is used to generate road view of the world(mapGenerator.py)
	* Goal point is chosen randomly in the road ahead
	* Reward of -10 is given if agent crosses boundary, +10 if reaches goal point and the game ends. Small penalty for choosing higher curvature curves.
	* Sparse reward : To deal with this I generated a reward vector(R) at each location on the road that points in the road direction(defined by waypoint) and has magnitude depending on the distance from the center of the lane. Agent is rewarded when it tries to make progress. The reward generated is integral(<R, ds>), where ds is agent displacement vec. Since R magnitude is greatest at center of lane, this means that agent is incentivized to stay in the middle. If this generated reward is negative then agent recieves -10 for moving in opposite direction and game ends.
	<p align="center"><img src="../images/Rwpaths.png" alt="magnitude" width="750" height="440"/></p>


* **Training**
	* Pixel wise features/embedding is created and fed to ConvNet : features are {cos_yaw, sin_yaw, goal_point, reward_vec_x, reward_vec_y}
	* After giving the entire map at once, the model didn't worked, possibly because of too many parameters. There was also too many straight road examples in the game which was causing it to overfit and only do straight line movements. To deal with this I used
		* Only local 128x128 map window with the player at the center when training. This reduces the number of conv layers and dense layer width.
		* I saved the environment state for trajectories where the model has previously failed. Instead of completely randomly sampling a newgame, now I also sampled and resumed from these failed trajectories which contained more curved road cases.
	* Some Results : Red is start, green is end
	<p align="center"><img src="../images/paths_1.png" alt="magnitude" width="450" height="420"/><img src="../images/paths_2.png" alt="magnitude" width="450" height="420"/></p>

## Dynamic Navigation
### Short Summary of Methods Tried
Deriving from the feature representation of the static planner shown above, I added dynamic objects(vehicle and pedestrian) embedding via the GRU network, capturing the time series behavior of the object and helping with the markov assumption. The features is now 128x128x16 spatial embeddings containing information about player, static and dynamic objects stored in individual pixel. Tried [GAIL](https://arxiv.org/pdf/1606.03476.pdf) approach first with expert data collected from carla prebuilt agent but it didn't work because : The computation of double backpropogation graph(for computing Hv) was very slow, even optimized fisher product was struggling with gpu, the support for backprop on gradients is apparently not there for RNN class in pytorch. <br>
Tried A2C and PPO approach with continious and mixed action. Action space was {steering, throttle, brake, quit} in which quit was binary output used by the agent if it wants to quit the game(reward was then given based on how and where it has parked the vehicle). Time and acceleration penalty was also given. Reward of -10 was given for leaving the road or collision. I faced two main problems with these approaches. The agent regularly gets into deadlock situation and chooses not to move,for example when near the edge of the road it may see that it will get -10 reward in future thus brakes and doesn't move further. Reducing this negative reward and increasing time penalty didn't helped very much as the model was slow at learning as critic didn't converged to account for dispersed time penalty. Carla is a heavy simulator, with the online approaches being sample inefficient and hard to parallelize, the training was very slow and had to be abondoned due to limited hardware.


### Offline Model
* **Model**
















