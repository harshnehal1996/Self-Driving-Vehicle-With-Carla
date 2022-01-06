# Mapping and Localization

## Lidar Semantic Segmentation and Mapping

### Generated Lidar Map
<p align="center"><img src="../images/3.png" alt="lidar_seg" width="800" height="440"/><img src="../images/1.png" alt="lidar_seg" width="800" height="440"/></p>

* How to Run : [here](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla#to-generate-lidar-map)
* Main Program file : [create_map.cpp](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/perception/feature_extraction_and_mapping/create_map.cpp)

### Target
* Build a static semantic segmentation map from pre segmented images using Lidar, Camera and IMU data
* Extract and store 3D keypoints in this map to enable any agent to localize using Camera

### Things to Consider
1. Program should be efficient enough to deal with large amounts of lidar data points
2. Accuracy of 3D location of the keypoints will affect the localization accuracy within it
3. Accurate procedure to determine the class(road, lane marking, building, sidewalk) of a lidar data point


### Heuristic
#### 1. Data Collection:
I recorded the data which included the point cloud, 2D semantic segmentation, Camera and imu data from Carla(see [data_collection.py](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/data_collection_scripts/perception/localization/collect_trajectory.py))

#### 2. Preprocessing:
1. Used [R2D2](https://github.com/naver/r2d2), which is Deep Learning based method optimized for saliency and repeatability to extract keypoints in an RGB image. This produces list of keypoints and its feature vector for each recorded RGB image.
2. Incase of unsynchronized recorded data, Synchronize by matching the timestamps across sources.
3. Store this synchronized data in keyframes(same as camera-frame containing the 2D semantic segmentation). I stored it in the form of doubly-linked **List**.

#### 3. Projection: 
Since we have the semantic segmentation images we can project lidar points onto to the cameraframe to find its class.

In order to keep the overall complexity O(num lidar points), projecting every lidar point to every cameraframe must be avoided. To make it efficient I did two things
1. Parallelized the entire operation across multiple cores using openmp.
2. Only project to the cameraframes that are nearby to the lidar point.
In order to achieve this, before projecting a lidar point, first the nearest cameraframe(**K**) to the point is searched in the linked-list. Then the point is only projected to the cameraframes that are less than 40m away from "K". 

#### 3. Segmentation: 
Each lidar points has to be classified amoung fixed number of classes. For this we score the point based on many different projections. In the previous step let say the lidar point(**P**) was projected in N different camera frames. From these projection, an inverse distance weighted  score is formed for all different types of semantic classes. Thus if in the "ith" projection of "P", the class was "j" then the "jth" class gets 
1 / (distance of "P" in "i") extra score and all other class gets zero. This means we give more weightage to nearby observation than far away which I found out to work well with real-world lane segmentation data where the model confidence decreases the farther the markings are.

To improve accuracy further two other factors were considered.
1. **See-through projection elimination**: 
	* Since lidar points are directly projected onto the cameraframes without any regards to its visibility this means that lidar points behind any opaque objects can still show up in the projection. To eliminate these projections I do the following. *Execute Parallelly*:
		1. Divide the cameraframe in small *sized* 2D voxels(size depend on projected points density inside the voxel)
		2. I assume that the lowest distance projection point(**P**) in the voxel is visible in the cameraframe.
		3. Perform a **DBScan** clustering(based on distance) from just the point "P" to find points that are located closeby in space
		4. Eliminate all the other points not part of this cluster


2. **Obscure projection elimination**:
	* Edges can be a tricky case to deal since a slight over-extension of labels around it in the 2D segmentation image can potentially affect large number of lidar points behind it. You can imagine this by visualizing how fast the shawdow of a sphere on a wall behind it grows in size if we move the sphere a tiny amount towards the light source. To eliminate this I clipped the segmented regions from the outward boundary(having radially outward gradient) of the objects. The gradients/boundary were detected using Sobel filter. Visualization below for semantic class "lane marking". Red is outward boundary(where labels are omitted) and green is the inner boundary(untouched). 
	<p align="center"><img src="../images/compare.png" alt="lidar_seg" width="450" height="300"/><img src="../images/grad.png" alt="lidar_seg" width="450" height="300"/></p>


#### 4. Keypoint extraction: 




## Localization 

### Some Result
<p align="center"><img src="../images/localization.gif" alt="Localization" width="1000" height="450"/> 
Left side is localized projection in the lidar map. The right side is the reference frame(from lidar map) detected for observation<br>
Matching pair of points are shown between reference and trajectory frame
</p>

* How to Run : [here](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla#to-run-localization)
* Main Program file : [main.cpp](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/perception/localization/main.cpp)




