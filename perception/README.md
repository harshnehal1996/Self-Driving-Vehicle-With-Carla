# Mapping and Localization

## Lidar Semantic Segmentation and Mapping

### Generated Lidar Map
<p align="center"><img src="../images/3.png" alt="lidar_seg" width="800" height="440"/><img src="../images/1.png" alt="lidar_seg" width="800" height="440"/></p>

* How to Run : [here](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla#to-generate-lidar-map)
* Main Program file : [create_map.cpp](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/perception/feature_extraction_and_mapping/create_map.cpp)

### Aim
* Build a static semantic segmentation map from pre segmented images using Lidar, Camera and IMU data
* Extract and store 3D keypoints in this map to enable any agent to localize using Camera

### Things to Consider
1. Program should be efficient enough to deal with large amounts of lidar data points
2. Accuracy of 3D location of the keypoints will affect the localization accuracy within it 
3. Accurate procedure to determine the class(road, lane marking, building, sidewalk) of a lidar data point


### Heuristic

#### 1. Preprocessing
1. I recorded the data which included the point cloud, 2D semantic segmentation, Camera and imu data from Carla(see [data_collection.py](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/data_collection_scripts/perception/localization/collect_trajectory.py))
2. Used [R2D2](https://github.com/naver/r2d2), which is Deep Learning based method optimized for saliency and repeatability to extract keypoints in an RGB image. This produces list of keypoints and its feature vector for each recorded RGB image.
3. Incase of unsynchronized recorded data, Match all the data from different sources with each other based on similarity in their timestamp
4. Store this synchronized data in keyframes. I stored it in the form of doubly-linked list.

#### 2. Projection : 
3. For every keyframe(K) define a set of "neighbor keyframes" : All keyframes that is atmost 40m away from "K".
4. Project each lidar point to the keyframe. In order to keep the overall complexity O(lidar points), projecting every lidar point to every keyframe must be avoided. Instead we do it the following way. 
	* For every lidar point

#### 3. Segmentation : 
5. Score and detemine class for every point projection:
	* 
6. 

#### 4. Keypoint extraction : 




## Localization 

<p align="center"><img src="../images/localization.gif" alt="Localization" width="1000" height="450"/> 
Left side is localized projection in the lidar map. The right side is the reference frame(from lidar map) detected for observation<br>
Matching pair of points are shown between reference and trajectory frame
</p>



