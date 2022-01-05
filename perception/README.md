# Mapping and Localization

## Lidar Semantic Segmentation and Mapping

### Some Results
<p align="center"><img src="../images/3.png" alt="lidar_seg" width="800" height="440"/><img src="../images/1.png" alt="lidar_seg" width="800" height="440"/></p>

* How to Run : [here](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla#to-generate-lidar-map)
* Language used : C++

### Aim
* Build a static semantic segmentation map from pre segmented images using Lidar, Camera and IMU data
* Extract and store 3D keypoints in this map to enable any agent to localize using Camera

### Things to Consider
1. Program should be efficient enough to deal with large amounts of lidar data points
2. Accuracy of 3D location of the keypoints will affect the localization accuracy within it 
3. Accurate procedure to determine the class(road, lane marking, building, sidewalk) of a lidar data point

### Method
* I recorded the data which included the point cloud, 2D semantic segmentation, Camera and imu data from Carla(see [data_collection.py](https://github.com/harshnehal1996/Self-Driving-Vehicle-With-Carla/blob/master/data_collection_scripts/perception/localization/collect_trajectory.py))
*   



## Localization 

<p align="center"><img src="../images/localization.gif" alt="Localization" width="1000" height="450"/> 
Left side is localized projection in the lidar map. The right side is the reference frame(from lidar map) detected for observation<br>
Matching pair of points are shown between reference and trajectory frame
</p>



