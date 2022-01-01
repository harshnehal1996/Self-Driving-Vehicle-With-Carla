# Self-Driving-Vehicle-With-Carla

## Basic Setup Required
* Clone this repository
* Install carla 0.9.10 on your system.
	* You can choose to directly download the docker from [link](https://carla.readthedocs.io/en/latest/build_docker/)
	* Download or make .egg file for PythonAPI 0.9.10 see [make](https://carla.readthedocs.io/en/0.9.10/build_system/) or [download](https://github.com/carla-simulator/carla/releases)
	* Check if nvidia is supported with the docker 
	```bash
	sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
	``` 
	* To run carla with docker use
	```bash
	docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.10 bash -c "SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -carla-rpc-port=2000 -opengl"
	```
* Install required python libs from requirement.txt

## To generate lidar map
1. Prepare Data
	* Run the carla docker 
	
	* Run the data collection script: Spawns an agent that automatically drives and collect data inside the main carla map.Press "r" to start recording. press "q" to exit. After running the script you may want to stop the docker.
	```bash
	cd <project_dir>/data_collection_scripts/perception/mapping
	python3 collect_points.py <path to carla PythonAPI *.egg file>
	```
	* Output from the script in "out" folder: camera frames, lidar frames, segmentation frames, snaptimes and imu data

	* Clone [R2D2](https://github.com/naver/r2d2) feature extractor somewhere in your system

	* I have created a bash script to help extract keypoints for my use. You many edit the bash script to change any parameters for r2d2 to suit your need. Run the following commands to generate feature descriptors(check requirements for running R2D2). 
	```bash
	cd <project_dir>/perception/feature_extraction_and_mapping/
	chmod a+x extract_keypoints.sh
	./extract_keypoints.sh <r2d2 root path> <project_dir>/data_collection_scripts/perception/mapping/out/
	```

2. Run program
	* Install dependencies : opencv4, pcl>=1.7, openmp, jsoncpp
	
	* Build 
	```bash
	cd <project_dir>/perception/feature_extraction_and_mapping/
	cmake .
	make
	```

	* Run to generate and visualize map
	```bash
	./cmap <project_dir>/data_collection_scripts/perception/mapping/out/
	```

	* output saved in "map_out" in binary directory: contains keyframes, descriptors and each class's point cloud 


## To run localization
1. Prepare Data
	* Run the carla docker 

	* Run the data collection script: Spawns an actor. Control the actor using (w,a,s,d). Press "r" to start recording. press "q" to exit. After running the script you may want to stop the docker.
	```bash
	cd <project_dir>/data_collection_scripts/perception/localization
	python3 collect_trajectories.py <path to carla PythonAPI *.egg file>
	```

	* Output from the script in "trajectory_out" folder: camera frames, snaptimes and imu data

	* Run R2D2 feature extraction on the output, same as in the case of lidar map generation. See instructions above

2. Run program
	* Install dependencies : opencv4, pcl>=1.7, openmp, jsoncpp
	
	* Build : you can add extra flags to control use of float or double matrix, image display, ransac procedure
	```bash
	cd <project_dir>/perception/localization/
	cmake .
	make
	```

	* Run to generate localized path
	```bash
	./ukf <path to "map_out" dir generated from lidar mapping> <project_dir>/data_collection_scripts/perception/localization/trajectory_out/
	```

	* output saved in "path_out.json"


## To train RL agent
* Download and extract [collected_trajectories]() in the RL_local_planner folder
* Edit "carla_pylibs" attribute in path.py inside RL_local_planner with address of .egg carla pylibs 
* Run the carla docker
* Run training model : edit attributes inside config class of the model to control training params
```bash
cd <project_dir>/RL_local_planner
python3 main_<sac, ppo or a2c>.py <tensorboard log dir> <path to pretrained model. leave empty if no pretrained model>
```


