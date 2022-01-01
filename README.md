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
* Install required python libs from requirement.txt

## To generate lidar map
1. Prepare Data
	* Run the docker with 
	```bash
	docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.10 bash -c "SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -carla-rpc-port=2000 -opengl"
	```
	* Run the data collection script: Spawns an agent that automatically drives and collect data inside the main carla map. press q to exit. After running the script you may want to stop the docker.
	```bash
	cd <project_dir>/data_collection_scripts/perception/mapping
	python3 collect_points.py <path to carla PythonAPI *.egg file>
	```
	* Output from the script : camera frames, lidar frames, segmentation frames, snaptimes and imu data produced in the same folder

	* Clone [R2D2](https://github.com/naver/r2d2) feature extractor somewhere in your system

	* Run the following commands to generate feature descriptors from R2D2 (check requirements for running R2D2). 
	```bash

	```



2. Build from makefile
	*

3. Run the binary
	*

## To run localization
1. Prepare Data
	* 


## To train RL agent


## To test RL agent

