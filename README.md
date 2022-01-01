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

## To generate lidar map
1. Prepare Data
	* Run the docker with 
	```bash
	docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.10 bash -c "SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -carla-rpc-port=2000 -opengl"
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

