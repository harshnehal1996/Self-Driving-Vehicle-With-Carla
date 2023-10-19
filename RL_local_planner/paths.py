import os

project_path = os.getcwd()

class ProjectPaths:
	expert_directory = os.path.join(project_path, 'collected_trajectories/')
	grid_dir = os.path.join(project_path, 'cache/image.png')
	path_to_save = os.path.join(project_path, 'weights/both/')
	carla_pylibs = 'path/to/carla_pylibs.egg' # example : '/home/user/carla_sim/carla/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg'
