import torch
from imports import device
from paths import ProjectPaths


class config:
    num_action = 2
    throttle_pos = [-0.6, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    steer_pos = [-0.7, -0.5, -0.3, -0.2, -0.1, -0.05,
                 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    v_scale = 6
    a_scale = 5
    w_scale = 40
    num_rays = 26
    random_reward = -1.5
    steering_gain = 0.5
    embedding_size = 16
    render = True
    camera_render = False
    save_every = 1
    tick_per_action = 2
    tick_after_start = 18
    continious_range = 1.0
    buffer_size = int(1e6)
    neta = 0.996
    p_min = -40
    sigma_min = -40
    sigma_max = 4
    min_alpha = [0.0, 0.1]
    max_alpha = [1.0, 1.0]
    save_dict_cycle = 5
    alpha = torch.Tensor([[0.2, 0.2]]).to(device)
    target_entropy_steering = 0.3
    target_entropy_throttle = 0.5
    batch_size = 128
    random_explore = 16000
    polyak = 0.995
    gamma = 0.99
    min_buffer_size = 16000
    step_per_lr_update = 4
    update_every = 156
    save_every = 1
    lr = 0.0003
    fps = 15
    seed = None
    sim_port = 2000
    port = 8000
    hybrid = False
    num_vehicle = 40
    num_pedestrian = 0
    # expert_directory = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/collected_trajectories/'
    # grid_dir = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/cache/image.png'
    # path_to_save = '/home/harsh/project_files/weights/both/'
    expert_directory = ProjectPaths.expert_directory
    grid_dir = ProjectPaths.grid_dir
    path_to_save = ProjectPaths.path_to_save
