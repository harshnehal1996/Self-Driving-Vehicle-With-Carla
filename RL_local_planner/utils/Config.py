class config: # used only for legacy reason, should not be edited
    conv_size = [19, 38, 52, 70, 110, 150, 196]
    padding = [1, 0, 0, 0, 0, 0]
    kernel_size = [5, 3, 3, 3, 3, 4]
    num_action = 3
    v_scale = 6
    a_scale = 5
    w_scale = 40
    sigma_min = -40
    sigma_max = 4
    min_p = -40
    max_p = -0.009
    lookback = 10
    embedding_size = 8
    dynamic_size_x = 129
    dynamic_size_y = 129
    brake_scale = 0.9
    render = False
    optimized_memory = True
    dynamic_size = max(dynamic_size_x, dynamic_size_y)
    h_dynamic_size_x = 64
    h_dynamic_size_y = 64
    cache_size_x = 513
    cache_size_y = 513
    skips = 3
    fps = 15
    seed = None
    port = 8000
    hybrid = False 
    num_vehicle = 10
    num_pedestrian = 10
    expert_directory = '../collected_trajectories/'
    grid_dir = '../cache/image.png'
    path_to_save = ''
