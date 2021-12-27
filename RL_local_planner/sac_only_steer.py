#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob

# try:
#     sys.path.append(glob.glob('/home/harsh/Documents/carla_sim/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

sys.path.append(glob.glob('/home/ubuntu/carla-0.9.10-py3.6-linux-x86_64.egg')[0])


import pygame
import queue
import carla
import pickle
import traceback
import cv2
import time
import shutil
import math
from copy import deepcopy
from PIL import Image
from scipy.sparse import lil_matrix
from collections import OrderedDict
from carla import VehicleLightState as vls

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.categorical as C
import torch.distributions.bernoulli as bn
from torch.utils.tensorboard import SummaryWriter

import Dataset


use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')


class ActorManager(object):
    def __init__(self, client, world, tm, map_ratio, ego_actor='vehicle.audi.a2'):
        self.client = client
        self.world = world
        self.traffic_manager = tm
        self.ego_type = ego_actor
        self.all_vehicle_objects = []
        self.all_pedestrian_objects = []
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.box = [[], []]
        self.vehicle_objects = []
        self.pedestrian_objects = []
        self.map_ratio = map_ratio
    
    def __fill_measurement_grid(self, idx, measurement, resolution=0.75):
        
        for i in range(len(measurement)):
            x1, x2 = np.floor(measurement[i][0][0] * self.map_ratio), np.ceil(measurement[i][-1][0] * self.map_ratio)
            y1, y2 = np.floor(measurement[i][0][1] * self.map_ratio), np.ceil(measurement[i][-1][1] * self.map_ratio)
            x = np.arange(x1, x2 + resolution / 2, resolution)
            y = np.arange(y1, y2 + resolution / 2, resolution)
            xx, yy = np.meshgrid(x, y)
            xx = np.expand_dims(xx, -1)
            yy = np.expand_dims(yy, -1)
            grid = np.concatenate([xx, yy], axis=-1).reshape(-1, 2).T
            self.box[idx].append(grid)

    def spawn_npc(self, number_of_vehicles, number_of_walkers, safe=True, car_lights_on=False):
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        num_ego_vehicle=0

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            np.random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            print('requested %d vehicles, but could only find %d spawn points' % (number_of_vehicles, number_of_spawn_points))
            number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            if n < num_ego_vehicle:
                blueprint = np.random.choice(self.world.get_blueprint_library().filter(self.ego_type))
            else:
                blueprint = np.random.choice(blueprints)
            
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot') if n >= num_ego_vehicle else blueprint.set_attribute('role_name', 'hero_' + str(n))

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                print(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        percentagePedestriansRunning = 0.2      # how many pedestrians will run
        percentagePedestriansCrossing = 0.2     # how many pedestrians will walk through the road
        
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = np.random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (np.random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        self.world.tick()
        
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
        self.traffic_manager.global_percentage_speed_difference(30.0)
        
        self.np_vehicle_objects = [actor for actor in self.world.get_actors()                                   if 'vehicle' in actor.type_id                                    and 'hero' not in actor.attributes['role_name']]
        
        self.np_pedestrian_objects = [actor for actor in self.world.get_actors() if 'walker.pedestrian' in actor.type_id]
        
        measurements = []
        
        for i in range(len(self.np_vehicle_objects)):
            vertex = self.np_vehicle_objects[i].bounding_box.get_local_vertices()
            measurements.append([])
            for v in vertex:
                measurements[i].append([v.x, v.y, v.z])
        
        self.__fill_measurement_grid(0, measurements)
        measurements = []
        
        for i in range(len(self.np_pedestrian_objects)):
            vertex = self.np_pedestrian_objects[i].bounding_box.get_local_vertices()
            measurements.append([])
            for v in vertex:
                measurements[i].append([v.x, v.y, v.z])
        
        self.__fill_measurement_grid(1, measurements)        
        

    def destroy_all_npc(self):
        if not len(self.vehicles_list):
            return

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(1)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.box = [[], []]
        self.np_pedestrian_objects = []
        self.np_vehicle_objects = []


# In[6]:


class CarlaSyncMode(object):
    def __init__(self, client, world, no_rendering, *sensors, **kwargs):
        self.world = world
        self.sensors = []
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.rendering = 0
        self.no_rendering = no_rendering
        self.offset = 1
        self.world_callback_id = -1
        self.callback = None
        self.ret = None

    def enter(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=self.no_rendering,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        
        time.sleep(1)

    def make_queue(self, register_event):
        q = queue.Queue()
        callback_id = register_event(q.put)
        self._queues.append(q)
        return callback_id
    
    def add_render_queue(self, sensor):
        if len(self.sensors):
            return -1

        self.sensors.append(sensor)
        self.offset = 2
        self.rendering = 1
        self.make_queue(sensor.listen)
        return 0
    
    def add_main_queue(self):
        if len(self._queues):
            return
        
        self.world_callback_id = self.make_queue(self.world.on_tick)
    
    def add_sensor_queue(self, sensor):
        self.sensors.append(sensor)
        self.make_queue(sensor.listen)
        return len(self.sensors) - 1

    def reset(self, hard_reset=False):
        self.offset = 1
        self.rendering = 0
        
        for i in range(len(self.sensors)):
            try:
                if self.sensors[i] is not None:
                    self.sensors[i].destroy()
            except:
                traceback.print_exception(*sys.exc_info())
        
        self.sensors = []
        n = len(self._queues)
        for i in range(1, n):
            try:
                del self._queues[1]
            except:
                traceback.print_exception(*sys.exc_info())
        
        if hard_reset:
            self.world.remove_on_tick(self.world_callback_id)
            self.world_callback_id = -1
            del self._queues[0]
            self._queues = []
            self.ret = None
            self.callback = None

    def remove_sensor_queue(self, id):
        try:
            self.sensors[id].destroy()
        except:
            traceback.print_exception(*sys.exc_info())
        
        self.sensors[id] = None
        
        if id == 0 and self.rendering:
            self.rendering = 0
            self.offset = 1

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(self._queues[i], timeout) for i in range(self.offset)]
        assert all(x.frame == self.frame for x in data)
        data = data + [self._retrieve_data(self._queues[i], None, block=False) for i in range(self.offset, len(self._queues))]
        if self.callback:
            self.ret = self.callback(data[0])
        
        return data

    def exit(self, *args, **kwargs):
        self._settings.no_rendering_mode = False
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout, block=True):        
        while True:
            if not block and sensor_queue.empty():
                return None
            data = sensor_queue.get(block=block, timeout=timeout)
            if data.frame == self.frame:
                return data

# In[7]:

class config:
    conv_size = [19, 38, 52, 70, 90, 120, 169]
    padding = [1, 0, 0, 0, 0, 0]
    kernel_size = [5, 3, 3, 3, 3, 4]
    num_action = 3
    throttle_pos = [-0.6, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    steer_pos = [-0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    v_scale = 6
    a_scale = 5
    w_scale = 40
    steering_gain = 0.5
    p_min = -25
    p_max = -0.009
    lookback = 10
    embedding_size = 8
    dynamic_size_x = 129
    dynamic_size_y = 129
    brake_scale = 0.9
    render = True
    use_shared_networks = False
    camera_render = False
    memory_optimized = True
    dynamic_size = max(dynamic_size_x, dynamic_size_y)
    h_dynamic_size_x = 64
    h_dynamic_size_y = 64
    cache_size_x = 513
    cache_size_y = 513
    save_every = 1
    tick_per_action = 2
    tick_after_start = 15
    max_w = 1
    use_tanh_clip = True
    skips = 2
    fps = 15
    seed = None
    port = 8000
    hybrid = False 
    num_vehicle = 0
    num_pedestrian = 0
    # expert_directory = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/collected_trajectories/'
    # grid_dir = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/cache/image.png'
    # path_to_save = '/home/harsh/project_files/weights/'
    expert_directory = '/home/ubuntu/project_files/collected_trajectories/'
    grid_dir = '/home/ubuntu/project_files/cache/image.png'
    path_to_save = '/home/ubuntu/project_files/weights/'


# In[8]:


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
carla_map = world.get_map()


# In[9]:


directories = os.listdir(config.expert_directory)


# In[10]:


dumps = []
for d in directories:
    file = os.path.join(config.expert_directory, d, 'trajectories')
    try:
        with open(file, 'rb') as dbfile:
            dumps.append(pickle.load(dbfile))
    except:
        pass


# In[11]:


ds = Dataset.Dataset(dumps, carla_map, 1024, 1024, config.grid_dir)


# In[ ]:

# throttle has no negative positions available ? -> equivalent to const throttle at different
# use const throttle to train for 50k trajectories -> (pi, v)
# freeze the upper network and train the throttle network -> same problem..? unseen states by the steering

# remove negative reward completely when going out of range? only keep positive rewards(might as well increase) for field.\
# -> problem : when encountering a moving obstacle. He may just suboptimally jump out of the map?
# small time negative reward when at low speed. -5 reward if hit obstacle. 0 when out of the map. lateral_g rewards = 0
# +5 reward if reach target. -5 for wrong direction.
# other rewards come from field follow.
# only one brake option with -0.6 intensity

# field_rw = 0.6
# out_rw = -1
# low speed rw = -0.02
# high speed rw = -1
# collision_rw = -2



class Environment(object):    
    def __init__(self, num_hero, dataset, client, world, max_rw_distance=20):
        self.dgen = dataset
        self.mgen = self.dgen.mgen
        self.window = dataset.window
        self.h_dynamic_size_x = dataset.h_dynamic_size_x
        self.h_dynamic_size_y = dataset.h_dynamic_size_y
        self.top = dataset.top
        self.map_ratio = dataset.map_ratio
        self.mgen = dataset.mgen
        self.center = dataset.center
        self.radius = dataset.radius
        self.h_cache_size_y = config.cache_size_y // 2
        self.h_cache_size_x = config.cache_size_x // 2
        self.hero_box = self.dgen.box[0][0]
        self.speed_on_stop_th = dataset.speed_on_stop_th
        self.min_safe_distance = 5
        self.max_heros = num_hero
        self.hero_list = []
        self.active_actor = []
        self.min_field_reward = 0.03
        self.beta = 0.5
        self.random_theta_variation = 16
        self.cos_max_theta_stop = 1 / np.sqrt(2)
        self.resolution = 0.09
        self.dt = max(config.skips, 1) / config.fps
        self.MAX_TRY_STOP_SPEED = 1
        self.MIN_TRY_STOP_SPEED = 5
        self.dist = self.compute_shortest_distance_matrix()
        self.SPEED_PENALTY_UPPER_LIMIT = 60 / 3.6
        self.SPEED_PENALTY_LOWER_LIMIT = 2 / 3.6
        
        self.g_rw = -0.08
        self.max_field_reward = 0.6
        self.boundary_cross_reward = -1
        self.low_speed_penalty = 0.02
        self.time_penalty = 0
        self.high_speed_penalty = 1
        self.MAX_REWARD = 10
        self.MIN_REWARD = -5
        self.collision_reward = -2
        self.reward_drop_rate = self.MAX_REWARD / (max_rw_distance ** self.beta)
        
        self.sync_mode = None
        self.am = None
        self.world = world
        self.client = client
        self.blueprint_library = None
        self.camera_semseg = None
        self.is_initialized = False
        self.print_ = False
        self._path = []
        self.throttle_map = dict([(i, y) for i, y in enumerate(config.throttle_pos)])
        self.steer_map = dict([(i, y) for i, y in enumerate(config.steer_pos)])
        self.npc_information = {'current_timestamp' : -1, 'location_queue' : []}

    def __has_affecting_landmark(self, waypoint, search_distance):
        lmark = waypoint.get_landmarks_of_type(search_distance, '1000001')
        
        for i in range(len(lmark)):
            wp = lmark[i].waypoint
            if wp.road_id == waypoint.road_id and wp.lane_id * waypoint.lane_id > 0:
                return True

        return False

    def __create_cached_map(self, state, s_pt, pos):
        cached_map = torch.Tensor(config.conv_size[0], config.cache_size_y, config.cache_size_x)
        static_map = state[0]
        img = static_map[0]
        stop_target, through_target = static_map[-1]
        occ_pix = self.dgen.world_to_pixel(pos, offset=(-self.h_cache_size_x, -self.h_cache_size_y))
        reward_field_x, reward_field_y = static_map[1]
        cached_map[0] = torch.from_numpy(reward_field_x[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[1] = torch.from_numpy(reward_field_y[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[2] = torch.from_numpy(img[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[3] = self.dgen.drivable[occ_pix[1]-self.h_cache_size_y : occ_pix[1]+self.h_cache_size_y+1, occ_pix[0]-self.h_cache_size_x : occ_pix[0]+self.h_cache_size_x+1]
        cached_map[4] = self.dgen.lanes[occ_pix[1]-self.h_cache_size_y : occ_pix[1]+self.h_cache_size_y+1, occ_pix[0]-self.h_cache_size_x : occ_pix[0]+self.h_cache_size_x+1]
        cached_map[5] = torch.from_numpy(stop_target[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[6 : 6 + config.embedding_size] = 0
        cached_map[6 + config.embedding_size :]  = 0
        offset = s_pt.copy() - np.array([self.h_cache_size_x, self.h_cache_size_y])
        state[2] = [cached_map, offset]
    
    def __get_angle_diff(self, a, b):
        choice = [360 + a - b, a - b, -360 + a - b]
        minimum = abs(choice[0])
        index = 0

        for i in range(1, len(choice)):
            if abs(choice[i]) < minimum:
                minimum = abs(choice[i])
                index = i

        return np.sign(choice[index]) * minimum

    def __append_dynamic_features(self, idx, state, pos, dynamic_objects):
        if not len(dynamic_objects):
            state[3].append([[], [], []])
            return
        
        rnn_features = np.zeros((config.lookback, len(dynamic_objects), 4), dtype=np.float32)

        skips = max(config.skips, 1)
        measurement = []
        
        position = self.npc_information['location_queue']
        obs_index = len(position) - 1
        
        for k, index in enumerate(dynamic_objects):
            t = 0
            
            for r in range(obs_index, skips - 1, -skips):
                if t == config.lookback:
                    break
                if t == 0:
                    cur = position[r][idx][index]
                    prev = position[r - skips][idx][index]
                else:
                    cur = prev
                    prev = position[r - skips][idx][index]
                
                rnn_features[config.lookback - t - 1, k] = np.array([(cur[0] - prev[0]) / (self.dt * config.v_scale), (cur[1] - prev[1]) / (self.dt * config.v_scale), self.__get_angle_diff(cur[2], prev[2]) / (self.dt * config.w_scale), idx])
                t += 1
            
            if t < config.lookback:
                assert t
                print('found insufficient data, required=%d, available=%d' % (config.lookback + 1, t))
                current_features = rnn_features[config.lookback - t, k]
                for r in range(t, config.lookback):
                    rnn_features[config.lookback - r - 1, k] = current_features
        
        rnn_features = torch.from_numpy(rnn_features)
        for i in range(len(dynamic_objects)):
            measurement.append(self.am.box[idx][dynamic_objects[i]])
        
        state[3].append([rnn_features, pos, measurement])
    
    def update_db(self, timestep):
        timestep = timestep.elapsed_seconds
        
        if self.npc_information['current_timestamp'] == timestep:
            return True

        info_queue = self.npc_information['location_queue']
        position = [[], []]
        
        for i, v in enumerate(self.am.np_vehicle_objects):
            transform = v.get_transform()
            x, y = transform.location.x, transform.location.y
            yaw = transform.rotation.yaw
            position[0].append([x, y, yaw])

        for i, v in enumerate(self.am.np_pedestrian_objects):
            transform = v.get_transform()
            x, y = transform.location.x, transform.location.y
            yaw = transform.rotation.yaw
            position[1].append([x, y, yaw])

        info_queue.append(position)
        self.npc_information['current_timestamp'] = timestep

        if len(info_queue) > config.lookback * config.skips + 1:
            info_queue.pop(0)
            return True

        return False
    
    def remove_db_info(self):
        del self.npc_information
        self.npc_information = {'current_timestamp' : -1, 'location_queue' : []}

    def compute_shortest_distance_matrix(self, save_mat=True):
        road_ids = {}
        n = 0
        m = len(self.top)
        adj = {}
        self.keys = []

        for i in range(m):
            if self.top[i][0].road_id == self.top[i][1].road_id:
                continue
            
            s1 = self.top[i][0].road_id
            s2 = self.top[i][0].lane_id > 0
            
            if adj.__contains__((s1, s2)):
                adj[(s1, s2)].append((self.top[i][0], self.top[i][1]))
            else:
                adj[(s1, s2)] = [(self.top[i][0], self.top[i][1])]
            
            s1 = self.top[i][1].road_id
            s2 = self.top[i][1].lane_id > 0
            
            if not adj.__contains__((s1, s2)):
                adj[(s1, s2)] = []
            
            if not road_ids.__contains__((self.top[i][0].road_id, self.top[i][0].lane_id > 0)):
                road_ids[(self.top[i][0].road_id, self.top[i][0].lane_id > 0)] = n
                self.keys.append((self.top[i][0].road_id, self.top[i][0].lane_id > 0))
                n += 1
            
            if not road_ids.__contains__((self.top[i][1].road_id, self.top[i][1].lane_id > 0)):
                road_ids[(self.top[i][1].road_id, self.top[i][1].lane_id > 0)] = n
                self.keys.append((self.top[i][0].road_id, self.top[i][0].lane_id > 0))
                n += 1
        
        self.adj = adj
        self.num_nodes = n
        
        assert n == len(self.adj)
        
        path = os.path.join(config.expert_directory, 'dist_map1.npy')
        if os.path.isfile(path):
            dist = np.load(path)
            return dist
        
        dist = [[np.inf if i != j else 0 for j in range(n)] for i in range(n)]

        for e in range(m):
            i = road_ids[(self.top[e][0].road_id, self.top[e][0].lane_id > 0)]
            j = road_ids[(self.top[e][1].road_id, self.top[e][1].lane_id > 0)]
            if i != j:
                dist[i][j] = 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        dist = np.array(dist)
        
        if save_mat:
            with open(path, 'wb') as mat:
                np.save(mat, dist)            

        return dist

    def get_safe_start_point(self, c):
        nodes = list(range(self.num_nodes))
        
        while len(nodes):
            idx = np.random.randint(len(nodes))
            w = nodes[idx]
            minimum = np.inf

            for u in self.active_actor:
                minimum = min(minimum, self.dist[u][w], self.dist[w][u])
                if minimum <= c:
                    nodes.pop(idx)
                    break

            if minimum > c:
                return w
        
        return -1

    def try_spawn_hero(self, s1, respawn_distance=4, max_respawn_attempt=5):
        for i in range(max_respawn_attempt):
            transform = s1.transform
            transform.rotation.yaw += np.clip(np.random.randn(), -1, 1) * self.random_theta_variation
            vehicle = self.world.try_spawn_actor(self.blueprint_library.filter('vehicle.audi.a2')[0], transform)
            if vehicle is None:
                snew = s1.next(respawn_distance)
                if not len(snew):
                    print('unable to find spawn point!')
                    return None, -1

                if s1.road_id != snew[0].road_id:
                    print('unable to spawn actor!')
                    return None, -1

                s1 = snew[0]
            else:
                if config.camera_render and len(self.active_actor) == 0:
                    self.camera_semseg = self.world.spawn_actor(
                        self.blueprint_library.find('sensor.camera.semantic_segmentation'),
                        carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                        attach_to=vehicle)
                
                collision_sensor = self.world.spawn_actor(
                    self.blueprint_library.find('sensor.other.collision'),
                    carla.Transform(), attach_to=vehicle)
                self.hero_list.append(vehicle)
                id = self.sync_mode.add_sensor_queue(collision_sensor)
                self.sync_mode.tick(timeout=2)  # should not be done during a trajectory
                
                return s1, id

        return None, -1

    def try_spawn_random_trajectory(self, num_roads, search_distance=30):
        if len(self.active_actor) > 0:
            if len(self.active_actor) >= self.max_heros:
                return [], -1

            w = self.get_safe_start_point(max(num_roads + 2, self.min_safe_distance))
            
            if w == -1:
                print('unable to find node with distance safe_distance=%d', max(num_roads + 2, self.min_safe_distance))
                return [], -1
        else:
            w = np.random.randint(self.num_nodes)

        r, l = self.keys[w]
        out_len = len(self.adj[(r, l)])
        if out_len == 0:
            return [], -1
        
        safe_1, safe_2 = self.adj[(r, l)][np.random.choice(out_len)]
        r, l = safe_2.road_id, safe_2.lane_id > 0
        out_len = len(self.adj[(r, l)])
        if out_len == 0:
            return [], -1
        
        res = [safe_1]
        s1, s2 = self.adj[(r, l)][np.random.choice(out_len)]

        snew = s1.next(3)
        if len(snew) and s1.road_id == snew[0].road_id:
            s1 = snew[0]
        
        s1, id = self.try_spawn_hero(s1)
        
        if not s1:
            return [], id

        res.append(s1)
        self.active_actor.append(w)
        if num_roads == 1:
            return res, id
        
        if not self.__has_affecting_landmark(s2, search_distance):
            res.append(s2)
        else:
            return res, id
        
        for i in range(num_roads-1):
            k1 = res[-1].road_id
            k2 = res[-1].lane_id > 0
            if self.adj.__contains__((k1, k2)):
                out_len = len(self.adj[(k1, k2)])
                if out_len == 0:
                    return res, id
                
                _, t2 = self.adj[(k1, k2)][np.random.choice(out_len)]
                if not self.__has_affecting_landmark(t2, search_distance):
                    res.append(t2)
                else:
                    return res, id
            else:
                return res, id

        return res, id
        

    def start_new_game(self, path=[], max_num_roads=5, max_retry=3, tick_once=False):
        if not len(path):
            while max_retry >=0 and not len(path):
                path, id = self.try_spawn_random_trajectory(max_num_roads)
                max_retry -= 1

            if not len(path):
                print('unable to start a new game!')
                return []
        else:
            if len(self.active_actor) < 3:
                _, id = self.try_spawn_hero(path[0])
            else:
                return []

        num_roads = len(path)
        try:
            state = self.mgen.sparse_build_multiple_area_upwards(path, max_reward=1.0, min_reward=0.3,                                                                 zero_padding=max(self.h_cache_size_y, self.h_cache_size_x) + 4,                                                                 retain_road_label=True)
        except:
            traceback.print_exception(*sys.exc_info())
            self._path = path
            raise Exception

        wp = state[-2]
        j = 0
        lt = len(wp)
        while j < lt:
            length = len(wp[j])
            k = 0
            while k < length:
                if not len(wp[j][k]):
                    wp[j].pop(k)
                    length -= 1
                    continue
                k += 1
            if not len(wp[j]):
                wp.pop(j)
                lt -= 1
                continue
            j += 1
        
        origin = state[-1]
        road_segment = state[0].copy()
        state[0][state[0] > 0] = 1
        waypoints = state[-2]
        e_pt = None
        qpoints = []
        stop_road = -1
        
        for road_no in range(len(waypoints)-1, -1, -1):
            for lane_no in range(len(waypoints[road_no])):
                for wp_no in range(len(waypoints[road_no][lane_no])-1, -1, -1): # start from mid if road not connecting road and goto end, else start from end
                    e_pt = np.array([waypoints[road_no][lane_no][wp_no].transform.location.x, waypoints[road_no][lane_no][wp_no].transform.location.y])
                    e_pt = np.around((e_pt - origin) * self.map_ratio).astype(np.int32)
                    qpoints = self.dgen.fix_end_line(state[1], state[0], e_pt, offset=-10*self.window, size=9*self.window)
                    if len(qpoints):
                        stop_road = road_no + 1
                        break
                if len(qpoints):
                    break
            if len(qpoints):
                break
        
        if stop_road == -1:
            self.reset()
            raise Exception('.......ERROR : failed to find end point...........')
        
        target_stop = lil_matrix(state[0].shape, dtype=np.float32)
        self.dgen.color_points_in_quadrilateral(qpoints, target_stop, val=1)
        
        state.append([target_stop, None])
        ref_point = (qpoints[0] + qpoints[1] + qpoints[2] + qpoints[3]) / 4
        hero_transform = self.hero_list[-1].get_transform()
        pos_t = np.array([hero_transform.location.x, hero_transform.location.y])
        pos_t = pos_t - origin
        yaw = hero_transform.rotation.yaw
        angle_t = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
        
        start_state = [state,                      [],                      [],                      [],                      e_pt,                      origin,                      len(self.hero_list) - 1,                      [road_segment, stop_road, False],                      pos_t,                      angle_t,                      0,                      id,                      ref_point,                      [0, 0, 0]]
        
        if tick_once:
            dummy_action = np.zeros((1, 4), dtype=np.float32)
            dummy_action[0, 0] = 6
            dummy_action[0, 1] = 2
            self.step([start_state], dummy_action, override=True)

        return start_state

    def step(self, state, action, override=False):
        start_state = []
        
        for i in range(len(state)):
            start_state.append(state[i][10])
        
        for i in range(len(state)):
            if state[i][10] == 2:
                continue

            if state[i][10] == 1:
                action[i,2] = 1
                continue
            
            if not override and self.is_quit_available(state[i]):
                action[i,2] = 1
            else:
                action[i,2] = 0
        
        sensor_data = None
        for tick in range(config.tick_per_action):
            for i in range(len(state)):
                if state[i][10] == 2:
                    continue
                
                hero = self.hero_list[state[i][6]]
                if action[i,2]: 
                    control = hero.get_control()
                    control.brake = 1.0
                    control.throttle = 0
                    hero.apply_control(control)
                    state[i][10] = 1
                    state[i][13][0] = float(control.steer)
                    state[i][13][1] = 0
                    state[i][13][2] = 1
                else:
                    target_steer = self.steer_map[int(action[i,0])]
                    throttle = 0.4 + np.clip(np.random.randn(), -0.75, 0.75) * 0.1
                    brake = 0
                    steer = state[i][13][0]
                    steer = (1 - config.steering_gain) * steer + config.steering_gain * target_steer
                    control = carla.VehicleControl()
                    control.steer = float(steer)
                    control.throttle = float(throttle)
                    control.brake = float(brake)
                    hero.apply_control(control)
                    state[i][13][0] = float(steer)
                    state[i][13][1] = float(throttle)
                    state[i][13][2] = float(brake)
            
            sensor_data = self.sync_mode.tick(timeout=2.0)
        
        snapshot = sensor_data[0]

        if config.camera_render:
            image_semseg = sensor_data[1]
        else:
            image_semseg = None
        
        col_events = sensor_data[1:]

        reward = []
        for i in range(len(state)):
            r = self.process_step(state[i], col_events[state[i][11]])
            reward.append(r)

        for i in range(len(state)):
            if state[i][10] == 2 and start_state[i] != 2:
                self.sync_mode.remove_sensor_queue(state[i][11])
                self.hero_list[state[i][6]].destroy()
                self.hero_list[state[i][6]] = None
        
        return reward, start_state, image_semseg

    def process_step(self, state, col_event):
        if state[10] == 2:
            return 0
        
        if col_event:
            reward = self.collision_reward
            state[10] = 2
            return reward
        
        hero = self.hero_list[state[6]]
        origin = state[5]
        e_pt = state[4]
        current_transform = hero.get_transform()
        yaw = current_transform.rotation.yaw
        angle = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
        x, y = current_transform.location.x, current_transform.location.y
        pos_t = np.array([x, y])
        s_pt = np.around((pos_t - origin) * self.map_ratio).astype(np.int32)

        if state[10] == 1:
            velocity = hero.get_velocity()
            v_x, v_y = velocity.x, velocity.y
            speed = np.sqrt(v_x * v_x + v_y * v_y)
            half_kernel = 2
            loc_y = s_pt[1]
            loc_x = s_pt[0]
            F = np.array([state[0][1][0][loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean().item(),                          state[0][1][1][loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean().item()])
            d = np.linalg.norm(F)
            
            if d < 1e-3:
                print('out of range with state 1')
                state[10] = 2
                return self.MIN_REWARD
            field_vector = F / d
            
            if speed <= self.MAX_TRY_STOP_SPEED:
                state[10] = 2
                print('stopping.....')
                D = state[0][-1][0][s_pt[1]-half_kernel : s_pt[1]+half_kernel+1, s_pt[0]-half_kernel : s_pt[0]+half_kernel+1].sum()
                cosine = field_vector.dot(angle)
                if cosine < self.cos_max_theta_stop:
                    print('last reward : ', self.MIN_REWARD)
                    return self.MIN_REWARD
                
                if D > 1:
                    reward = self.MAX_REWARD * cosine
                else:
                    ref_point = state[12]
                    goal_distance = np.linalg.norm(ref_point - s_pt) / self.map_ratio
                    reward = max(self.MAX_REWARD - self.reward_drop_rate * (goal_distance ** self.beta), 0) * cosine
                
                print('last reward : ', reward)
                return reward
            else:
                return 0

        through = state[7][-1]

        if through:
            half_kernel = 2
            D = state[0][-1][1][s_pt[1]-half_kernel : s_pt[1]+half_kernel+1, s_pt[0]-half_kernel : s_pt[0]+half_kernel+1].sum()
            if D > 1:
                print('found through target')
                reward = self.MAX_REWARD
                state[10] = 2
                return reward

        velocity = hero.get_velocity()
        acc = hero.get_acceleration()
        v_x, v_y = velocity.x, velocity.y
        a_x, a_y = acc.x, acc.y
        speed = np.sqrt(v_x * v_x + v_y * v_y)
        if self.print_:
            print('vehicle_speed : %f km/h' % (speed * 3.6))
            print('acceleration:   %f m/s^2' % np.sqrt(a_x * a_x + a_y * a_y))

        if not len(state[2]):
            self.__create_cached_map(state, s_pt, pos_t)

        cached_map, offset = state[2]

        if speed < 1e-3:
            longitudinal_acc = a_x * angle[0] + a_y * angle[1]
        else:
            longitudinal_acc = (a_x * v_x + a_y * v_y) / speed

        control = hero.get_control()
        steering = control.steer
        
        half_kernel = 2
        loc_y = s_pt[1] - offset[1]
        loc_x = s_pt[0] - offset[0]
        F = np.array([cached_map[0, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean(),                      cached_map[1, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean()])
        d = np.linalg.norm(F)

        if d < 1e-4:
            state[10] = 2
            return self.boundary_cross_reward
        else:
            F = F / d

        c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)
        if c_y - self.h_dynamic_size_y < 0 or c_x - self.h_dynamic_size_x < 0 or           c_y + self.h_dynamic_size_y + 1 > config.cache_size_y or           c_x + self.h_dynamic_size_x + 1 > config.cache_size_x:
            self.__create_cached_map(state, s_pt, pos_t)
            cached_map, offset = state[2]
            loc_x = s_pt[0] - offset[0]
            loc_y = s_pt[1] - offset[1]
            c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)

        R = np.array([[angle[0], -angle[1]], [angle[1], angle[0]]])
        T = np.array([loc_x - c_x + self.h_dynamic_size_x,                      loc_y - c_y + self.h_dynamic_size_y]).reshape(2, 1)
        points = np.around(R.dot(self.hero_box) + T).astype(np.int32)
        dynamic_map = state[1]
        
        if len(dynamic_map):
            dynamic_map[6+config.embedding_size:] = 0
        
        state[1] = cached_map[:, c_y-self.h_dynamic_size_y : c_y+self.h_dynamic_size_y+1, c_x-self.h_dynamic_size_x : c_x+self.h_dynamic_size_x+1]
        dynamic_map = state[1]
        hero_start = 6 + config.embedding_size
        footprint = 1
        val = torch.Tensor([footprint, v_x / config.v_scale, v_y / config.v_scale, longitudinal_acc / config.a_scale, steering]).reshape(-1, 1)
        dynamic_map[hero_start:, points[1], points[0]] = val
        
        v_pos = []
        p_pos = []
        v_dynamic_objects = []
        p_dynamic_objects = []
        position = self.npc_information['location_queue'][-1]

        for i in range(len(self.am.np_vehicle_objects)):
            pos = position[0][i]
            x, y = round((pos[0] - origin[0]) * self.map_ratio), round((pos[1] - origin[1]) * self.map_ratio)
            if abs(x - c_x - offset[0]) < self.h_dynamic_size_x and abs(y - c_y - offset[1]) < self.h_dynamic_size_y:
                v_pos.append([np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180), x - c_x - offset[0] + self.h_dynamic_size_x,                              y - c_y - offset[1] + self.h_dynamic_size_y])
                v_dynamic_objects.append(i)

        for i in range(len(self.am.np_pedestrian_objects)):
            pos = position[1][i]
            x, y = round((pos[0] - origin[0]) * self.map_ratio), round((pos[1] - origin[1]) * self.map_ratio)
            if abs(x - c_x - offset[0]) < self.h_dynamic_size_x and abs(y - c_y - offset[1]) < self.h_dynamic_size_y:
                p_pos.append([np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180), x - c_x - offset[0] + self.h_dynamic_size_x,                              y - c_y - offset[1] + self.h_dynamic_size_y])
                p_dynamic_objects.append(i)
        
        state[3] = []
        self.__append_dynamic_features(0, state, v_pos, v_dynamic_objects)
        self.__append_dynamic_features(1, state, p_pos, p_dynamic_objects)
        
        if len(state[3][0][0]) and len(state[3][1][0]):
            accum_tensor = torch.cat([state[3][0][0], state[3][1][0]], axis=1)
        elif len(state[3][0][0]):
            accum_tensor = state[3][0][0]
        elif len(state[3][1][0]):
            accum_tensor = state[3][1][0]
        else:
            accum_tensor = []
        
        accum_position = state[3][0][1] + state[3][1][1]
        accum_points = state[3][0][2] + state[3][1][2]
        state[3] = [accum_tensor, accum_position, accum_points]
        
        pos_t = pos_t - origin
        angle_tm1 = state[9]
        pos_tm1 = state[8]
        
        distance = np.linalg.norm(pos_tm1 - pos_t)
        accum = 0
        
        if distance < self.resolution:
            reward = 0
        else:
            for r in np.arange(distance, 0, -self.resolution):
                alpha = r / distance
                gradient = alpha * angle + (1 - alpha) * angle_tm1
                point = alpha * pos_t + (1 - alpha) * pos_tm1
                point = np.around((point * self.map_ratio) - offset).astype(np.int32)
                accum += cached_map[:2, point[1], point[0]].numpy().dot(gradient) / np.linalg.norm(gradient)
            
            reward = accum * self.resolution * self.max_field_reward
            
            if reward < -1e-3:
                state[10] = 2
                print('vehicle moving in opposite direction!')
                return self.MIN_REWARD
        
        if speed < self.SPEED_PENALTY_LOWER_LIMIT:
            reward -= self.low_speed_penalty        
        elif speed > self.SPEED_PENALTY_UPPER_LIMIT:
            reward -= self.high_speed_penalty
        
        reward -= self.time_penalty
        state[9] = angle
        state[8] = pos_t
        acc_total = a_x * a_x + a_y * a_y
        lateral_g = np.sqrt(max(acc_total - longitudinal_acc ** 2, 0)) / 9.81
        if self.print_:
            print('lateral_g : %f' % lateral_g)
        reward += self.g_rw * lateral_g
        
        return reward

    def is_quit_available(self, state, speed_cond=True):
        if state[7][-1]:
            return False

        if speed_cond:
            hero = self.hero_list[state[6]]
            v = hero.get_velocity()
            speed = np.sqrt(v.x * v.x + v.y * v.y)
            if speed > self.MIN_TRY_STOP_SPEED:
                return False
        
        x, y = np.around(state[8] * self.map_ratio).astype(np.int32)
        half_kernel = 6
        D = state[0][-1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum()
        if D >= 1:
            return True
        
        return False
    
    def draw_image(self, array, blend=False):
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        self.display.blit(image_surface, (0, 0))
    
    def render(self, state, x, reward):
        self.clock.tick()
        
        velocity = x[6+config.embedding_size]
        embs = torch.sqrt(torch.pow(x[6], 2) + torch.pow(x[7], 2))
        embs[embs != 0] = 1.0
        dyn_img = torch.stack([(x[2] + x[3]) / 2, embs, velocity], axis=-1)
        image = (dyn_img.detach().cpu().numpy() * 255).astype(np.uint8)
        image = cv2.resize(image, (1080, 1080))
        self.draw_image(image)
        steer, throttle, brake = state[13]
        self.display.blit(
            self.font.render('%f step reward' % reward, True, (255, 255, 255)),
            (8, 10))
        self.display.blit(
            self.font.render('steer=%f, throttle=%f, brake=%f step reward' % (steer, throttle, brake),
            True, (255, 255, 255)), (8, 28))
        pygame.display.flip()
    
    def get_font(self):
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)
    
    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False
    
    def parse_inputs(self, state):
        if self.should_quit():
            return None

        keys = pygame.key.get_pressed()
        action = np.zeros(4, dtype=np.float32)
        if keys[pygame.K_n]:
            action[2] = np.log(0.99)
        else:
            action[2] = np.log(0.0001)

        milliseconds = config.tick_per_action / config.fps
        controls = state[13]

        _steer_cache = controls[0]
        throttle = controls[1]
        brake = controls[2]

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = min(throttle + 0.01, 0.8)
        else:
            throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = min(brake + 0.2, 1)
        else:
            brake = 0

        steer_increment =  milliseconds
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if _steer_cache > 0:
                _steer_cache = 0
            else:
                _steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if _steer_cache < 0:
                _steer_cache = 0
            else:
                _steer_cache += steer_increment
        else:
            _steer_cache = 0.0
        
        steer = round(_steer_cache, 4)
        state[13][0] = float(steer)
        
        action[0] = steer
        action[1] = throttle - brake

        return action
    
    def exit(self):
        self.am.destroy_all_npc()
        self.reset(hard_reset=True, tick_once=False)
        self.remove_db_info()
        self.world.tick()
        time.sleep(1)
        self.sync_mode.exit()

    def reset(self, state=[], hard_reset=False, tick_once=True):
        self.sync_mode.reset(hard_reset=hard_reset)
        for h in self.hero_list:
            if h is not None:
                try:
                    h.destroy()
                except:
                    traceback.print_exception(*sys.exc_info())
        
        self.hero_list = []
        self.active_actor = []
        self.camera_semseg = None
        
        num_actor = len(state)
        if len(state):
            for i in range(num_actor):
                del state[i][0]
                del state[i][0]
                del state[i][0]
                del state[i][0]
        
        if tick_once:
            self.sync_mode.tick(timeout=2.0)
    
    def initialize(self):
        if self.is_initialized:
            return
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager(config.port)
        if config.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
        if config.seed is not None:
            self.traffic_manager.set_random_device_seed(config.seed)
        
        settings = self.world.get_settings()
        self.traffic_manager.set_synchronous_mode(True)
        self.sync_mode = CarlaSyncMode(self.client, self.world, not config.camera_render, fps=config.fps)
        self.sync_mode.enter()
        self.am = ActorManager(self.client, self.world, self.traffic_manager, self.map_ratio)
        self.am.spawn_npc(config.num_vehicle, config.num_pedestrian)
        self.sync_mode.add_main_queue()
        ret = False
        self.sync_mode.callback = self.update_db
        
        while not ret:
            self.sync_mode.tick(timeout=2.0)
            ret = self.sync_mode.ret
        
        self.is_intialized = True 
    
    def initialize_display(self):
        if config.render:
            pygame.init()
            self.display = pygame.display.set_mode(
                            (1080, 1080),
                            pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            self.font = self.get_font()
        else:
            self.display = None
            self.clock = None
            self.font = None
    
    def hard_reset(self):
        self.am.destroy_all_npc()
        self.reset(hard_reset=True, tick_once=False)
        self.remove_db_info()
        self.world.tick()
        
        time.sleep(1)
        self.am.spawn_npc(config.num_vehicle, config.num_pedestrian)
        self.sync_mode.add_main_queue()
        
        ret = False
        self.sync_mode.callback = self.update_db
        while not ret:
            self.sync_mode.tick(timeout=2.0)
            ret = self.sync_mode.ret


# In[13]:


# env.exit()


# In[14]:


env = Environment(10, ds, client, world)


# In[15]:

env.initialize()

# In[54]:
# env.reset()
# In[16]:


# _traffic_manager = env.traffic_manager
# _sync_mode = env.sync_mode
# _am = env.am
# npc_information = env.npc_information
# _hero_list = env.hero_list
# _active_actor = env.active_actor

# In[60]:

# env.traffic_manager = _traffic_manager
# env.sync_mode = _sync_mode
# env.am = _am
# env.npc_information = npc_information
# env.hero_list = _hero_list
# env.active_actor = _active_actor

# In[17]:


# receptor -> 3 * 3 ->


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.dynamics_encoder = nn.GRU(4, config.embedding_size)
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        for i in range(1, len(config.conv_size) - 1):
            self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=config.kernel_size[i-1], padding=1))
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
            
        self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=config.kernel_size[-1]))
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
    
    def __forward_pass(self, x):
        for i in range(len(self.conv) - 1):
            x = self.conv[i](x)
            x = self.leakyRelu(x)
            x = self.pool[i](x)
        
        x = self.conv[-1](x)
        x = self.leakyRelu(x)
        x = self.flatten(x)
        
        return x
    
    def __encode_dynamics(self, features, state, return_device_clone=False):
        objects = state[0]
        position = state[1]
        points = state[2]

        if len(position):
            num_objects = len(position)
            objects = objects.to(device)
            _, hn = self.dynamics_encoder(objects)
            hn = hn.squeeze(0).unsqueeze(-1)
            
            for i in range(num_objects):
                cos_t, sin_t, x, y = position[i]
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                T = np.array([[x], [y]])
                pos = np.around(R.dot(points[i]) + T).astype(np.int32)
                bounds = np.logical_and(np.logical_and(pos[0] >= 0, pos[0] < config.dynamic_size_x), np.logical_and(pos[1] >= 0, pos[1] < config.dynamic_size_y))
                pos = np.unique(pos[:, bounds], axis=1)
                features[6 : 6 + config.embedding_size, pos[1], pos[0]] += hn[i]
        
        if return_device_clone:
            return [objects.detach() if len(objects) else [], position, points]
        else:
            return None
    
    def forward(self, state, return_embeddings=False, return_device_clone=False):
        batch_size = len(state)
        x = torch.stack([state[i][0][1] for i in range(batch_size)], axis=0).to(device)
        
        if return_device_clone:
            z = [[None for _ in range(batch_size)], x.clone()]
        else:
            z = None
        
        for i in range(batch_size):
            ret = self.__encode_dynamics(x[i], state[i][0][0], return_device_clone=return_device_clone)
            if return_device_clone:
                z[0][i] = ret
        
        if return_embeddings:
            y = x.detach().clone()
        else:
            y = None
        
        x = self.__forward_pass(x)
        return x, y, z
    
    def forward_(self, state, clone=False):
        batch_size = len(state[0])
        x = state[1]
        if clone:
            x = x.clone()
        
        for i in range(batch_size):
            self.__encode_dynamics(x[i], state[0][i])
        
        return self.__forward_pass(x)

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        self.masks = []

    def forward(self, x, start=-1, end=-1, use_precomputed_mask=False):
        if self.training:
            if use_precomputed_mask:
                x = x * torch.cat(self.masks[start:end], axis=0) * (1.0 / (1 - self.p))
            else:
                mask = self.binomial.sample(x.size()).to(device)
                x = x * mask * (1.0 / (1 - self.p))
                self.masks.append(mask)
        return x
    
    def clear_mask(self):
        del self.masks[:]
        self.masks = []

class ActorCritic(nn.Module):    
    def __init__(self, backbones):
        super(ActorCritic, self).__init__()
        self.backbones = nn.ModuleList(backbones)
        self.o1_size = len(config.steer_pos)
        self.o2_size = len(config.throttle_pos)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = Dropout(0.3)
        self.mlp_1 = nn.Linear(config.conv_size[-1], 144)
        self.mlp_2 = nn.Linear(144, 128)
        self.mlp_3 = nn.Linear(128, 96)
        self.mlp_4 = nn.Linear(96, 64)
        self.mlp_5 = nn.Linear(64, self.o1_size)
        self.mlp_6 = nn.Linear(96, 48)
        self.mlp_7 = nn.Linear(48, self.o2_size)
        self.lsoft = nn.LogSoftmax(dim=1)
        self.lsig = nn.LogSigmoid()
        self.start = -1
        self.end = -1
        self.is_train = False
        
        self.c_mlp_1 = nn.Linear(config.conv_size[-1], 128)
        self.c_mlp_2 = nn.Linear(128, 64)
        self.c_mlp_3 = nn.Linear(64, 32)
        self.c_mlp_4 = nn.Linear(32, 1)
    
    def forward(self, state, return_value=True):
        val = None
        
        if config.use_shared_networks:
            x_state, _, _ = self.backbones[0](state)
            stats = self.actor_forward(x_state)
            if return_value:
                val = self.critic_forward(x_state)
        else:
            x_state_actor, _, state_clone = self.backbones[0](state, return_device_clone=return_value)
            stats = self.actor_forward(x_state_actor)
            if return_value:
                x_state_critic = self.backbones[1].forward_(state_clone)
                val = self.critic_forward(x_state_critic)
            
        return stats, val
    
    def actor_forward(self, x_state):        
        x = self.mlp_1(x_state)
        x = self.leakyRelu(x)
        x = self.dropout_1(x, start=self.start, end=self.end, use_precomputed_mask=self.is_train)
        
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        
        x = self.mlp_3(x)
        x_mid = self.leakyRelu(x)

        x = self.mlp_4(x_mid)
        x = self.leakyRelu(x)
        x_ret = self.mlp_5(x)
        x_ret = torch.clamp(self.lsoft(x_ret), config.p_min, config.p_max)
        
        return x_ret

    def critic_forward(self, x_state):        
        x = self.c_mlp_1(x_state)
        x = self.leakyRelu(x)
        
        x = self.c_mlp_2(x)
        x = self.leakyRelu(x)
        
        x = self.c_mlp_3(x)
        x = self.leakyRelu(x)

        x_ret = self.c_mlp_4(x)
        
        return x_ret.squeeze(-1)
    
    def sample_from_density_fn(self, log_st, return_entropy=True, deterministic=False):
        if deterministic:
            pi = 0
            action = torch.stack([torch.argmax(log_st, axis=-1), torch.argmax(log_th, axis=-1)], axis=1)
            signal_prob = None
        else:
            m1 = C.Categorical(probs=torch.exp(log_st))
            s1 = m1.sample()
            action = s1.unsqueeze(-1)
            if return_entropy:
                pi = m1.log_prob(s1)
                entropy = m1.entropy()
            else:
                pi = None
                entropy = None
        
        return action, pi, entropy
    
    def log_pi(self, state):
        stats, val = self.forward(state)
        log_st = stats
        
        batch_size = len(state)
        index = torch.stack([state[i][1][:1] for i in range(batch_size)], axis=0)
        log_p1 = torch.gather(log_st, 1, index).squeeze(-1)
        pi = log_p1
        entropy = -((torch.exp(log_st) * log_st).sum(-1)).sum()
        
        return [pi, val, entropy]

# In[18]:


backbone_1 = Backbone()
backbones = [backbone_1]
if not config.use_shared_networks:
    backbone_2 = Backbone()
    backbones.append(backbone_2)


# In[19]:


actor_critic = ActorCritic(backbones)
actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
if use_cuda:
    actor_critic.cuda()

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
    actor_critic_optimizer.load_state_dict(checkpoint['actor_critic_optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['num_trajectories']


def train_memory_optimized_trajectory(env, actor, actor_critic_optimizer, loss_fn, step_offset, writer, normalize=False, batch_size=8, v_coeff=0.5, alpha=0.1, lambda_g=0.95, gamma=0.99, max_steps=600, max_attempt=3):
    state = []
    for i in range(max_attempt):
        state = env.start_new_game(max_num_roads=np.random.choice([3]), tick_once=True)
        if len(state):
            break

    if not len(state):
        return -1, 0, 0

    if state[10] == 2:
        print('game finished before it started')
        env.reset()
        return 0, 0, 0

    for i in range(config.tick_after_start):
        batch_action = np.zeros((1, 3), dtype=np.int64)
        batch_action[0, 0] = 6
        batch_action[0, 1] = 5
        env.step([state], batch_action, override=True)

    if state[10] == 2:
        print('game finished after const ticking')
        env.reset()
        return 0, 0, 0

    actor_critic.train()
    policy_data = []
    rewards = []
    num_steps = 0
    actor_critic.is_train = False
    
    with torch.no_grad():
        for i in range(max_steps):
            if state[10] == 0:
                stats, _ = actor_critic.forward([[[state[3], state[1]]]], return_value=False)
                action, _, _ = actor_critic.sample_from_density_fn(stats, return_entropy=False)
                policy_data.append([[deepcopy(state[3]), state[1].clone()], action.squeeze(0)])
                action = np.array([[*action.squeeze(0).cpu().numpy(), 0, 0]], dtype=np.int64)
                rewards.append(0)
                num_steps += 1
            else:
                action = np.array([[0, 0, 1]], dtype=np.int64)

            reward, start_state, _ = env.step([state], action)
            rewards[-1] += reward[0]
            stop = state[10] == 2
            
            if stop:
                break

    env.reset()
    effective_size = num_steps

    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    steps_per_epoch = int(np.ceil(effective_size / batch_size))
    actor_critic_optimizer.zero_grad()
    gae_lambda = 0
    last_value = 0
    actor_critic.is_train = True
    start = effective_size
    
    for j in range(steps_per_epoch):
        end = start
        start = max(end - batch_size, 0)
        batch_policy_data = policy_data[start:end]
        mini_batch_size = end - start
        actor_critic.start = start
        actor_critic.end = end
        log_pi, values, entropy = actor_critic.log_pi(batch_policy_data)
        
        batch_advantage = [0 for _ in range(mini_batch_size)]
        d_values = values.detach()
        k = mini_batch_size - 1
        
        for i in range(end - 1, start - 1, -1):
            gae_0 = rewards[i] + gamma * last_value - d_values[k]
            gae_lambda = gae_0 + gamma * lambda_g * gae_lambda
            last_value = d_values[k]
            batch_advantage[k] = gae_lambda
            k -= 1
        
        assert k == -1
        batch_advantage = torch.stack(batch_advantage)
        batch_returns = batch_advantage + d_values
        
        actor_loss = -(log_pi * batch_advantage).sum() / effective_size
        value_loss = torch.pow(values - batch_returns, 2).sum() / effective_size
        entropy_loss = -entropy / effective_size
        cumm_loss = actor_loss + v_coeff * value_loss + alpha * entropy_loss
        cumm_loss.backward()

        total_actor_loss += actor_loss.item()
        total_critic_loss += value_loss.item()
        total_entropy -= entropy_loss.item()

    actor_critic_optimizer.step()
    actor_critic.dropout_1.clear_mask()

    total_reward_earned = sum(rewards)
    args = [total_actor_loss, total_critic_loss, total_entropy, (gae_lambda + last_value).item(), total_reward_earned, rewards[-1], num_steps]
    print('a_loss %.3f, c_loss %.3f, entropy %.3f, G_rw %.3f, t_rw %.3f, e_rw %.3f, length %d'          % tuple([round(arg, 3) for arg in args]))
    writer.add_scalar('actor_score', -total_actor_loss, step_offset)
    writer.add_scalar('critic_loss', total_critic_loss, step_offset)
    writer.add_scalar('entropy', total_entropy, step_offset)
    writer.add_scalar('reward_earned', total_reward_earned, step_offset)
    writer.add_scalar('g_reward', (gae_lambda + last_value).item(), step_offset)
    writer.add_scalar('length', num_steps, step_offset)
    
    return 1, total_reward_earned, total_entropy


# In[32]:


def train_trajectory(env, actor, actor_critic_optim, loss_fn, step_offset, writer, normalize=False, v_coeff=0.5, alpha=0.46, lambda_g=0.95, gamma=0.99, max_steps=200, max_attempt=3):
    state = []
    for i in range(max_attempt):
        state = env.start_new_game(max_num_roads=np.random.choice([3]), tick_once=True)
        if len(state):
            break
    
    if not len(state):
        return -1, 0, 0
    
    if state[10] == 2:
        print('game finished before it started')
        env.reset()
        return 0, 0, 0

    for i in range(config.tick_after_start):
        batch_action = np.zeros((1, 3), dtype=np.int64)
        batch_action[0, 0] = 6
        batch_action[0, 1] = 5
        env.step([state], batch_action, override=True)
    
    if state[10] == 2:
        print('game finished after const ticking')
        env.reset()
        return 0, 0, 0
    
    actor_critic.train()
    actor_critic_optim.zero_grad()
    advantages = []
    logp = []
    values = []
    rewards = []
    entropies = []
    num_steps = 0
    actor_critic.is_train = False
    
    for i in range(max_steps):
        if state[10] == 0:
            stats, value = actor_critic.forward([[[state[3], state[1]]]])
            action, pi, entropy = actor_critic.sample_from_density_fn(stats)
            action = np.array([[*action.squeeze(0).cpu().numpy(), 0, 0]], dtype=np.int64)
            values.append(value)
            entropies.append(entropy)
            rewards.append(0)
            num_steps += 1
        else:
            action = np.array([[0, 0, 1]], dtype=np.int64)
            pi = None
        
        with torch.no_grad():
            reward, start_state, _ = env.step([state], action)
            rewards[-1] += reward[0]
            stop = state[10] == 2
        
        logp.append(pi)

        if stop:
            break
    
    env.reset()
    
    advantage = [0 for _ in range(num_steps)]
    gae_lambda = 0
    values = torch.cat(values)
    d_values = values.detach()
    for i in reversed(range(num_steps)):
        gae_0 = rewards[i] + (gamma * d_values[i+1] if i < num_steps - 1 else 0) - d_values[i]
        gae_lambda = gae_0 + gamma * lambda_g * gae_lambda
        advantage[i] = gae_lambda
    advantage = torch.stack(advantage)
    returns = advantage + d_values
    if normalize:
        if num_steps > 1:
            advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-6)
    logp = torch.cat(logp)
    entropies = torch.cat(entropies)
    
    actor_loss = -(logp * advantage).mean()
    critic_loss = loss_fn(values, returns)
    entropy_loss = -entropies.mean()
    cumm_loss = actor_loss + v_coeff * critic_loss + alpha * entropy_loss
    cumm_loss.backward()
    actor_critic_optimizer.step()
    total_reward_earned = sum(rewards)
    args = [actor_loss.item(), critic_loss.item(), -entropy_loss.item(), returns[0].item(), total_reward_earned, rewards[-1], num_steps]
    print('a_loss %.3f, c_loss %.3f, entropy %.3f, G_rw %.3f, t_rw %.3f, e_rw %.3f, length %d'          % tuple([round(arg, 4) for arg in args]))
    writer.add_scalar('actor_score', -actor_loss.item(), step_offset)
    writer.add_scalar('critic_loss', critic_loss.item(), step_offset)
    writer.add_scalar('entropy', -entropy_loss.item(), step_offset)
    writer.add_scalar('reward_earned', total_reward_earned, step_offset)
    writer.add_scalar('g_reward', returns[0].item(), step_offset)
    
    return 1, total_reward_earned, -entropy_loss.item()


def train_networks(writer, env, actor_critic, actor_critic_optimizer, sample_per_epoch=100, failure_patience=15, epoch=100, start_epoch=0, num_trajectories=0, save_prefix=''):
    epoch += start_epoch
    patience = failure_patience
    loss_fn = nn.MSELoss()
    
    if config.memory_optimized:
        train_fn = train_memory_optimized_trajectory
    else:
        train_fn = train_trajectory
    
    for i in range(start_epoch, epoch):
        total_reward = 0
        total_entropy = 0
        
        for j in range(sample_per_epoch):
            status, reward, entropy = train_fn(env, actor_critic, actor_critic_optimizer, loss_fn, num_trajectories, writer)
            if status == 1:
                patience = min(failure_patience, patience + 0.1)
            elif status == 0:
                continue
            else:
                print('ERROR : unable to train trajectory')
                if patience <= 0:
                    raise Exception('unable to train trajectory with actor_list %d' % (len(env.hero_list)))
                patience -= 1
                continue
            
            total_reward += reward
            total_entropy += entropy
            num_trajectories += 1
        
        print('\n..................... completed Epoch : ', str(i), ' .......................\n')
        
        if (i+1) % config.save_every == 0 and config.path_to_save != '':
            torch.save({
                'epoch': i,
                'num_trajectories' : num_trajectories,
                'actor_critic_state_dict': actor_critic.state_dict(),
                'actor_critic_optimizer_state_dict': actor_critic_optimizer.state_dict(),
                'average_reward' : total_reward / sample_per_epoch,
                'average_entropy' : entropy / sample_per_epoch,
                }, config.path_to_save + 'checkpoint_' + save_prefix + str(i) + '.pth')


logger = SummaryWriter(sys.argv[1])

try:
    start_epoch=0
    num_trajectories=0
    if len(sys.argv) > 2:
        start_epoch, num_trajectories = load_model(sys.argv[2])
        print('starting from ...', start_epoch)
    train_networks(logger, env, actor_critic, actor_critic_optimizer, start_epoch=start_epoch, num_trajectories=num_trajectories)
except:
    traceback.print_exception(*sys.exc_info())
finally:
    print('cleaning')
    logger.close()
    env.exit()



# logger = SummaryWriter('runs/oct3')
# start_epoch=0
# num_trajectories=0
# train_networks(logger, env, actor_critic, actor_critic_optimizer, start_epoch=start_epoch, num_trajectories=num_trajectories)

