#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cd project_files


# In[2]:


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
import torch.distributions.normal as N
import torch.distributions.categorical as C
from torch.utils.tensorboard import SummaryWriter

import Dataset


# In[3]:


use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')


# In[4]:


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
        self.boundary_points = [[], []]
        self.map_ratio = map_ratio
    
    def __fill_measurement_grid(self, idx, measurement, resolution=0.75):
        for i in range(len(measurement)):
            self.boundary_points[idx].append([[measurement[i][0][0] * self.map_ratio, measurement[i][-1][0] * self.map_ratio],                                              [measurement[i][0][1] * self.map_ratio, measurement[i][-1][1] * self.map_ratio]])
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


# In[5]:


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


# In[14]:


class config:
    num_action = 1
    throttle_pos = [-0.6, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    steer_pos = [-0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
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
    continious_range = 0.85
    buffer_size = int(1e6)
    neta = 0.996
    p_min = -40
    sigma_min = -40
    sigma_max = 4
    min_alpha = 0.1
    max_alpha = 1.0
    save_dict_cycle = 5
    alpha = 0.2
    target_entropy_steering = 0.5
    batch_size = 128
    random_explore = 4000
    polyak = 0.995
    gamma = 0.99
    min_buffer_size = 4000
    step_per_lr_update = 4
    update_every = 156
    save_every = 1
    lr = 0.0003
    fps = 15
    seed = None
    sim_port = 4000
    port = 8200
    hybrid = False 
    num_vehicle = 40
    num_pedestrian = 0
    # expert_directory = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/collected_trajectories/'
    # grid_dir = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/cache/image.png'
    # path_to_save = '/home/harsh/project_files/weights/steer/'
    expert_directory = '/home/ubuntu/project_files/collected_trajectories/'
    grid_dir = '/home/ubuntu/project_files/cache/image.png'
    path_to_save = '/home/ubuntu/project_files/weights/steer/'


#when v < 2.5 m/s and no vehicle in range
# throttle range becomes ~ [0.2 < 0.4 > 0.6]
#else
# throttle range becomes ~ [-0.5, 1]
#steer range is always [-0.85, 0.85]


# In[21]:


class Environment(object):    
    def __init__(self, num_hero, dataset, client, world, max_rw_distance=20):
        self.dgen = dataset
        self.mgen = self.dgen.mgen
        self.window = dataset.window
        self.top = dataset.top
        self.map_ratio = dataset.map_ratio
        self.mgen = dataset.mgen
        self.center = dataset.center
        self.radius = dataset.radius
        self.hero_box = self.dgen.box[0][0]
        self.speed_on_stop_th = dataset.speed_on_stop_th
        self.min_safe_distance = 5
        self.max_heros = num_hero
        self.hero_list = []
        self.active_actor = []
        self.min_field_reward = 0.03
        self.beta = 0.5
        self.random_theta_variation = 0#12
        self.cos_max_theta_stop = 1 / np.sqrt(2)
        self.resolution = 0.09
        self.MAX_TRY_STOP_SPEED = 1
        self.MIN_TRY_STOP_SPEED = 5
        self.dist = self.compute_shortest_distance_matrix()
        self.SPEED_PENALTY_UPPER_LIMIT = 60 / 3.6
        self.SPEED_PENALTY_LOWER_LIMIT = 5 / 3.6
        self.max_ray_distance = 20 * self.map_ratio
        self.min_section_speed = 1
        self.const_dist_scale = 25
        self.stop_distance = 80
        self.start_bonus_time = 36
        self.const_timescale = self.const_dist_scale * (config.fps * 1.0 / (config.tick_per_action * self.min_section_speed))
        self.reward_factor = 4
        self.g_rw = 0 / self.reward_factor
        self.max_field_reward = 0.6 / self.reward_factor
        self.boundary_cross_reward = -8 / self.reward_factor
        self.max_time_penalty = 0.08 / self.reward_factor
        self.high_speed_penalty = 0 / self.reward_factor
        self.MAX_REWARD = 16 / self.reward_factor
        self.MIN_REWARD = -10 / self.reward_factor
        self.collision_reward = -8 / self.reward_factor
        self.cutoff_speed = 0
        self.reward_above = 0
        self.positive_turnout_speed = 15
        self.reward_drop_rate = self.MAX_REWARD / (max_rw_distance ** self.beta)
        self.ray_angles = np.concatenate([np.arange(-90, 91, 10), np.array([120, 150, 170, 180, 190, 210, 240])])
        self.add_rays()
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

    def __has_affecting_landmark(self, waypoint, search_distance):
        lmark = waypoint.get_landmarks_of_type(search_distance, '1000001')
        
        for i in range(len(lmark)):
            wp = lmark[i].waypoint
            if wp.road_id == waypoint.road_id and wp.lane_id * waypoint.lane_id > 0:
                return True

        return False
    
    def add_rays(self, step=0.5):
        self.track_points = []
        max_len = 0
        
        for angle in self.ray_angles:
            direction = np.array([np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)])
            pos = np.zeros(2)
            distance = 0
            self.track_points.append([])
            while distance < self.max_ray_distance:
                self.track_points[-1].append(pos)
                pos = pos + step * direction
                distance += step
            self.track_points[-1] = np.around(self.track_points[-1]).astype(np.int32)
            index = np.unique(self.track_points[-1], axis=0, return_index=True)[1]
            self.track_points[-1] = [self.track_points[-1][i] for i in sorted(index)]
            max_len = max(max_len, len(self.track_points[-1]))
        
        for i in range(len(self.track_points)):
            diff = max_len - len(self.track_points[i])
            for _ in range(diff):
                self.track_points[i].insert(0, np.array([0, 0]))
        
        self.track_points = np.array(self.track_points)
        self.track_points = self.track_points.transpose(0, 2, 1)
        self.points_per_ray = max_len
    
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
    
    def color_tiles(self, state, num_tiles):
        roads = state[-2]
        origin = state[-1]
        target_stop = np.zeros(state[0].shape, dtype=np.float32)
        qpoints = None
        stop_road = -1
        counter = 0
        last_points = None
        
        for i in reversed(range(len(roads))):
            left = roads[i][-1]
            right = roads[i][0]
            stop_road = i+1
            
            if len(left) != len(right):
                print('Warning : extra points detected l=%d, r=%d' % (len(left), len(right)))
            length = min(len(left), len(right))
            
            for j in range(1, length):
                if counter >= num_tiles:
                    return stop_road, qpoints, target_stop, last_points
                diff_l1 = -left[-j].lane_width / 2
                diff_r1 = right[-j].lane_width / 2
                diff_l2 = -left[-j-1].lane_width / 2
                diff_r2 = right[-j-1].lane_width / 2
                g_l1 = left[-j].transform
                g_r1 = right[-j].transform
                g_l2 = left[-j-1].transform
                g_r2 = right[-j-1].transform
                rv_l1 = g_l1.rotation.get_right_vector()
                rv_l2 = g_l2.rotation.get_right_vector()
                rv_r1 = g_r1.rotation.get_right_vector()
                rv_r2 = g_r2.rotation.get_right_vector()
                rv_l1 = np.array([rv_l1.x, rv_l1.y])
                rv_r1 = np.array([rv_r1.x, rv_r1.y])
                rv_l2 = np.array([rv_l2.x, rv_l2.y])
                rv_r2 = np.array([rv_r2.x, rv_r2.y])
                rv_l1 /= (np.linalg.norm(rv_l1) + 1e-6)
                rv_l2 /= (np.linalg.norm(rv_l2) + 1e-6)
                rv_r1 /= (np.linalg.norm(rv_r1) + 1e-6)
                rv_r2 /= (np.linalg.norm(rv_r2) + 1e-6)
                x_l1 = self.map_ratio * (diff_l1 * rv_l1 + np.array([g_l1.location.x, g_l1.location.y]) - origin)
                x_r1 = self.map_ratio * (diff_r1 * rv_r1 + np.array([g_r1.location.x, g_r1.location.y]) - origin)
                x_l2 = self.map_ratio * (diff_l2 * rv_l2 + np.array([g_l2.location.x, g_l2.location.y]) - origin)
                x_r2 = self.map_ratio * (diff_r2 * rv_r2 + np.array([g_r2.location.x, g_r2.location.y]) - origin) 
                
                try:
                    self.dgen.color_points_in_quadrilateral([x_l1, x_r1, x_r2, x_l2], target_stop, val=1)
                    qpoints = [x_l1, x_r1, x_r2, x_l2]
                    last_points = (i, length-j)
                except:
                    print('Warning unable to color quadilateral', qpoints)
                    traceback.print_exception(*sys.exc_info())
                
                counter += 1
            
            counter += 1
        
        return stop_road, qpoints, target_stop, last_points
    
    def partition_into_section(self, state, lp, spawn_wp, max_section_time=30, ref_lane=0, const_scale=True):
        roads = state[-3]
        distance = 0
        last_road = lp[0] + 1
        reference_points = []
        start_loc = spawn_wp.transform.location
        min_dist = np.inf
        ref_loc = -1
        
        for i in reversed(range(last_road)):
            if i == last_road - 1:
                size = lp[1]
            else:
                size = len(roads[i][ref_lane]) - 1
            
            for j in reversed(range(size)):
                loc_1 = roads[i][ref_lane][j+1].transform.location
                loc_2 = roads[i][ref_lane][j].transform.location
                delta = np.sqrt((loc_1.x - loc_2.x) ** 2 + (loc_1.y - loc_2.y) ** 2)
                distance += delta
                reference_points.append([distance, np.array([loc_2.x, loc_2.y])])
                if spawn_wp.road_id == roads[i][ref_lane][j].road_id:
                    start_dist = np.sqrt((start_loc.x - loc_2.x) ** 2 + (start_loc.y - loc_2.y) ** 2)
                    if start_dist < min_dist:
                        min_dist = start_dist
                        ref_loc = len(reference_points) - 1
            
            if i != 0:
                loc_1 = roads[i][ref_lane][0].transform.location
                loc_2 = roads[i-1][ref_lane][-1].transform.location
                delta = np.sqrt((loc_1.x - loc_2.x) ** 2 + (loc_1.y - loc_2.y) ** 2)
                distance += delta
                reference_points.append([distance, np.array([loc_2.x, loc_2.y])])
                if spawn_wp.road_id == roads[i-1][ref_lane][-1].road_id:
                    start_dist = np.sqrt((start_loc.x - loc_2.x) ** 2 + (start_loc.y - loc_2.y) ** 2)
                    if start_dist < min_dist:
                        min_dist = start_dist
                        ref_loc = len(reference_points) - 1
        
        if len(reference_points) == 0:
            print('no points to follow!')
            return None, 0
        
        reference_points.reverse()
        
        if ref_loc == -1:
            print('unable to find close point', reference_points)
            return None, 0
        
        ref_loc = len(reference_points) - ref_loc - 1
        section_status = np.zeros(len(reference_points), dtype=np.float32)
        section_id = np.zeros(len(reference_points), dtype=np.int32)
        total_distance = reference_points[ref_loc][0]
        partition_distance = self.min_section_speed * max_section_time
        distance = 0
        j = 0
        ranges = []
        i = ref_loc
        ref_len = len(reference_points)
        ranges.append([i, ref_len])
        normalizer = []
        
        while i < ref_len - 1:
            delta = -(reference_points[i+1][0] - reference_points[i][0])
            
            if delta >= partition_distance:
                print('delta too large!!', delta, i, section_status, reference_points, section_id)
                return None, 0
            
            if distance + delta >= partition_distance:  
                normalizer.append(distance)
                distance = 0
                section_status[i] = 0
                section_id[i] = j+1
                ranges[-1][1] = i
                ranges.append([i, ref_len])
                j += 1
            
            distance += delta
            section_status[i+1] = distance
            section_id[i+1] = j
            
            i += 1
        
        normalizer.append(distance)
        
        if 2 * normalizer[-1] < partition_distance and len(normalizer) > 1:
            section_status[ranges[-1][0]:ranges[-1][1]] += normalizer[-2]
            assert section_id[ranges[-2][0]] == len(ranges) - 2, print(section_id, len(ranges) - 2)
            section_id[ranges[-1][0]:ranges[-1][1]] = len(ranges) - 2
            normalizer[-2] += normalizer[-1]
            normalizer.pop()
            ranges[-2][1] = ref_len
            ranges.pop()
        
        if const_scale:
            for j in range(len(normalizer)):
                section_status[ranges[j][0]:ranges[j][1]] = normalizer[j] - section_status[ranges[j][0]:ranges[j][1]]
            section_status /= self.const_dist_scale
        else:
            for j in range(len(normalizer)):
                section_status[ranges[j][0]:ranges[j][1]] /= normalizer[j]
        
        normalizer = np.array(normalizer)
        timestep_per_section = np.ceil(normalizer * (config.fps * 1.0 / (config.tick_per_action * self.min_section_speed))).astype(np.int32)
        timestep_per_section[0] += self.start_bonus_time
        return [reference_points, section_status, section_id, timestep_per_section], ref_loc
    
    def start_new_game(self, path=[], max_num_roads=3, max_retry=3, tick_once=False):
        if not len(path):
            while max_retry >= 0 and not len(path):
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

        spawn_wp = path[1] 
        num_roads = len(path)
        try:
            state = self.mgen.var_build_multiple_area_upwards(path, max_reward=1.0, min_reward=0.3, zero_padding=16, retain_road_label=False, use_sparse_mat=False)
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
        waypoints = state[-2]
        e_pt = None
        stop_road, qpoints, target_stop, lp = self.color_tiles(state, 7)
        
        if qpoints is None:
            self.sync_mode.remove_sensor_queue(id)
            self.active_actor.pop()
            self.hero_list[-1].destroy()
            self.hero_list.pop()
            self.sync_mode.tick(timeout=2)
            print('.......ERROR : failed to find end point...........')
            return []
        
        state.append([target_stop, None])
        ref_point = (qpoints[0] + qpoints[1] + qpoints[2] + qpoints[3]) / 4
        hero_transform = self.hero_list[-1].get_transform()
        pos_t = np.array([hero_transform.location.x, hero_transform.location.y])
        pos_t = pos_t - origin
        s_pt = np.around(pos_t * self.map_ratio).astype(np.int32)
        
        if state[-1][0][s_pt[1] - 6 : s_pt[1] + 7, s_pt[0] - 6 : s_pt[0] + 7].sum() > 0:
            print('starting point too close to the finish...')
            self.sync_mode.remove_sensor_queue(id)
            self.active_actor.pop()
            self.hero_list[-1].destroy()
            self.hero_list.pop()
            self.sync_mode.tick(timeout=2)
            return []
        
        partitions, ref_loc = self.partition_into_section(state, lp, spawn_wp)
        if partitions is None:
            print('partition is None !!')
            self.sync_mode.remove_sensor_queue(id)
            self.active_actor.pop()
            self.hero_list[-1].destroy()
            self.hero_list.pop()
            self.sync_mode.tick(timeout=2)
            return []
        
        yaw = hero_transform.rotation.yaw
        angle_t = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
        start_state = [state, None, None, [], e_pt, origin, len(self.hero_list) - 1, 0, pos_t, angle_t, 0,  id, ref_point, [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [0, ref_loc, partitions[2][ref_loc]], partitions]
        
        if tick_once:
            dummy_action = np.zeros((1, 3), dtype=np.float32)
            dummy_action[0, 0] = 0
            dummy_action[0, 1] = 0
            self.step([start_state], dummy_action, override=True)

        return start_state

    def step(self, state, action, override=False, trace=False):
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
                    state[i][13][0].append(float(control.steer))
                    state[i][13][0].pop(0)
                    state[i][13][1].append(-1)
                    state[i][13][1].pop(0)
                else:
                    target_steer = action[i,0]
                    mix_action = action[i,1]                    
                    steer = state[i][13][0][-1]
                    steer = (1 - config.steering_gain) * steer + config.steering_gain * target_steer
                    throttle = max(mix_action, 0)
                    brake = -min(mix_action, 0)
                    control = carla.VehicleControl()
                    control.steer = float(steer)
                    control.throttle = float(throttle)
                    control.brake = float(brake)
                    hero.apply_control(control)
                    state[i][13][0].append(float(steer))
                    state[i][13][0].pop(0)
                    state[i][13][1].append(mix_action)
                    state[i][13][1].pop(0)
            sensor_data = self.sync_mode.tick(timeout=2.0)
        
        snapshot = sensor_data[0]

        if config.camera_render:
            image_semseg = sensor_data[1]
        else:
            image_semseg = None
        
        col_events = sensor_data[1:]

        reward = []
        for i in range(len(state)):
            r = self.process_step(state[i], col_events[state[i][11]], trace)
            reward.append(r)
        
        for i in range(len(state)):
            if state[i][10] == 2 and start_state[i] != 2:
                self.sync_mode.remove_sensor_queue(state[i][11])
                self.hero_list[state[i][6]].destroy()
                self.hero_list[state[i][6]] = None
        
        return reward, start_state, image_semseg
    
    def update_closest_neighbor(self, state, pos_t):
        ref_pts = state[15][0]
        current_idx = state[14][1]
        new_ref = -1
        minimum = np.inf
        
        for i in range(current_idx, min(current_idx + 6, len(ref_pts))):
            d = np.linalg.norm(ref_pts[i][1] - pos_t)
            if d < minimum:
                minimum = d
                new_ref = i
        
        if new_ref != -1:
            state[14][1] = new_ref

    def shoot_rays(self, state, current_transform, trace=False):
        relative_angle = self.ray_angles
        hero = self.hero_list[state[6]]
        yaw = current_transform.rotation.yaw
        absolute_angle = relative_angle + yaw
        origin = state[5]
        
        x, y = current_transform.location.x, current_transform.location.y
        pos_t = np.array([x, y])
        s_pt = (pos_t - origin) * self.map_ratio
        angle = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
        alpha = 1
        half_kernel = 1
        
        R = np.array([[angle[0], -angle[1]], [angle[1], angle[0]]])
        T = s_pt.reshape(-1, 1)
        pos = np.around(R @ self.track_points + T).astype(np.int32)
        
        img = state[0][0]
        size_y, size_x = img.shape
        flat_pos = pos.transpose(1, 0, 2)
        bounds = np.logical_and(np.logical_and(flat_pos[0] >= 0, flat_pos[0] < size_x), np.logical_and(flat_pos[1] >= 0, flat_pos[1] < size_y))
        not_bounds = np.logical_not(bounds)
        flat_pos[1][not_bounds] = 0
        flat_pos[0][not_bounds] = 0
        points = img[flat_pos[1], flat_pos[0]]
        trace_map = state[2]
        control = hero.get_control()
        steering = control.steer
        
        if trace:
            if trace_map is None:
                new_img = img.copy()
                sections = state[15][2]
                new_img[new_img > 0] = 0.8
                for i in range(1, len(sections)):
                    if sections[i-1] != sections[i] or i == len(sections) - 1:
                        start = np.around((state[15][0][i][1] - origin) * self.map_ratio).astype(np.int32)
                        qp = self.dgen.fix_end_line(state[0][1], img, start, offset=-self.window, size=2*self.window)
                        if not len(qp):
                            continue
                        try:
                            self.dgen.color_points_in_quadrilateral(qp, new_img, val=1)
                        except:
                            pass
                trace_map = np.concatenate([new_img[np.newaxis,:,:], np.zeros((2, size_y, size_x), dtype=np.float32)], axis=0)
                target_array = state[0][-1][0]
                trace_map[0] = (trace_map[0] + target_array) / 2
                state[2] = trace_map
            
            cur = np.around(R.dot(self.hero_box) + T).astype(np.int32)
            trace_map[1:] *= 0
            trace_map[2, cur[1], cur[0]] = 1
            trace_map[1, flat_pos[1][bounds], flat_pos[0][bounds]] = 0.5

        max_squared_distance = self.max_ray_distance ** 2
        static_features = state[1]
        dynamic_features = {}
        if static_features is None:
            static_features = np.zeros(config.num_rays + 16, dtype=np.float32)
        else:
            static_features[:] *= 0
        
        self_velocity = hero.get_velocity()
        self_acc = hero.get_acceleration()
        self_omega = hero.get_angular_velocity().z

        def get_intersection_distance(boundary_points, line_angle, local_point):
            cos_t, sin_t =  np.cos(line_angle), np.sin(line_angle)
            h_width = abs(boundary_points[0][0])
            h_height = abs(boundary_points[1][0])
            
            min_distance = np.inf
            point = [0, 0]

            if abs(sin_t) > 1e-3:
                inv_slope = cos_t / sin_t
                for y in boundary_points[1]:
                    x = local_point[0,0] + inv_slope * (y - local_point[1,0])
                    if abs(x) <= h_width:
                        distance = np.sqrt((x - local_point[0,0]) ** 2 + (y - local_point[1,0]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            point = [x,y]

            if abs(cos_t) > 1e-3:
                slope = sin_t / cos_t
                for x in boundary_points[0]:
                    y = local_point[1,0] + slope * (x - local_point[0,0])
                    if abs(y) <= h_height:
                        distance = np.sqrt((x - local_point[0,0]) ** 2 + (y - local_point[1,0]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            point = [x,y]
            
            if (point[0] - local_point[0,0]) * cos_t + (point[1] - local_point[1,0]) * sin_t >= 0:
                return min_distance
            else:
                return np.inf

            return min_distance
        
        delta = 100
        for i in range(len(self.am.np_pedestrian_objects)):
            transform = self.am.np_pedestrian_objects[i].get_transform()
            x, y = transform.location.x, transform.location.y
            vehicle_yaw = transform.rotation.yaw
            pos = [x, y, vehicle_yaw]
            x, y = (pos[0] - origin[0]) * self.map_ratio, (pos[1] - origin[1]) * self.map_ratio
            if trace:
                _x, _y = np.around(x).astype(np.int32), np.around(y).astype(np.int32)
                if _x >= 16 and _y >= 16 and _x < size_x - 16 and _y < size_y - 16:
                    cos_t, sin_t = np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    T = np.array([[x], [y]])
                    box = (R.dot(self.am.box[0][i]) + T).astype(np.int32)
                    trace_map[2, box[1], box[0]] = 1.0
            if (x - s_pt[0]) ** 2 + (y - s_pt[1]) ** 2 < max_squared_distance + delta:
                cos_t, sin_t = np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                T = np.array([[x], [y]])
                local_point = R.T.dot(s_pt.reshape(-1,1) - T)
                for j in range(config.num_rays):
                    line_angle = (absolute_angle[j] - pos[2]) * np.pi / 180
                    distance = get_intersection_distance(self.am.boundary_points[1][i], line_angle, local_point)
                    if distance != np.inf:
                        if dynamic_features.__contains__(j):
                            features = dynamic_features[j]
                            if features[1] < distance:
                                continue
                        else:
                            features = np.zeros(16, dtype=np.float32)
                            dynamic_features[j] = features
                        features[0] = 1
                        features[1] = distance
                        features[2] = i
                        features[12], features[13] = np.cos((pos[2] - yaw) * np.pi / 180), np.sin((pos[2] - yaw) * np.pi / 180)
                        
        for i in range(len(self.am.np_vehicle_objects)):
            transform = self.am.np_vehicle_objects[i].get_transform()
            x, y = transform.location.x, transform.location.y
            vehicle_yaw = transform.rotation.yaw
            pos = [x, y, vehicle_yaw]
            x, y = (pos[0] - origin[0]) * self.map_ratio, (pos[1] - origin[1]) * self.map_ratio
            if trace:
                _x, _y = np.around(x).astype(np.int32), np.around(y).astype(np.int32)
                if _x >= 16 and _y >= 16 and _x < size_x - 16 and _y < size_y - 16:
                    cos_t, sin_t = np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    T = np.array([[x], [y]])
                    box = (R.dot(self.am.box[0][i]) + T).astype(np.int32)
                    trace_map[2, box[1], box[0]] = 1.0
            if (x - s_pt[0]) ** 2 + (y - s_pt[1]) ** 2 < max_squared_distance + delta:
                cos_t, sin_t = np.cos(pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                T = np.array([[x], [y]])
                local_point = R.T.dot(s_pt.reshape(-1,1) - T)
                for j in range(config.num_rays):
                    line_angle = (absolute_angle[j] - pos[2]) * np.pi / 180
                    distance = get_intersection_distance(self.am.boundary_points[0][i], line_angle, local_point)
                    if distance != np.inf:
                        if dynamic_features.__contains__(j):
                            features = dynamic_features[j]
                            if features[1] < distance:
                                continue
                        else:
                            features = np.zeros(16, dtype=np.float32)
                            dynamic_features[j] = features
                        features[0] = 0
                        features[1] = distance
                        features[2] = i
                        features[12], features[13] = np.cos((pos[2] - yaw) * np.pi / 180), np.sin((pos[2] - yaw) * np.pi / 180)
        
        R = np.array([[angle[0], -angle[1]], [angle[1], angle[0]]])
        loc_x, loc_y = np.around(s_pt).astype(np.int32)
        ret = False
        
        for j in range(config.num_rays):
            intersection = None
            for i in range(self.points_per_ray):
                if not bounds[j, i]:
                    break
                
                x = flat_pos[0][j][i]
                y = flat_pos[1][j][i]
                if points[j, i] == 0:
                    if img[y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum() == 0:
                        intersection = np.array([x, y])
                        break
                if state[0][-1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum() > 0:
                    break
                if trace:
                    trace_map[1, y, x] = 1.0
                    trace_map[2, y, x] = 1.0

            if intersection is not None:
                distance = np.linalg.norm(intersection - s_pt)
            else:
                distance = self.max_ray_distance

            if dynamic_features.__contains__(j):
                features = dynamic_features[j]
                index = int(features[2])
                if features[0] == 1:
                    vehicle_speed = self.am.np_pedestrian_objects[index].get_velocity()
                    vehicle_acc = self.am.np_pedestrian_objects[index].get_acceleration()
                    vehicle_omega = self.am.np_pedestrian_objects[index].get_angular_velocity().z
                    features[14] = 1
                    features[15] = 0
                else:
                    vehicle_speed = self.am.np_vehicle_objects[index].get_velocity()
                    vehicle_acc = self.am.np_vehicle_objects[index].get_acceleration()
                    vehicle_omega = self.am.np_vehicle_objects[index].get_angular_velocity().z
                    features[14] = 0
                    features[15] = 1
                
                if j >= 2 and j <= 16:
                    if features[1] < self.max_ray_distance:
                        ret = True

                features[0] = steering
                features[1] /= self.max_ray_distance
                features[1] = 1 - features[1]
                features[2], features[3] = (vehicle_acc.x - self_acc.x) / config.a_scale, (vehicle_acc.y - self_acc.y) / config.a_scale
                features[4], features[5] = (vehicle_speed.x - self_velocity.x) / config.v_scale, (vehicle_speed.y - self_velocity.y) / config.v_scale
                features[6], features[7] = vehicle_speed.x / config.v_scale, vehicle_speed.y / config.v_scale
                features[8], features[9] = vehicle_acc.x / config.a_scale, vehicle_acc.y / config.a_scale
                features[10], features[11] = vehicle_omega / config.w_scale,  (vehicle_omega - self_omega) / config.w_scale
                features[2:10] = features[2:10].reshape(4, 2).dot(R).reshape(-1)

            static_features[j] = distance / self.max_ray_distance
            static_features[j] = 1 - static_features[j]
        
        potential = (state[0][1][:, loc_y, loc_x]).copy()
        d = np.linalg.norm(potential)
        if d > 1e-3:
            potential /= d
        static_features[config.num_rays], static_features[config.num_rays + 1] = self_velocity.x / config.v_scale, self_velocity.y / config.v_scale
        static_features[config.num_rays + 2], static_features[config.num_rays + 3] = self_acc.x / config.a_scale, self_acc.y / config.a_scale
        static_features[config.num_rays + 4], static_features[config.num_rays + 5] = potential[0], potential[1]
        static_features[config.num_rays + 6] = self_omega / config.w_scale
        static_features[config.num_rays + 7] = potential.dot(angle) * self.max_field_reward * d
        static_features[config.num_rays:config.num_rays + 6] = static_features[config.num_rays:config.num_rays + 6].reshape(3, 2).dot(R).reshape(-1)
        index = []
        dyn_features = []
        
        for key in dynamic_features.keys():
            index.append(key + 1)
            dyn_features.append(dynamic_features[key])        
        
        if len(index):
            container = list(zip(index, dyn_features))
            np.random.shuffle(container)
            index, dyn_features = list(zip(*container))
        
        index = list(index)
        dyn_features = list(dyn_features)
        index.append(config.num_rays+2)
        index.insert(0, config.num_rays+1)
        dyn_features.append(np.zeros(16, dtype=np.float32))
        dyn_features.insert(0, np.zeros(16, dtype=np.float32))
        
        state[3] = [torch.LongTensor(index), torch.FloatTensor(dyn_features)]
        state[1] = static_features
        
        return ret

    def process_step(self, state, col_event, trace=False):
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
            F = np.array([state[0][1][0][loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean().item(), state[0][1][1][loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean().item()])
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

        velocity = hero.get_velocity()
        acc = hero.get_acceleration()
        v_x, v_y = velocity.x, velocity.y
        a_x, a_y = acc.x, acc.y
        speed = np.sqrt(v_x * v_x + v_y * v_y)
        if self.print_:
            print('vehicle_speed : %f km/h' % (speed * 3.6))
            print('acceleration:   %f m/s^2' % np.sqrt(a_x * a_x + a_y * a_y))
        
        if speed < 1e-3:
            longitudinal_acc = a_x * angle[0] + a_y * angle[1]
        else:
            longitudinal_acc = (a_x * v_x + a_y * v_y) / speed
        
        half_kernel = 2
        loc_y = s_pt[1]
        loc_x = s_pt[0]
        d = state[0][0][loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].sum()
        
        if d < 1e-4:
            state[10] = 2
            return self.boundary_cross_reward
        
        vehicle_inrange = self.shoot_rays(state, current_transform, trace)
        self.update_closest_neighbor(state, pos_t)
        reference_points = state[15][0]
        ref_index = state[14][1]
        features = state[1]
        features[config.num_rays + 15] = speed / config.v_scale
        features[config.num_rays + 8] = reference_points[ref_index][0] / self.stop_distance
        features[config.num_rays + 9] = state[13][0][1]
        features[config.num_rays + 10] = state[13][1][1]
        features[config.num_rays + 11] = state[13][0][3]
        features[config.num_rays + 12] = state[13][1][3]
        features[config.num_rays + 13] = state[13][0][5]
        features[config.num_rays + 14] = state[13][1][5]
        
        if speed > self.cutoff_speed or vehicle_inrange:
            state[7] = 1
        else:
            state[7] = 0
        
        pos_t = pos_t - origin
        angle_tm1 = state[9]
        pos_tm1 = state[8]         
        
        distance = np.linalg.norm(pos_tm1 - pos_t)
        accum = 0
        reward = 0
        
        if distance < self.resolution:
            reward = 0
        elif speed > self.reward_above:
            for r in np.arange(distance, 0, -self.resolution):
                alpha = r / distance
                gradient = alpha * angle + (1 - alpha) * angle_tm1
                point = alpha * pos_t + (1 - alpha) * pos_tm1
                point = np.around(point * self.map_ratio).astype(np.int32)
                accum += state[0][1][:2, point[1], point[0]].dot(gradient) / np.linalg.norm(gradient)
            
            reward = accum * self.resolution * self.max_field_reward
            
            if reward < -1e-3:
                state[10] = 2
                print('vehicle moving in opposite direction!')
                return self.MIN_REWARD
        
        if speed > self.SPEED_PENALTY_UPPER_LIMIT:
            reward -= self.high_speed_penalty
        
        # coeff = self.max_time_penalty * 3.6 / self.positive_turnout_speed
        # time_penalty = max(self.max_time_penalty - coeff * speed, -0.1)
        # reward -= time_penalty
        state[9] = angle
        state[8] = pos_t
        acc_total = a_x * a_x + a_y * a_y
        lateral_g = np.sqrt(max(acc_total - longitudinal_acc ** 2, 0)) / 9.81
        if self.print_:
            print('lateral_g : %f' % lateral_g)
        reward += self.g_rw * lateral_g
        
        return reward

    def is_quit_available(self, state, speed_cond=False):
        if speed_cond:
            hero = self.hero_list[state[6]]
            v = hero.get_velocity()
            speed = np.sqrt(v.x * v.x + v.y * v.y)
            if speed > self.MIN_TRY_STOP_SPEED:
                return False
        
        x, y = np.around(state[8] * self.map_ratio).astype(np.int32)
        half_kernel = 4
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
    
    def render(self, state, reward, evaluation, pdist):
        self.clock.tick()
        image = state[2].transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (1080, 1080))
        self.draw_image(image)
        
        steer = state[13][0][-1]
        mix_action = state[13][1][-1]
        throttle = max(mix_action, 0)
        brake = -min(mix_action, 0)
        
        self.display.blit(
            self.font.render('%f accumulated reward' % reward, True, (255, 255, 255)),
            (8, 10))
        self.display.blit(
            self.font.render('steer=%f, throttle=%f, brake=%f' % (steer, throttle, brake),
            True, (255, 255, 255)), (8, 28))
        self.display.blit(
            self.font.render('qeval : %f' % evaluation,
            True, (255, 255, 255)), (8, 64))
        self.display.blit(
            self.font.render('pdist : %f' % pdist,
            True, (255, 255, 255)), (8, 82))
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
        action = np.zeros((1, 2), dtype=np.float32)

        milliseconds = config.tick_per_action / config.fps
        _steer_cache = state[13][0][-1]
        mix_action = 2
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            mix_action = 4

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            mix_action = 0

        steer_increment =  milliseconds
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if _steer_cache > 0:
                _steer_cache = 0
            _steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if _steer_cache < 0:
                _steer_cache = 0
            _steer_cache += steer_increment
        else:
            _steer_cache = 0.0
        
        steer = np.clip(round(_steer_cache, 4), -0.7, 0.7)        
        action[0][0] = steer
        action[0][1] = mix_action

        return action
    
    def exit(self):
        self.am.destroy_all_npc()
        self.reset(hard_reset=True, tick_once=False)
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
        self.sync_mode.tick(timeout=2.0)
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
        self.world.tick()
        
        time.sleep(1)
        self.am.spawn_npc(config.num_vehicle, config.num_pedestrian)
        self.sync_mode.add_main_queue()
        
        ret = False
        self.sync_mode.callback = self.update_db
        while not ret:
            self.sync_mode.tick(timeout=2.0)
            ret = self.sync_mode.ret


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dynamic_encoder = nn.GRU(config.embedding_size + 16, 128, bidirectional=True)
        self.positional_embeddings = nn.Embedding(config.num_rays + 3, config.embedding_size, padding_idx=0)
        self.project = nn.Linear(128, 76)
        
        self.mlp_1 = nn.Linear(config.num_rays + 16, 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(140, 96)
        self.mlp_4 = nn.Linear(1, 4)
        self.mlp_5 = nn.Linear(100, 64)
        self.mlp_6 = nn.Linear(64, 32)
        self.mlp_7 = nn.Linear(32, 1)
    
    def forward(self, x_static, index_seq, sequence, input_lengths, action):
        batch_size = len(input_lengths)
        positional = self.positional_embeddings(index_seq)
        dynamic_feature = torch.cat([sequence, positional], axis=2)
        packed = nn.utils.rnn.pack_padded_sequence(dynamic_feature, input_lengths, batch_first=False, enforce_sorted=False)
        _, hidden = self.dynamic_encoder(packed)
        x_dynamic = torch.mean(hidden, axis=0)
        
        x = self.mlp_1(x_static)
        x = self.leakyRelu(x)
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        x = torch.cat([x, self.leakyRelu(self.project(x_dynamic))], axis=1)
        x = self.mlp_3(x)
        x = self.leakyRelu(x)
        
        y = self.leakyRelu(self.mlp_4(action))
        x = torch.cat([x, y], axis=1)
        x = self.mlp_5(x)
        x = self.leakyRelu(x)
        x = self.mlp_6(x)
        x = self.leakyRelu(x)
        x_ret = self.mlp_7(x)
        
        return x_ret.squeeze(-1)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        self.dynamic_encoder = nn.GRU(config.embedding_size + 16, 128, bidirectional=True)
        self.positional_embeddings = nn.Embedding(config.num_rays + 3, config.embedding_size, padding_idx=0)
        self.mode_embedding = nn.Embedding(2, 8)
        self.project = nn.Linear(128, 76)
        self.mlp_1 = nn.Linear(config.num_rays + 16, 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(140, 120)
        self.mlp_4 = nn.Linear(120, 64)
        self.mlp_5 = nn.Linear(64, 2)
    
    def forward(self, x_static, index_seq, sequence, input_lengths):        
        batch_size = len(input_lengths)
        positional = self.positional_embeddings(index_seq)
        dynamic_feature = torch.cat([sequence, positional], axis=2)
        packed = nn.utils.rnn.pack_padded_sequence(dynamic_feature, input_lengths, batch_first=False, enforce_sorted=False)
        _, hidden = self.dynamic_encoder(packed)
        x_dynamic = torch.mean(hidden, axis=0)
        x_dynamic = self.dropout_1(x_dynamic)
        
        x = self.mlp_1(x_static)
        x = self.leakyRelu(x)
        x = self.dropout_2(x)
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        x = torch.cat([x, self.leakyRelu(self.project(x_dynamic))], axis=1)
        
        x = self.mlp_3(x)
        x_mid = self.leakyRelu(x)
        
        x = self.mlp_4(x_mid)
        x = self.leakyRelu(x)
        x_ret = self.mlp_5(x)
        
        mu = x_ret[:,:1]
        log_sigma = x_ret[:,1:]
        log_sigma = torch.clamp(log_sigma, config.sigma_min, config.sigma_max)
        
        return mu, log_sigma
    
    def sample_from_density_fn(self, mu, log_sigma, return_pi=True, deterministic=False):
        if deterministic:
            pi = None
            sample = mu
        else:
            std = torch.exp(log_sigma)
            ndist = N.Normal(mu, std)
            sample = ndist.rsample()
            if return_pi:
                pi = (ndist.log_prob(sample) - (2 * (np.log(2) - sample - F.softplus(-2 * sample)))).sum(1)
            else:
                pi = None
        
        action = torch.tanh(sample)
        return action, pi


def test_network(actor=None, max_frames=5000):
    setting = config.render
    config.render = True
    state = env.start_new_game(tick_once=True)
    total_reward = 0
    
    def get_data(this_batch):
        x_static_t = torch.stack([this_batch[i][0] for i in range(len(this_batch))]).to(device)
        index_t = [this_batch[i][1].to(device) for i in range(len(this_batch))]
        batch_t = [this_batch[i][2].to(device) for i in range(len(this_batch))]
        input_lengths_t = np.array([len(index_t[i]) for i in range(len(index_t))], dtype=np.int32)
        mask = torch.LongTensor([this_batch[i][3] for i in range(len(this_batch))]).to(device)
        index_t = nn.utils.rnn.pad_sequence(index_t)
        batch_t = nn.utils.rnn.pad_sequence(batch_t)
        
        return (x_static_t, index_t, batch_t, input_lengths_t, mask)

    for i in range(max_frames):
        if state[10] == 2:
            continue

        with torch.no_grad():            
            if actor is not None:
                if env.should_quit():
                    break
                action, _ = actor.sample_from_density_fn(*actor(*get_data([[torch.from_numpy(state[1]), state[3][0], state[3][1], state[7]]])), return_pi=False)
                evaluation = q1(*get_data([[torch.from_numpy(state[1]), state[3][0], state[3][1], state[7]]]), action)
                speed = state[1][-1] * config.v_scale
                evaluation = evaluation.squeeze(0).cpu().item()
            else:
                action = env.parse_inputs(state)
                if action is None:
                    break
                action = torch.Tensor(action)
                evaluation = 0
                prob_dist = 0
            
            batch_action = np.zeros((1, 3), dtype=np.float32)
            index = [0]
            batch_action[:,:-1][index] = action.detach().cpu().numpy()
            reward,_,_ = env.step([state], batch_action, trace=True)
            total_reward += reward[0]
            env.render(state, total_reward, evaluation, prob_dist)
            if i % 100 == 0:
                print('completed %f percent of trajectory' % round(i * 100 / max_frames, 2))
    
    print('total_reward....' , total_reward)
    env.reset()
    config.render = setting
    return state


class Experience_Buffer(object):
    def __init__(self, remove_size=25000):
        self.buffer_size = config.buffer_size
        self.state_buffer = [None for _ in range(self.buffer_size)]
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, config.num_action), dtype=np.float32)
        self.next_state_buffer = [None for _ in range(config.buffer_size)]
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rm_size = remove_size
        self.start = 0
        self.end = 0
        self.size = 0

    def load_state(self):
        try:
            dbfile = open(config.path_to_save + 'buffer/dict', 'rb')     
        except:
            print('file not found!')
            return

        exp_state = pickle.load(dbfile)
        self.state_buffer = exp_state['state_buffer']
        self.reward_buffer = exp_state['reward_buffer']
        self.action_buffer = exp_state['action_buffer']
        self.next_state_buffer = exp_state['next_state_buffer']
        self.done_buffer = exp_state['done_buffer']
        self.size = exp_state['size']
        self.start = exp_state['start']
        self.end = exp_state['end']
        dbfile.close()

    def save_state(self):
        exp_state = {}
        exp_state['state_buffer'] = self.state_buffer
        exp_state['reward_buffer'] = self.reward_buffer
        exp_state['action_buffer'] = self.action_buffer
        exp_state['next_state_buffer'] = self.next_state_buffer
        exp_state['done_buffer'] = self.done_buffer
        exp_state['size'] = self.size
        exp_state['start'] = self.start
        exp_state['end'] = self.end
        dbfile = open(config.path_to_save + 'buffer/dict' , 'wb')
        pickle.dump(exp_state, dbfile)
        dbfile.close()

    def remove_observations(self, size):
        size = min(size, self.size)
        self.end = (self.end + size) % self.buffer_size
        self.size -= size

    def add_observation(self, state, action, reward, next_state, done):
        if self.start == self.end and self.size:
            print('queue is full! freeing %d data', self.rm_size)
            self.remove_observations(self.rm_size)

        self.state_buffer[self.start] = state
        self.action_buffer[self.start] = action
        self.reward_buffer[self.start] = reward
        self.next_state_buffer[self.start] = next_state
        self.done_buffer[self.start] = done
        self.start = (self.start + 1) % self.buffer_size
        self.size += 1

    def generate_batch(self, batch_size, ratio=1, w=1, cmin=15000):
        batch_size = min(batch_size, self.size)
        if not batch_size:
            return None

        support = max(self.buffer_size * (w ** (1000 * ratio)), cmin)
        offset = max(self.size - support, 0)
        idx = np.random.randint(self.end + offset, self.end + self.size, size=batch_size) % self.buffer_size        
        x_static_t = torch.stack([self.state_buffer[i][0] for i in idx]).to(device)
        index_t = [self.state_buffer[i][1].to(device) for i in idx]
        batch_t = [self.state_buffer[i][2].to(device) for i in idx]
        input_lengths_t = np.array([len(index_t[i]) for i in range(len(index_t))], dtype=np.int32)
        index_t = nn.utils.rnn.pad_sequence(index_t)
        batch_t = nn.utils.rnn.pad_sequence(batch_t)
        
        x_static_t1 = torch.stack([self.next_state_buffer[i][0] for i in idx]).to(device)
        index_t1 = [self.next_state_buffer[i][1].to(device) for i in idx]
        batch_t1 = [self.next_state_buffer[i][2].to(device) for i in idx]
        input_lengths_t1 = np.array([len(index_t1[i]) for i in range(len(index_t1))], dtype=np.int32)
        index_t1 = nn.utils.rnn.pad_sequence(index_t1)
        batch_t1 = nn.utils.rnn.pad_sequence(batch_t1)

        batch = { 'state' : (x_static_t, index_t, batch_t, input_lengths_t), 'action' : torch.Tensor(self.action_buffer[idx]).to(device), 'reward' : torch.Tensor(self.reward_buffer[idx]).to(device), 'next_state' : (x_static_t1, index_t1, batch_t1, input_lengths_t1),  'done' : torch.Tensor(self.done_buffer[idx]).to(device)}
        
        return batch


def learn(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, max_trajectory_length=500, num_policy_trajectory=3, max_attempt=3, random_explore=False, counter=0, train_steps=0, neta=1):
    state = []
    
    for i in range(num_policy_trajectory):
        for j in range(max_attempt):
            game = env.start_new_game()
            if len(game):
                state.append(game)
                break
    
    if not len(state):
        print('failed to start a new game')
        env.reset()
        return -1, {}
    
    batch_size = len(state)
    num_states = 0
    timestep = 0
    count = sum([1 if state[i][10] == 2 else 0 for i in range(batch_size)])
    
    print('........preprocessing games........')
    if count == batch_size:
        print('game finished before it started, batch_size=%d' % batch_size)
        env.reset()
        return 0, {}

    for i in range(config.tick_after_start):
        batch_action = np.zeros((batch_size, 3), dtype=np.float32)
        for j in range(batch_size):
            if batch_size + i - j < config.tick_after_start:
                batch_action[j, 1] = 0
            else:
                batch_action[j, 1] = 0
        env.step(state, batch_action, override=True)
    
    count = sum([1 if state[i][10] == 2 else 0 for i in range(batch_size)])
    if count == batch_size:
        print('game finished after const ticking, batch_size=%d' % batch_size)
        env.reset()
        return 0, {}
    
    def get_data(this_batch):
        x_static_t = torch.stack([this_batch[i][0] for i in range(len(this_batch))]).to(device)
        index_t = [this_batch[i][1].to(device) for i in range(len(this_batch))]
        batch_t = [this_batch[i][2].to(device) for i in range(len(this_batch))]
        input_lengths_t = np.array([len(index_t[i]) for i in range(len(index_t))], dtype=np.int32)
        index_t = nn.utils.rnn.pad_sequence(index_t)
        batch_t = nn.utils.rnn.pad_sequence(batch_t)
        
        return (x_static_t, index_t, batch_t, input_lengths_t)
    
    print('games start count : ', batch_size - count)
    rewards = np.zeros(batch_size, dtype=np.float32)
    batch_action = np.zeros((batch_size, 3), dtype=np.float32)
    total_rewards = np.zeros(batch_size, dtype=np.float32)
    trajectory_length = np.zeros(batch_size, dtype=np.int32)
    policy_data = [None for _ in range(batch_size)]
    ac_loss, q1_loss, q2_loss, entropy  = 0, 0, 0, 0
    timestep = 0
    
    while timestep < max_trajectory_length and count < batch_size:
        this_batch = []
        index = []
        timestep += 1

        for i in range(batch_size):
            if state[i][10] == 0:
                counter += 1
                num_states += 1
                trajectory_length[i] += 1
                this_batch.append([torch.from_numpy(state[i][1]), state[i][3][0], state[i][3][1]])
                index.append(i)

        with torch.no_grad():
            if len(index):
                if random_explore:
                    action = np.random.uniform(low=-config.continious_range, high=config.continious_range, size=len(index)).astype(np.float32)
                    batch_action[:, 0][index] = action
                else:
                    action, _ = actor.sample_from_density_fn(*actor(*get_data(this_batch)), return_pi=False)
                    batch_action[:, 0][index] = action.squeeze(-1).detach().cpu().numpy()
                
                batch_action[:, 1][index] = np.clip(np.random.randn(len(index)) * 0.3 + 0.39, 0.36, 0.42) * np.random.choice([0,1], size=len(index), p=[0.1, 0.9])

                for i, idx in enumerate(index):
                    rewards[idx] = 0
                    policy_data[idx] = deepcopy(this_batch[i])
            
            step_rewards, start_state, _ = env.step(state, batch_action)
            
            for i in range(batch_size):
                rewards[i] += step_rewards[i]
                total_rewards[i] += step_rewards[i]
                
                if start_state[i] != 2 and state[i][10] != 1:
                    done = state[i][10] == 2
                    buffer.add_observation(deepcopy(policy_data[i]), deepcopy(batch_action[i][:1]), rewards[i], [torch.from_numpy(state[i][1]).clone(), state[i][3][0].clone(), state[i][3][1].clone()], done)
            
            count = sum([1 if state[i][10] == 2 else 0 for i in range(batch_size)])
            
        if counter >= config.update_every and buffer.size > config.min_buffer_size and not random_explore:
            num_batches = max(counter // config.step_per_lr_update, 1)
            counter = 0
            ac_loss, q1_loss, q2_loss, entropy = train_from_buffer(buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, num_steps=num_batches, train_steps=train_steps, neta=neta)
            train_steps += num_batches
    
    env.reset()
    
    for i in range(len(trajectory_length)):
        print('trajectory_length %d, reward_generated %.4f, last_reward %.4f, done %d' % (trajectory_length[i], total_rewards[i], rewards[i], state[i][10] == 2))
    
    ret = {'num_states' : num_states, 'train_stats' : [counter, train_steps, ac_loss, q1_loss, q2_loss, entropy], 'total_rewards' : total_rewards, 'trajectory_length' : trajectory_length}
    
    return 1, ret


def train_from_buffer(buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, qnormclip=1, num_steps=1, train_steps=0, neta=0.996, print_steps=False):
    q1.train()
    q2.train()
    actor.train()
    sum_ac_loss, sum_q1_loss, sum_q2_loss, sum_entropy = 0, 0, 0, 0
    
    for i in range(num_steps):
        batch = buffer.generate_batch(config.batch_size, (i+1)*1.0/num_steps, w=neta)
        
        with torch.no_grad():
            action, logp = actor.sample_from_density_fn(*actor(*batch['next_state']))
            tval1 = q1_target.forward(*batch['next_state'], action)
            tval2 = q2_target.forward(*batch['next_state'], action)
            tmin_val = torch.min(tval1, tval2)
            v_t1 = tmin_val - config.alpha * logp
            target = batch['reward'] + config.gamma * (1 - batch['done']) * v_t1
        
        val1 = q1.forward(*batch['state'], batch['action'])
        val2 = q2.forward(*batch['state'], batch['action'])
        loss_1 = loss_fn(val1, target)
        loss_2 = loss_fn(val2, target)
        q_losses = loss_1 + loss_2
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        q_losses.backward()
        nn.utils.clip_grad_norm_(q1.parameters(), qnormclip)
        nn.utils.clip_grad_norm_(q2.parameters(), qnormclip)
        q1_optimizer.step()
        q2_optimizer.step()
        
        for param in q1.parameters():
            param.requires_grad = False
        
        for param in q2.parameters():
            param.requires_grad = False
        
        action, logp = actor.sample_from_density_fn(*actor(*batch['state']))
        val1 = q1.forward(*batch['state'], action)
        val2 = q2.forward(*batch['state'], action)
        min_val = torch.min(val1, val2)
        state_loss = (config.alpha * logp - min_val).mean()
        actor_optimizer.zero_grad()
        state_loss.backward()
        actor_optimizer.step()
        
        logp = logp.detach()
        g_alpha = -(logp + config.target_entropy_steering).mean()
        config.alpha = np.clip(config.alpha - config.lr * g_alpha.item(), config.min_alpha, config.max_alpha)
        
        with torch.no_grad():
            for param, param_t in zip(q1.parameters(), q1_target.parameters()):
                param_t.data.mul_(config.polyak)
                param_t.data.add_((1 - config.polyak) * param.data)

            for param, param_t in zip(q2.parameters(), q2_target.parameters()):
                param_t.data.mul_(config.polyak)
                param_t.data.add_((1 - config.polyak) * param.data)
        
        for param in q1.parameters():
            param.requires_grad = True
        
        for param in q2.parameters():
            param.requires_grad = True
        
        q1.requires_grad = True
        q2.requires_grad = True
        differential_entropy = -logp.mean()
        
        ac_loss = state_loss.item()
        q1_loss = loss_1.item()
        q2_loss = loss_2.item()
        entropy = differential_entropy.item()
        
        sum_ac_loss += ac_loss
        sum_q1_loss += q1_loss
        sum_q2_loss += q2_loss
        sum_entropy += entropy
        
        train_steps += 1
        
        writer.add_scalar('actor_loss', ac_loss, train_steps)
        writer.add_scalar('q1_loss', q1_loss, train_steps)
        writer.add_scalar('q2_loss', q2_loss, train_steps)
        writer.add_scalar('entropy', differential_entropy.item(), train_steps)
        writer.add_scalar('alpha', config.alpha, train_steps)
        
        if print_steps:
            print('step[%d/%d] : actor %.4f q1 %.4f q2 %.4f steer entropy %.4f' % (i+1, num_steps, ac_loss, q1_loss, q2_loss, differential_entropy.item()))
    
    print('step %d  actor %.4f  q1 %.4f  q2 %.4f  entropy %.4f  alpha %.4f' % (num_steps, sum_ac_loss / num_steps, sum_q1_loss / num_steps, sum_q2_loss / num_steps, sum_entropy / num_steps, config.alpha))

    return sum_ac_loss / num_steps, sum_q1_loss / num_steps, sum_q2_loss / num_steps, sum_entropy / num_steps


# In[37]:


def train(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, random_explore=True, failure_patience=15, neta=1, num_trajectory=0, train_steps=0, current_epoch=0, epochs=2, step_per_epoch=5, save_prefix=''):
    q1_target.requires_grad = False
    q2_target.requires_grad = False
    for param_1, param_2 in zip(q1_target.parameters(), q2_target.parameters()):
        param_1.requires_grad = False
        param_2.requires_grad = False
    
    loss_fn = nn.MSELoss()
    random_reward = 0
    patience = 0
    
    if random_explore:
        total_steps = 0
        t = 0
        
        while total_steps < config.random_explore and t < 100:
            print('random exploration, steps=', total_steps)
            status, rets = learn(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, num_policy_trajectory=10, counter=0, train_steps=0, neta=neta, random_explore=True)
            if status != 1:
                continue
            rewards = np.array(rets['total_rewards']).mean()
            random_reward += rewards
            total_steps += rets['num_states']
            t += 1

        random_reward /= t
        config.random_reward = random_reward
    
    counter = 0
    epochs += current_epoch
    ac_loss, q1_loss, q2_loss, entropy = np.inf, np.inf, np.inf, np.inf
    
    for i in range(current_epoch, epochs):
        for k in range(step_per_epoch):
            status, rets = learn(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, num_policy_trajectory=10, counter=counter, train_steps=train_steps, neta=neta)
            
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
            
            counter, train_steps, ac_loss, q1_loss, q2_loss, entropy = rets['train_stats']
            rewards = rets['total_rewards'] 
            trajectory_lengths = rets['trajectory_length']
            
            for j in range(len(rewards)):
                num_trajectory += 1
                writer.add_scalar('relative_improvement', rewards[j] - config.random_reward, num_trajectory)
                writer.add_scalar('rewards', rewards[j], num_trajectory)
                writer.add_scalar('trajectory_length', trajectory_lengths[j], num_trajectory)
        
        neta = max(1 + (config.neta - 1) * ((i - current_epoch + 1) * 1.5 / (epochs - current_epoch)), config.neta)
        print('\n..............\n epoch %d completed : total train steps = %d, neta = %.4f \n..............\n' % (i, train_steps, neta))

        if (i+1) % config.save_every == 0 and config.path_to_save != '':
            print('hard resetting..')
            torch.save({
                'epoch': i,
                'actor_state_dict' : actor.state_dict(),
                'q1_state_dict' : q1.state_dict(),
                'q2_state_dict' : q2.state_dict(),
                'q1_target_state_dict' : q1_target.state_dict(),
                'q2_target_state_dict' : q2_target.state_dict(),
                'actor_optimizer_state_dict' : actor_optimizer.state_dict(),
                'q1_optimizer_state_dict' : q1_optimizer.state_dict(),
                'q2_optimizer_state_dict' : q2_optimizer.state_dict(),
                'actor_loss': ac_loss,
                'q1_loss' : q1_loss,
                'q2_loss' : q2_loss,
                'entropy' : entropy,
                'num_trajectories' : num_trajectory,
                'train_steps' : train_steps,
                'neta' : neta,
                'alpha' : config.alpha,
                }, config.path_to_save + 'checkpoint_' + save_prefix + str(i) + '.pth')
            if (i+1) % config.save_dict_cycle == 0:
                buffer.save_state()


def main():
    client = carla.Client('localhost', config.sim_port)
    client.set_timeout(2.0)
    world = client.get_world()
    carla_map = world.get_map()

    directories = os.listdir(config.expert_directory)
    logger = SummaryWriter(sys.argv[1])

    dumps = []
    for d in directories:
        file = os.path.join(config.expert_directory, d, 'trajectories')
        try:
            with open(file, 'rb') as dbfile:
                dumps.append(pickle.load(dbfile))
        except:
            pass

    q1 = Critic()
    q2 = Critic()
    actor = Actor()
    q1_target = deepcopy(q1)
    q2_target = deepcopy(q2)

    if use_cuda:
        q1.cuda()
        q2.cuda()
        q1_target.cuda()
        q2_target.cuda()
        actor.cuda()

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr)
    q1_optimizer = optim.Adam(q1.parameters(), lr=config.lr)
    q2_optimizer = optim.Adam(q2.parameters(), lr=config.lr)

    def load_model(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        q1.load_state_dict(checkpoint['q1_state_dict'])
        q2.load_state_dict(checkpoint['q2_state_dict'])
        q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        config.alpha = checkpoint['alpha']

        return checkpoint['epoch'], checkpoint['num_trajectories'], checkpoint['train_steps'], checkpoint['neta']

    
    ds = Dataset.Dataset(dumps, carla_map, 1024, 1024, config.grid_dir)
    env = Environment(10, ds, client, world)
    env.initialize()

    try:
        neta=1
        num_trajectory=0
        train_steps=0
        current_epoch=0
        
        if len(sys.argv) > 2:
            current_epoch, num_trajectory, train_steps, neta = load_model(sys.argv[2])
            print('starting from ...', current_epoch, num_trajectory, train_steps, neta, config.alpha)
        
        exp = Experience_Buffer()
        exp.load_state()
        train(env, exp, logger, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, random_explore=True,\
              epochs=70, step_per_epoch=15, current_epoch=current_epoch, neta=neta, num_trajectory=num_trajectory, train_steps=train_steps)

    except:
        traceback.print_exception(*sys.exc_info())
    finally:
        print('cleaning')
        logger.close()
        env.exit()
        exp.save_state()


if __name__ == '__main__':
    main()

