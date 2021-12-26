#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob

try:
    sys.path.append(glob.glob('/home/harsh/Documents/carla_sim/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

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
import torch.distributions.bernoulli as bn
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

import Dataset


# In[ ]:


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


# In[ ]:


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
        
        self.sensors.pop(id)
        self._queues.pop(id + 1)
        
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


# In[ ]:


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
carla_map = world.get_map()


# In[ ]:


_dir = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/collected_trajectories/'
directories = os.listdir(_dir)


# In[ ]:


dumps = []
for d in directories:
    file = os.path.join(_dir, d, 'trajectories')
    try:
        with open(file, 'rb') as dbfile:
            dumps.append(pickle.load(dbfile))
    except:
        pass


# In[ ]:


out = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/cache/image.png'
ds = Dataset.Dataset(dumps, carla_map, 1024, 1024, out)


# In[ ]:


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import carla

class MapGenerator(object):
    def __init__(self, size_x, size_y, map_ratio, OVERLAP_TH=169):
        self.map_ratio = map_ratio
        self.size_x = size_x
        self.size_y = size_y
        self.overlap = 0
        self.OVERLAP_TH = OVERLAP_TH
    
    def color_points_in_quadrilateral(self, P, img, origin, reward_field, max_reward, min_reward, val):
        for i in range(4):
            P[i] = P[i] - origin
        
        x_mid, x0_mid = (P[0] + P[1]) / 2 , (P[2] + P[3]) / 2
        diff = x_mid - x0_mid
        tangent = diff / np.linalg.norm(diff)
        normal_vec = (-tangent[1], tangent[0])
        drop_factor = (min_reward - max_reward) / max(np.linalg.norm(x_mid - P[0]), np.linalg.norm(x0_mid - P[2]))
        
        for i in range(4):
            P[i] = self.map_ratio * P[i]

        size_y, size_x = img.shape
        A1 = np.linalg.inv(np.vstack((P[1] - P[0], P[2] - P[0])).T)
        A2 = np.linalg.inv(np.vstack((P[3] - P[0], P[2] - P[0])).T)
        pnts = np.vstack(P).T
        
        x_max = round(min(np.max(pnts[0]) + 1, size_x))
        y_max = round(min(np.max(pnts[1]) + 1, size_y))
        x_min = round(max(np.min(pnts[0]) - 1, 0))
        y_min = round(max(np.min(pnts[1]) - 1, 0))
        b = np.zeros((2, 1))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                b[0][0] = x - P[0][0]
                b[1][0] = y - P[0][1]
                
                s, t = A1.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    if img[y, x] != 0 and img[y, x] != val:
                        self.overlap += 1
                    img[y, x] = val
                    _y = y / self.map_ratio
                    _x = x / self.map_ratio
                    distance = abs((_x - x0_mid[0]) * normal_vec[0] + (_y - x0_mid[1]) * normal_vec[1])
                    reward = drop_factor * distance + max_reward
                    reward_field[0][y, x], reward_field[1][y, x] = reward * tangent
                    continue

                s, t = A2.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    if img[y, x] != 0 and img[y, x] != val:
                        self.overlap += 1
                    img[y, x] = val
                    _y = y / self.map_ratio
                    _x = x / self.map_ratio
                    distance = abs((_x - x0_mid[0]) * normal_vec[0] + (_y - x0_mid[1]) * normal_vec[1]) 
                    reward = drop_factor * distance + max_reward
                    reward_field[0][y, x], reward_field[1][y, x] = reward * tangent
            
    
    def generate_drivable_area(self, waypoint):
        rtemp = waypoint
        ltemp = waypoint.get_left_lane()
        lanes = []

        while rtemp and rtemp.lane_type == carla.libcarla.LaneType.Driving and rtemp.lane_id * waypoint.lane_id > 0:
            L = rtemp.previous_until_lane_start(2)
            L.reverse()
            lanes.append(L + rtemp.next_until_lane_end(2))
            rtemp = rtemp.get_right_lane()

        lanes.reverse()

        while ltemp and ltemp.lane_type == carla.libcarla.LaneType.Driving and ltemp.lane_id * waypoint.lane_id > 0:
            L = ltemp.previous_until_lane_start(2)
            L.reverse()
            lanes.append(L + ltemp.next_until_lane_end(2))
            ltemp = ltemp.get_left_lane()

        plots = [[[], []] for _ in range(2 * len(lanes))]
        for i in range(len(lanes)):
            length = len(lanes[i])
            j = 0
            prev_boundary_1 = 0
            prev_boundary_2 = 0
            while j < length:
                if lanes[i][j].road_id != waypoint.road_id or lanes[i][j].lane_id * waypoint.lane_id < 0:
                    lanes[i].pop(j)
                    length -= 1
                    continue
                g = lanes[i][j].transform
                loc = g.location
                rv = g.rotation.get_right_vector()
                T = np.array([loc.x, loc.y, loc.z])
                X = np.array([rv.x, rv.y, rv.z])
                w = lanes[i][j].lane_width / 2

                boundary_1 = w * X + T
                boundary_2 = -w * X + T

                prev_boundary_1 = boundary_1
                prev_boundary_2 = boundary_2

                plots[2*i][0].append(boundary_1[0])
                plots[2*i][1].append(boundary_1[1])
                plots[2*i+1][0].append(boundary_2[0])
                plots[2*i+1][1].append(boundary_2[1])
                j += 1

            plots[2*i] = np.array(plots[2*i])
            plots[2*i+1] = np.array(plots[2*i + 1])

        return plots, lanes
    
    def build_multiple_area_upwards(self, waypoints, max_reward=1, min_reward=0, padding=1, retain_road_label=False):
        num_roads = len(waypoints)
        offset = [self.size_x // 4, self.size_y // 4]
        plots = [None for _ in range(num_roads)]
        lanes = [None for _ in range(num_roads)]
        img = np.zeros((self.size_y, self.size_x))
        values = np.arange(0.1, 1.0, 0.8 / num_roads)
        reward_field = np.zeros((2, self.size_y, self.size_x))
        self.overlap = 0
        
        for i in range(num_roads):
            plots[i], lanes[i] = self.generate_drivable_area(waypoints[i])
        
        x_min, y_min = 1e7, 1e7
        x_max, y_max = -1e7, -1e7
        
        for i in range(len(plots[0])):
            x_min = min(x_min, plots[0][i][0][0])
            y_min = min(y_min, plots[0][i][1][0])
            x_max = max(x_max, plots[0][i][0][0])
            y_max = max(y_max, plots[0][i][1][0])
        
        for i in range(num_roads):
            k = 0
            limit = max([len(plots[i][a][0]) for a in range(len(plots[i]))])
            stops = [False for _ in range(len(plots[i]))]
            while k < limit:
                for j in range(len(plots[i])):
                    if len(plots[i][j][0]) <= k or stops[j]:
                        continue
                    
                    x_min_ = min(x_min, plots[i][j][0][k])
                    y_min_ = min(y_min, plots[i][j][1][k])
                    x_max_ = max(x_max, plots[i][j][0][k])
                    y_max_ = max(y_max, plots[i][j][1][k])
                    x_req = round((x_max_ - x_min_ + 2 * padding) * self.map_ratio)
                    y_req = round((y_max_ - y_min_ + 2 * padding) * self.map_ratio)
                    
                    if x_req >= self.size_x or y_req >= self.size_y:
                        stops[j] = True                    
                    else:
                        x_max = x_max_
                        x_min = x_min_
                        y_max = y_max_
                        y_min = y_min_
                k += 1
            
            if np.sum(stops) > 0:
                break
        
        xreq = round((x_max - x_min + 2 * padding) * self.map_ratio)
        yreq = round((y_max - y_min + 2 * padding) * self.map_ratio)
        max_offset_x = self.size_x - xreq - 1
        max_offset_y = self.size_y - yreq - 1
        offset[0] = min(max_offset_x, offset[0]) / self.map_ratio
        offset[1] = min(max_offset_y, offset[1]) / self.map_ratio
        origin = np.array([x_min - offset[0] - padding, y_min - offset[1] - padding])
        traversable_wp = []
        stop = False
        
        for i in range(num_roads):
            traversable_wp.append([])
            self.overlap = 0
            for j in range(0, len(plots[i]), 2):
                traversable_wp[i].append(None)
                k = 0
                while k < len(plots[i][j][0]):
                    if round((plots[i][j][0][k] - x_min + 2 * padding) * self.map_ratio) >= self.size_x:
                        stop = True
                        break
                    if round((x_max - plots[i][j][0][k] + 2 * padding) * self.map_ratio) >= self.size_x:
                        stop = True
                        break
                    if round((plots[i][j][1][k] - y_min + 2 * padding) * self.map_ratio) >= self.size_y:
                        stop = True
                        break
                    if round((y_max - plots[i][j][1][k] + 2 * padding) * self.map_ratio) >= self.size_y:
                        stop = True
                        break
                    if round((plots[i][j+1][0][k] - x_min + 2 * padding) * self.map_ratio) >= self.size_x:
                        stop = True
                        break
                    if round((x_max - plots[i][j+1][0][k] + 2 * padding) * self.map_ratio) >= self.size_x:
                        stop = True
                        break
                    if round((plots[i][j+1][1][k] - y_min + 2 * padding) * self.map_ratio) >= self.size_y:
                        stop = True
                        break
                    if round((y_max - plots[i][j+1][1][k] + 2 * padding) * self.map_ratio) >= self.size_y:
                        stop = True
                        break
                    
                    if k > 0:
                        points = [plots[i][j][:, k], plots[i][j+1][:, k], plots[i][j+1][:, k-1], plots[i][j][:, k-1]]
                        self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
                    
                    k += 1
                
                traversable_wp[i][j // 2] = lanes[i][j // 2][:k]
            
            if self.overlap > self.OVERLAP_TH:
                print('warning overlapping roads detected with %d overlap pixels' % self.overlap)
                return self.build_multiple_area_upwards(waypoints[:i], max_reward, min_reward, padding, retain_road_label)
            
            if stop:
                break
        
        matches = []
        for i in range(num_roads - 1):
            matches.append(self.join_roads(plots[i], plots[i+1]))

        for i in range(num_roads - 1):
            keys = sorted(list(matches[i].keys()))
            if len(keys) % 2 != 0:
                print('warning : unmatched lanes in dict !!')
                mx,mn = max(keys), min(keys)
                tx,tn = matches[i][mx], matches[i][mn]
                points = [np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                          np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]]),                          np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                          np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]])]
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
                continue
            
            for j in range(0, len(keys), 2):
                mx,mn = keys[j+1], keys[j]
                tx,tn = matches[i][mx], matches[i][mn]
                points = [np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                          np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]]),                          np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                          np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]])]
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
        
        if not retain_road_label:
            img[img > 0] = 1.0
        
        return img, reward_field, traversable_wp, origin
    
    def sparse_build_multiple_area_upwards(self, waypoints, max_reward=1, min_reward=0, zero_padding=0, retain_road_label=False):
        num_roads = len(waypoints)
        offset = [self.size_x // 4, self.size_y // 4]
        plots = [None for _ in range(num_roads)]
        lanes = [None for _ in range(num_roads)]
        values = np.arange(1, num_roads+1)
        self.overlap = 0
        
        for i in range(num_roads):
            plots[i], lanes[i] = self.generate_drivable_area(waypoints[i])
        
        x_min, y_min = 1e7, 1e7
        x_max, y_max = -1e7, -1e7
        
        for i in range(num_roads):
            for j in range(len(plots[i])):
                x_min = min(x_min, np.min(plots[i][j][0]))
                y_min = min(y_min, np.min(plots[i][j][1]))
                x_max = max(x_max, np.max(plots[i][j][0]))
                y_max = max(y_max, np.max(plots[i][j][1]))
        
        x_range = round((x_max - x_min) * self.map_ratio) + 2 * zero_padding + 1
        y_range = round((y_max - y_min) * self.map_ratio) + 2 * zero_padding + 1
        img = lil_matrix((y_range, x_range), dtype=np.float32)
        reward_field_x = lil_matrix((y_range, x_range), dtype=np.float32)
        reward_field_y = lil_matrix((y_range, x_range), dtype=np.float32)
        reward_field = [reward_field_x, reward_field_y]
        origin = np.array([x_min - zero_padding / self.map_ratio, y_min - zero_padding / self.map_ratio])
        
        traversable_wp = []
        for i in range(num_roads):
            traversable_wp.append([])
            self.overlap = 0
            for j in range(0, len(plots[i]), 2):
                traversable_wp[i].append(None)
                k = 0
                while k < len(plots[i][j][0]):                    
                    if k > 0:
                        points = [plots[i][j][:, k], plots[i][j+1][:, k], plots[i][j+1][:, k-1], plots[i][j][:, k-1]]
                        self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
                    k += 1
                
                traversable_wp[i][j // 2] = lanes[i][j // 2][:k]
            
            if self.overlap > self.OVERLAP_TH:
                print('warning overlapping roads detected with %d overlap pixels' % self.overlap)
                return self.sparse_build_multiple_area_upwards(waypoints[:i], max_reward, min_reward, zero_padding, retain_road_label)
        
        matches = []
        for i in range(num_roads - 1):
            matches.append(self.join_roads(plots[i], plots[i+1]))

        for i in range(num_roads - 1):
            keys = sorted(list(matches[i].keys()))
            if len(keys) % 2 != 0:
                print('warning : unmatched lanes in dict !!')
                mx,mn = max(keys), min(keys)
                tx,tn = matches[i][mx], matches[i][mn]
                points = [np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                          np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]]),                          np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                          np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]])]
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
                continue
            
            for j in range(0, len(keys), 2):
                mx,mn = keys[j+1], keys[j]
                tx,tn = matches[i][mx], matches[i][mn]
                points = [np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                          np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]]),                          np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                          np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]])]
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward, values[i])
        
        if not retain_road_label:
            img[img > 0] = 1.0
        
        return [img, reward_field, traversable_wp, origin]
    
    def join_roads(self, plots_1, plots_2, threshold=3.0):
        M = {}
        V = {}
        F, T = None, None

        if len(plots_1) > len(plots_2):
            F = plots_1
            T = plots_2

            for i in range(len(T)):
                x = T[i][0][0]
                y = T[i][1][0]
                res = False
                dd = None
                if len(T[i][0]) > 1:
                    dd = np.array([-y + T[i][1][1], x - T[i][0][1]])
                    dd = dd / np.linalg.norm(dd)
                else:
                    res = True
                minimum, second_min = 1e7, 0
                m, s = -1, -1

                for k in range(len(F)):
                    _x = F[k][0][-1]
                    _y = F[k][1][-1]
                    if res:
                        distance = np.sqrt((_x - x) ** 2 + (_y - y) ** 2)
                    else:
                        distance = abs((_x - x) * dd[0] + (_y - y) * dd[1])
                    
                    if distance > threshold:
                        continue
                    if distance < minimum:
                        second_min = minimum
                        minimum = distance
                        s = m
                        m = k
                    elif distance < second_min:
                        second_min = distance
                        s = k

                if minimum == 1e7:
                    print('unable to find road less than %fm away' % threshold)
                else:
                    if not V.__contains__(m):
                        M[m] = i
                        V[m] = i
                    elif s != -1:
                        M[s] = i
                        V[s] = i
        else:
            F = plots_2
            T = plots_1

            for i in range(len(T)):
                x = T[i][0][-1]
                y = T[i][1][-1]
                res = False
                dd = None
                if len(T[i][0]) > 1:
                    dd = np.array([-y + T[i][1][-2], x - T[i][0][-2]])
                    dd = dd / np.linalg.norm(dd)
                else:
                    res = True
                minimum, second_min = 1e7, 0
                m, s = -1, -1

                for k in range(len(F)):
                    _x = F[k][0][0]
                    _y = F[k][1][0]
                    if res:
                        distance = np.sqrt((_x - x) ** 2 + (_y - y) ** 2)
                    else:
                        distance = abs((_x - x) * dd[0] + (_y - y) * dd[1])
                    
                    if distance > threshold:
                        continue
                    if distance < minimum:
                        second_min = minimum
                        minimum = distance
                        s = m
                        m = k
                    elif distance < second_min:
                        second_min = distance
                        s = k

                if minimum == 1e7:
                    print('unable to find road less than %fm away' % threshold)
                else:
                    if not V.__contains__(m):
                        M[i] = m
                        V[m] = i
                    elif s != -1:
                        M[i] = s
                        V[s] = i
        return M
    
    def generate_drivable_lanes(self, waypoint, plot=False, img=None):
        rtemp = waypoint
        ltemp = waypoint.get_left_lane()
        lanes = []

        while rtemp and rtemp.lane_type == carla.libcarla.LaneType.Driving and rtemp.lane_id * waypoint.lane_id > 0:
            L = rtemp.previous_until_lane_start(2)
            L.reverse()
            lanes.append(L + rtemp.next_until_lane_end(2))
            rtemp = rtemp.get_right_lane()

        lanes.reverse()

        while ltemp and ltemp.lane_type == carla.libcarla.LaneType.Driving and ltemp.lane_id * waypoint.lane_id > 0:
            L = ltemp.previous_until_lane_start(2)
            L.reverse()
            lanes.append(L + ltemp.next_until_lane_end(2))
            ltemp = ltemp.get_left_lane()

        plots = [[[], []] for _ in range(len(lanes) + 1)]
        for i in range(len(lanes)):
            length = len(lanes[i])
            j = 0
            while j < length:
                g = lanes[i][j].transform
                if lanes[i][j].road_id != waypoint.road_id or lanes[i][j].lane_id * waypoint.lane_id < 0:
                    lanes[i].pop(j)
                    length -= 1
                    continue
                loc = g.location
                rv = g.rotation.get_right_vector()
                T = np.array([loc.x, loc.y, loc.z])
                X = np.array([rv.x, rv.y, rv.z])
                w = lanes[i][j].lane_width / 2
                boundary = w * X + T
                plots[i][0].append(boundary[0])
                plots[i][1].append(boundary[1])
                if i == len(lanes) - 1:
                    boundary_2 = -w * X + T
                    plots[i+1][0].append(boundary_2[0])
                    plots[i+1][1].append(boundary_2[1])
                j += 1

            plots[i] = np.array(plots[i])
            if plot:
                plt.plot(-plots[i][0], plots[i][1])

            if i == len(lanes) - 1:
                plots[i+1] = np.array(plots[i+1])
                if plot:
                    plt.plot(-plots[i+1][0], plots[i+1][1])

        if plot:
            plt.show()

        return plots, [lanes[i][0] for i in range(len(lanes))], [lanes[i][-1] for i in range(len(lanes))]
    
    def build_upwards(self, waypoint_1, waypoint_2):
        plots_1, start_1, end_1 = self.generate_drivable_lanes(waypoint_1)
        plots_2, start_2, end_2 = self.generate_drivable_lanes(waypoint_2)

        match = self.join_roads(plots_1, plots_2)

        new_plots = []
        for i, j in match.items():
            new_plots.append([None,None])
            new_plots[-1][0] = np.concatenate([plots_1[i][0], plots_2[j][0]], axis=0)
            new_plots[-1][1] = np.concatenate([plots_1[i][1], plots_2[j][1]], axis=0)    
            plt.plot(-new_plots[-1][0], new_plots[-1][1])

        plt.show()
        return new_plots
    
    def build_multiple_upwards(self, waypoints):
        num_roads = len(waypoints)
        plots = [None for _ in range(num_roads)]
        for i in range(num_roads):
            plots[i], _, _ = self.generate_drivable_lanes(waypoints[i])

        matches = []
        for i in range(num_roads - 1):
            matches.append(self.join_roads(plots[i], plots[i+1]))

        new_plots = []
        ids = {}
        for i in range(len(plots[0])):
            new_plots.append(plots[0][i])
            ids[(i, 0)] = i

        for k in range(num_roads - 1):
            all_v = [1 for i in range(len(plots[k+1]))]
            for i, j in matches[k].items():
                n = ids[(i, k)]
                all_v[j] = 0
                new_plots[n] = np.concatenate([new_plots[n], plots[k+1][j]], axis=1)
                ids[(j, k+1)] = n

            for i in range(len(all_v)):
                if all_v[i]:
                    new_plots.append(plots[k+1][i])
                    ids[(i, k+1)] = len(new_plots) - 1

        for i in range(len(new_plots)):
            plt.plot(-new_plots[i][0], new_plots[i][1])

        plt.show()
        return new_plots


# In[ ]:


class config:
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
    expert_directory = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/collected_trajectories/'
    grid_dir = '/home/harsh/Documents/carla_sim/carla/PythonAPI/examples/cache/image.png'
    path_to_save = "/home/harsh/Documents/saved_model/sept10"


# In[ ]:


from imports import *
from mapGenerator import MapGenerator

class Dataset(object):
    def __init__(self, dumps, carla_map, size_x, size_y, occupancy_path, grid=[], map_ratio=4, window=3, out_patience=2, speed_on_stop_th=0.5):
        self.carla_map = carla_map
        self.top = carla_map.get_topology()
        self.orientations = [dumps[i]['orientation'] for i in range(len(dumps))]
        self.isVehicle = [dumps[i]['type_actor'] for i in range(len(dumps))]
        self.num_actor = [len(self.orientations[i]) for i in range(len(dumps))]
        self.num_heros = [dumps[i].__len__() - 3 for i in range(len(dumps))]
        self.map_ratio = map_ratio
        self.df_trajectory = [[None for _ in range(self.num_heros[j])] for j in range(len(dumps))]
        self.mgen = MapGenerator(size_x, size_y, map_ratio)
        self.grid = grid
        assert config.cache_size_y % 2 and config.cache_size_x % 2
        self.h_cache_size_y = config.cache_size_y // 2
        self.h_cache_size_x = config.cache_size_x // 2
        self.__load_in_pd(dumps)
        self.__fill_adj_list()
        self.__load_occupancy_grid(occupancy_path)
        self.box = []
        for i in range(len(dumps)):
            self.__fill_measurement_grid(dumps[i]['measurements'])
        self.window = window
        self.h_dynamic_size_x = 64
        self.h_dynamic_size_y = 64
        self.center = 64
        self.radius = 16
        self.dt = max(config.skips, 1) / config.fps
        self.speed_on_stop_th = speed_on_stop_th
        self.out_patience = out_patience
        self.num_sims = len(dumps)
        self.MAX_TRY_STOP_SPEED = 2
    
    def __fill_measurement_grid(self, measurement, resolution=0.75): 
        self.box.append([])
        
        for i in range(len(measurement)):
            x1, x2 = np.floor(measurement[i][0][0] * self.map_ratio), np.ceil(measurement[i][-1][0] * self.map_ratio)
            y1, y2 = np.floor(measurement[i][0][1] * self.map_ratio), np.ceil(measurement[i][-1][1] * self.map_ratio)
            x = np.arange(x1, x2 + resolution / 2, resolution)
            y = np.arange(y1, y2 + resolution / 2, resolution)
            xx, yy = np.meshgrid(x, y)
            xx = np.expand_dims(xx, -1)
            yy = np.expand_dims(yy, -1)
            grid = np.concatenate([xx, yy], axis=-1).reshape(-1, 2).T
            self.box[-1].append(grid)
    
    def __load_occupancy_grid(self, occupancy_path):
        if not len(self.grid):
            self.grid = cv2.imread(occupancy_path)
            self.grid = cv2.cvtColor(self.grid, cv2.COLOR_BGR2GRAY)
            self.grid[self.grid == 86] = 0
        self.scale = 1
        margin = 50
        waypoints = self.carla_map.generate_waypoints(2)
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin
        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = min_x, min_y
        sidewalk = np.zeros(self.grid.shape, dtype=np.uint8)
        sidewalk[self.grid == 137] = 255
        drivable = np.zeros(self.grid.shape, dtype=np.uint8)
        drivable[np.logical_or(np.logical_or(self.grid == 50, self.grid == 187), self.grid == 185)] = 255
        lanes = np.zeros(self.grid.shape, dtype=np.uint8)
        lanes[np.logical_or(self.grid == 187, self.grid == 185)] = 255
        comp = 12 / self.map_ratio
        self.sidewalk = torch.from_numpy(cv2.resize(sidewalk, (int(self.grid.shape[0] / comp), int(self.grid.shape[0] / comp))).astype(np.float32) / 255)
        self.drivable = torch.from_numpy(cv2.resize(drivable, (int(self.grid.shape[0] / comp), int(self.grid.shape[0] / comp))).astype(np.float32) / 255)
        self.lanes = torch.from_numpy(cv2.resize(lanes, (int(self.grid.shape[0] / comp), int(self.grid.shape[0] / comp))).astype(np.float32) / 255)
        self.sidewalk = F.pad(self.sidewalk, (self.h_cache_size_x, self.h_cache_size_x, self.h_cache_size_y, self.h_cache_size_y))
        self.drivable = F.pad(self.drivable, (self.h_cache_size_x, self.h_cache_size_x, self.h_cache_size_y, self.h_cache_size_y))
        self.lanes = F.pad(self.lanes, (self.h_cache_size_x, self.h_cache_size_x, self.h_cache_size_y, self.h_cache_size_y))
        self.corrected_map_ratio = 12 / (self.grid.shape[0] / (self.grid.shape[0] // comp))
    
    def __load_in_pd(self, dumps):
        for d in range(len(dumps)):
            self.df_time = pd.DataFrame(np.array([np.array(self.orientations[d][0])[:,0],                                                 [i for i in range(len(self.orientations[d][0]))]]).T,                                                 columns=['time', 'idx'])
            for j in range(self.num_heros[d]):
                ins = dumps[d][str(j)]
                i = len(ins) - 1
                while not len(ins[i]):
                    i -= 1
                ins = ins[:i+1]
                stacked_traj = np.vstack(ins)
                df_ins = pd.DataFrame(stacked_traj, columns=['trajectory', 'time', 'throttle', 'steering', 'brake', 'v_x', 'v_y', 'a_x', 'a_y'])
                df = pd.merge(df_ins, self.df_time, how='inner', on='time')
                assert df.__len__() == df_ins.__len__()
                df['idx'] = df['idx'].astype(np.int32)
                self.df_trajectory[d][j] = df
    
    def __fill_adj_list(self):
        self.adj = {}
        for i in range(len(self.top)):
            s2 = self.top[i][0].lane_id > 0
            s1 = self.top[i][0].road_id
            s3 = self.top[i][1].road_id
            s4 = self.top[i][1].lane_id > 0

            if s1 == s3 and s2 * s4 > 0:
                continue

            self.adj[(s1, s2, s3, s4)] = True
    
    def __create_cached_map(self, state):
        cached_map = torch.Tensor(config.conv_size[0], config.cache_size_y, config.cache_size_x)
        static_map = state[0]
        img = static_map[0]
        stop_target, through_target = static_map[-1]
        occ_pix = self.world_to_pixel(state[9], offset=(-self.h_cache_size_x, -self.h_cache_size_y))
        s_pt = np.around(state[4] * self.map_ratio).astype(np.int32)
        reward_field_x, reward_field_y = static_map[1]
        cached_map[0] = torch.from_numpy(reward_field_x[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[1] = torch.from_numpy(reward_field_y[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[2] = torch.from_numpy(img[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[3] = self.drivable[occ_pix[1]-self.h_cache_size_y : occ_pix[1]+self.h_cache_size_y+1, occ_pix[0]-self.h_cache_size_x : occ_pix[0]+self.h_cache_size_x+1]
        # cached_map[4] = self.sidewalk[occ_pix[1]-self.h_cache_size_y : occ_pix[1]+self.h_cache_size_y+1, occ_pix[0]-self.h_cache_size_x : occ_pix[0]+self.h_cache_size_x+1]
        cached_map[4] = self.lanes[occ_pix[1]-self.h_cache_size_y : occ_pix[1]+self.h_cache_size_y+1, occ_pix[0]-self.h_cache_size_x : occ_pix[0]+self.h_cache_size_x+1]
        cached_map[5] = torch.from_numpy(stop_target[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[6] = torch.from_numpy(through_target[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[7 : 7 + config.embedding_size] = 0
        cached_map[7 + config.embedding_size :]  = 0
        offset = s_pt.copy() - np.array([self.h_cache_size_x, self.h_cache_size_y])
        state[2] = [cached_map, offset]
    
    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self.corrected_map_ratio * (location[0] - self._world_offset[0])
        y = self.scale * self.corrected_map_ratio * (location[1] - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]
    
    def __generate_path_seq(self, sim, actor, index):
        M = {}
        L = []
        for i in range(len(index)):
            position = self.orientations[sim][actor][index[i]]
            x = position[4]
            y = position[5]
            z = position[6]
            loc = carla.Location(x,y,z)
            wp = self.carla_map.get_waypoint(loc)
            if not M.__contains__((wp.road_id, wp.lane_id > 0)):
                L.append(wp)
                M[(wp.road_id, wp.lane_id > 0)] = 1
            else:
                M[(wp.road_id, wp.lane_id > 0)] += 1

        return M, L
    
    def __get_likely_waypoint_seq(self, M, L):
        dp = np.zeros(len(L), dtype=np.int32)
        conn = np.zeros(len(L), dtype=np.int32)
        
        for i in range(len(L)):
            s1, s2 = (L[i].road_id, L[i].lane_id > 0)
            dp[i] = M[(s1, s2)]
            maximum = 0
            conn[i] = -1
            for j in range(i-1, -1, -1):
                s3, s4 = (L[j].road_id, L[j].lane_id > 0)
                if self.adj.__contains__((s3, s4, s1, s2)):
                    if maximum < dp[j]:
                        maximum = dp[j]
                        conn[i] = j
            dp[i] += maximum
        
        start = np.argmax(dp)
        path = []
        while start != -1:
            path.append(start) 
            start = conn[start]
        path = [L[i] for i in reversed(path)]
        
        return path, dp, conn
    
    def __append_dynamic_features(self, idx, sim, obs_index, state, pos, dynamic_objects):
        if not len(dynamic_objects):
            state[10].append([[], [], []])
            return
        
        rnn_features = np.zeros((config.lookback, len(dynamic_objects), 4), dtype=np.float32)
        skips = max(config.skips, 1)
        measurement = []
        
        for k, index in enumerate(dynamic_objects):
            t = 0
            
            for r in range(obs_index, skips - 1, -skips):
                if t == config.lookback:
                    break
                if t == 0:
                    dyn_data = self.orientations[sim][index][r]
                    cur_yaw, cur_x, cur_y = dyn_data[1], dyn_data[4], dyn_data[5]
                    dyn_data = self.orientations[sim][index][r - skips]
                    prev_yaw, prev_x, prev_y = dyn_data[1], dyn_data[4], dyn_data[5]
                else:
                    cur_x = prev_x
                    cur_y = prev_y
                    cur_yaw = prev_yaw
                    dyn_data = self.orientations[sim][index][r - skips]
                    prev_yaw, prev_x, prev_y = dyn_data[1], dyn_data[4], dyn_data[5]
                
                rnn_features[config.lookback - t - 1, k] = np.array([(cur_x - prev_x) / (self.dt * config.v_scale),                                                           (cur_y - prev_y) / (self.dt * config.v_scale),                                                           (cur_yaw - prev_yaw) / (self.dt * config.w_scale),                                                           idx])
                t += 1
            
            if t < config.lookback:
                assert t
                current_features = rnn_features[config.lookback - t, k]
                for r in range(t, config.lookback):
                    rnn_features[config.lookback - r - 1, k] = current_features
        
        rnn_features = torch.from_numpy(rnn_features)
        for i in range(len(dynamic_objects)):
            measurement.append(self.box[sim][dynamic_objects[i]])
        
        state[10].append([rnn_features, pos, measurement])
    
    def extract_trajectory(self, sim=-1, hero=-1, trajectory_no=-1):
        try:
            if sim == -1:
                sim = np.random.randint(self.num_sims)

            if hero == -1:
                hero = np.random.randint(self.num_heros[sim])

            total = self.df_trajectory[sim][hero].iloc[-1]['trajectory']
            if trajectory_no == -1:
                trajectory_no = np.random.randint(total+1)

            df = self.df_trajectory[sim][hero][self.df_trajectory[sim][hero]['trajectory'] == trajectory_no]
            index = list(df['idx'])
            M, seq = self.__generate_path_seq(sim, hero, index)
            path, dp, conn = self.__get_likely_waypoint_seq(M, seq)
            state = self.mgen.sparse_build_multiple_area_upwards(path, max_reward=1.0, min_reward=0.3,                                                                 zero_padding=max(self.h_cache_size_y, self.h_cache_size_x) + 1,                                                                 retain_road_label=True)        
            if len(state[-2]) != len(path):
                j = len(state[-2])
                n = len(path)
                return []
            else:
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

                road = state[-2][-1][0]
                origin = state[-1]
                road_segment = state[0].copy()
                state[0][state[0] > 0] = 1

                row = df.iloc[-1]
                e_pt = np.array([self.orientations[sim][hero][int(row['idx'])][4] - origin[0],                                 self.orientations[sim][hero][int(row['idx'])][5] - origin[1]])
                e_pt = np.around(e_pt * self.map_ratio).astype(np.int32)
                qpoints = self.fix_end_line(state[1], state[0], e_pt, offset=-5*self.window, size=7*self.window)
                mandate = False
                if not len(qpoints):
                    if len(road) > 1:
                        _last = -2
                    else:
                        _last = -1
                    e_pt = np.array([road[_last].transform.location.x, road[_last].transform.location.y])
                    e_pt = np.around((e_pt - origin) * self.map_ratio).astype(np.int32)
                    qpoints = self.fix_end_line(state[1], state[0], e_pt, offset=-6*self.window, size=7*self.window)
                    if not len(qpoints):
                        print('ERROR : failed to find end point')
                        return []
                    mandate = True

                vx, vy = row['v_x'], row['v_y']
                end_speed = np.sqrt(vx*vx + vy*vy)
                target_pass = lil_matrix(state[0].shape, dtype=np.float32)
                target_stop = lil_matrix(state[0].shape, dtype=np.float32)
                if mandate or end_speed > self.speed_on_stop_th:
                    self.color_points_in_quadrilateral(qpoints, target_pass, val=1)
                    quit_stat = False
                else:
                    self.color_points_in_quadrilateral(qpoints, target_stop, val=1)
                    quit_stat = True

                state.append([target_stop, target_pass])
                w = 1
                i = config.skips
                x,y = 0,0
                position = None

                while i < len(index):
                    position = self.orientations[sim][hero][index[i]]
                    x = position[4]
                    y = position[5]
                    _j, _i = round(self.map_ratio * (x - origin[0])), round(self.map_ratio * (y - origin[1]))
                    val = state[0][_i - w : _i + w + 1, _j - w : _j + w + 1].sum()
                    if val > 0:
                        break
                    i += 1
                
                if i == len(index):
                    raise Exception("error")

                s_pt = np.array([x - origin[0], y - origin[1]])
                yaw = position[1]
                angle = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
                trajectory_list = [state, [], [], angle, s_pt, e_pt, i, origin, [], [x, y], [], [road_segment, len(path), quit_stat], 0, sim, hero, trajectory_no]

        except:
            traceback.print_exception(*sys.exc_info())
            print(sim, hero, trajectory_no)
            return []
        
        return trajectory_list
    

    def trajectory_step(self, state):
        dynamic_map = state[1]
        trajectory_no = state[-1]
        hero = state[-2]
        sim = state[-3]
        s_pt = np.around(state[4] * self.map_ratio).astype(np.int32)
        
        if not len(state[2]):
            self.__create_cached_map(state)
        
        cached_map, offset = state[2]
        
        step_index = state[6]
        angle = state[3]
        origin = state[7]
        df = state[8]      
        
        if not len(df):
            df = self.df_trajectory[sim][hero][self.df_trajectory[sim][hero]['trajectory'] == trajectory_no]
            state[8] = df
        
        if step_index >= len(df):
            return [], True

        can_quit = self.has_quit_available(state)
        
        row = df.iloc[step_index]
        v_x, v_y = row['v_x'], row['v_y']
        a_x, a_y = row['a_x'], row['a_y']
        speed = np.sqrt(v_x * v_x + v_y * v_y)
        
        if speed < 1e-3:
            longitudinal_acc = a_x * angle[0] + a_y * angle[1]
        else:
            longitudinal_acc = (a_x * v_x + a_y * v_y) / speed
        
        steering_angle = row['steering']
        
        half_kernel = 1
        loc_y = s_pt[1]-offset[1]
        loc_x = s_pt[0]-offset[0]
        F = np.array([cached_map[0, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean(),                      cached_map[1, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean()])
        d = np.linalg.norm(F)
        if d < 1e-4:
            print('track out of range hero=%d, trajectory=%d, index=%d' % (hero, trajectory_no, step_index))
            F = angle.copy()
            if state[-4] >= self.out_patience:
                done = True
                action_t = []
                return action_t, done
            
            state[-4] += 1
            
        else:
            F = F / d
        
        c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)
        if c_y - self.h_dynamic_size_y < 0 or c_x - self.h_dynamic_size_x < 0 or           c_y + self.h_dynamic_size_y + 1 > config.cache_size_y or           c_x + self.h_dynamic_size_x + 1 > config.cache_size_x:
            self.__create_cached_map(state)
            cached_map, offset = state[2]
            loc_x = s_pt[0] - offset[0]
            loc_y = s_pt[1] - offset[1]
            c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)
        
        R = np.array([[F[0], -F[1]], [F[1], F[0]]])
        T = np.array([loc_x - c_x + self.h_dynamic_size_x,                      loc_y - c_y + self.h_dynamic_size_y]).reshape(2, 1)
        points = np.around(R.dot(self.box[sim][hero]) + T).astype(np.int32)
        
        if len(dynamic_map):
            dynamic_map[7+config.embedding_size:] = 0
        
        state[1] = cached_map[:, c_y-self.h_dynamic_size_y : c_y+self.h_dynamic_size_y+1, c_x-self.h_dynamic_size_x : c_x+self.h_dynamic_size_x+1]
        dynamic_map = state[1]
        hero_start = 7 + config.embedding_size
        val = torch.Tensor([v_x / config.v_scale, v_y / config.v_scale, longitudinal_acc / config.a_scale, steering_angle]).reshape(-1, 1)
        dynamic_map[hero_start:, points[1], points[0]] = val
        obs_index = int(df.iloc[step_index]['idx'])
        v_pos = []
        p_pos = []
        v_dynamic_objects = []
        p_dynamic_objects = []
        
        for i in range(self.num_actor[sim]):
            obs = self.orientations[sim][i][obs_index]
            x, y = round((obs[4] - origin[0]) * self.map_ratio), round((obs[5] - origin[1]) * self.map_ratio)
            _yaw_ = obs[1]
            if i == hero:
                continue
            
            if abs(x - c_x - offset[0]) < self.h_dynamic_size_x and abs(y - c_y - offset[1]) < self.h_dynamic_size_y:
                if self.isVehicle[sim][i]:
                    v_pos.append([np.cos(_yaw_ * np.pi / 180), np.sin(_yaw_ * np.pi / 180), x - c_x - offset[0] + self.h_dynamic_size_x, y - c_y - offset[1] + self.h_dynamic_size_y])
                    v_dynamic_objects.append(i)
                else:
                    p_pos.append([np.cos(_yaw_ * np.pi / 180), np.sin(_yaw_ * np.pi / 180), x - c_x - offset[0] + self.h_dynamic_size_x, y - c_y - offset[1] + self.h_dynamic_size_y])
                    p_dynamic_objects.append(i)
        
        state[10] = []
        self.__append_dynamic_features(0, sim, obs_index, state, v_pos, v_dynamic_objects)
        self.__append_dynamic_features(1, sim, obs_index, state, p_pos, p_dynamic_objects)
        
        if len(state[10][0][0]) and len(state[10][1][0]):
            accum_tensor = torch.cat([state[10][0][0], state[10][1][0]], axis=1)
        elif len(state[10][0][0]):
            accum_tensor = state[10][0][0]
        elif len(state[10][1][0]):
            accum_tensor = state[10][1][0]
        else:
            accum_tensor = []
        
        accum_position = state[10][0][1] + state[10][1][1]
        accum_points = state[10][0][2] + state[10][1][2]
        state[10] = [accum_tensor, accum_position, accum_points]

        step_index += 1
        state[6] = step_index
        done = False
        
        if step_index == len(df):
            done = True
            action_t = []
        else:
            half_kernel = 6
            x, y = s_pt
            D = state[0][-1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum()
            if D > 0:
                signal = 1
            else:
                signal = 0
            
            action_t = np.array([df.iloc[step_index]['steering'], df.iloc[step_index]['throttle'], df.iloc[step_index]['brake'],                                     signal & can_quit, can_quit])
            obs_index = int(df.iloc[step_index]['idx'])
            dyn_data = self.orientations[sim][hero][obs_index]
            yaw, x, y = dyn_data[1], dyn_data[4], dyn_data[5]
            s_pt = np.array([x - origin[0], y - origin[1]])
            angle = np.array([np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
            state[3] = angle
            state[4] = s_pt
            state[9] = [x, y]
        
        return action_t, done
    
    def fix_end_line(self, field, img, start, offset=0, step_size=1, size=2, max_it=100, kernel_size=3):
        half_kernel = kernel_size // 2
        F = np.array([field[0][start[1]-half_kernel : start[1]+half_kernel+1, start[0]-half_kernel : start[0]+half_kernel+1].mean(),                      field[1][start[1]-half_kernel : start[1]+half_kernel+1, start[0]-half_kernel : start[0]+half_kernel+1].mean()])
        d = np.linalg.norm(F)
        if d <= 1e-5:
            print('warning : end point out of range')
            return []
        
        R = F.copy()
        lpos = start.copy().astype(np.float32)
        rpos = start.copy().astype(np.float32)
        new_dir = np.zeros(2)

        for i in range(max_it):
            new_dir[0] = -F[1]
            new_dir[1] = F[0]
            norm = np.linalg.norm(new_dir)
            lpos = lpos + (new_dir * step_size) / norm
            x, y = np.around(lpos).astype(np.int32)
            if img[y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean() == 0:
                break
            D = np.array([field[0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),                          field[1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
            new_norm = np.linalg.norm(D)
            if new_norm < 1e-5 or D.dot(F) <= 0.5 * norm * new_norm:
                break
            F = D
        
        F = R
        lpos = lpos + (F * offset) / d
        l1pos = lpos + (F * size) / d

        for i in range(max_it):
            new_dir[0] = -F[1]
            new_dir[1] = F[0]
            norm = np.linalg.norm(new_dir)
            rpos = rpos - (new_dir * step_size) / norm
            x, y = np.around(rpos).astype(np.int32)
            if img[y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean() == 0:
                break
            D = np.array([field[0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),                          field[1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
            new_norm = np.linalg.norm(D)
            if new_norm < 1e-5 or D.dot(F) <= 0.5 * new_norm * norm:
                break
            F = D
        
        F = R
        rpos = rpos + (F * offset) / d
        r1pos = rpos + (F * size) / d

        return [l1pos, r1pos, rpos, lpos]
    
    def has_quit_available(self, state):
        if not state[11][2]:
            return False

        df = state[8]
        step_index = state[6]
        row = df.iloc[step_index]
        v_x, v_y = row['v_x'], row['v_y']
        speed = np.sqrt(v_x * v_x + v_y * v_y)
        if speed > self.MAX_TRY_STOP_SPEED:
            return False
        
        x, y = np.around(state[4] * self.map_ratio).astype(np.int32)
        half_kernel = 6
        D = state[0][-1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum()
        half_kernel = 1
        
        if D > 0:
            return True
        
        if state[11][0][y, x] != state[11][1]:
            return False
        
        D = np.array([state[0][1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),                      state[0][1][1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
        d = np.linalg.norm(D)
        if d < 1e-3:
            return False
        D = D / d
        angle = state[3]
        if angle.dot(D) < 0.86:
            return False
        
        return True
    
    def color_points_in_quadrilateral(self, P, img, val=1):        
        size_y = img.shape[0]
        size_x = img.shape[1]
        
        A1 = np.linalg.inv(np.vstack((P[1] - P[0], P[2] - P[0])).T)
        A2 = np.linalg.inv(np.vstack((P[3] - P[0], P[2] - P[0])).T)
        pnts = np.vstack(P).T
        
        x_max = round(min(np.max(pnts[0]) + 1, size_x))
        y_max = round(min(np.max(pnts[1]) + 1, size_y))
        x_min = round(max(np.min(pnts[0]) - 1, 0))
        y_min = round(max(np.min(pnts[1]) - 1, 0))
        b = np.zeros((2, 1))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                b[0][0] = x - P[0][0]
                b[1][0] = y - P[0][1]
                
                s, t = A1.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    img[y, x] = val
                    continue

                s, t = A2.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    img[y, x] = val



class Buffer(object):
    def __init__(self, ds, train_val_ratio=0.8):
        self.train_data = []
        self.val_data = []
        self.pointer = 0
        self.data_generator = ds
        self.ratio = train_val_ratio
        self.train_size = 0
        self.train_map = {}
        self.val_map = {}

    def save_map(self, file_name='dict'):
        with open(config.path_to_save + '/'+ file_name, 'wb') as dbfile:
            state_map = {'train' : self.train_map, 'val' : self.val_map}
            pickle.dump(state_map, dbfile)
            dbfile.close()

    def load_map(self, file_name='dict'):
        with open(config.path_to_save + '/'+ file_name, 'rb') as dbfile:
            state_map = pickle.load(dbfile)
            self.train_map = state_map['train']
            self.val_map = state_map['val']
    
    def sample_data_points(self, num_points, shuffle=True):
        dpts=0
        data = []
        self.reset()
        
        while dpts < num_points:
            state = self.data_generator.extract_trajectory()
            if not len(state):
                dpts += 1
                continue
            
            while True:
                try:
                    control, done = self.data_generator.trajectory_step(state[0])
                except:
                    traceback.print_exception(*sys.exc_info())
                    return
                if done:
                    break
                quit = self.data_generator.has_quit_available(state[0])
                data.append([[deepcopy(state[0][10]), state[0][1].clone(), quit], control, (state[0][-3], state[0][-2], state[0][-1], state[0][6])])
                dpts += 1
        
        if shuffle:
            np.random.shuffle(data)
        
        for i in range(len(data)):
            tup = data[i][-1]

            if self.train_map.__contains__(tup):
                self.train_data.append(data[i][:-1]) 
            elif self.val_map.__contains__(tup):
                self.val_data.append(data[i][:-1]) 
            else:
                istrain = np.random.choice(2, p=[1-self.ratio, self.ratio])
                if istrain:
                    self.train_map[tup] = True
                    self.train_data.append(data[i][:-1])
                else:
                    self.val_map[tup] = True
                    self.val_data.append(data[i][:-1])

        self.train_size = len(self.train_data)
        print('updated train_size %d, val_size %d' % (self.train_size, len(self.val_data)))
    
    def reset(self):
        self.train_data = []
        self.val_data = []
        self.pointer = 0
        self.train_size = 0
    
    def get_validation_data(self):
        if not len(self.val_data):
            return [], []

        return [x[0] for x in self.val_data], torch.vstack([y[1] for y in self.val_data])
    
    def get_next_batch(self, size):
        if self.train_size == 0:
            return [], []

        data_size = self.train_size
        size = size % data_size
        
        if data_size >= self.pointer + size:
            ret_data = self.train_data[self.pointer : self.pointer+size]
            self.pointer += size
        else:
            end = (self.pointer + size) % data_size 
            ret_data = self.train_data[self.pointer:] + self.train_data[:end]
            self.pointer = end
        
        return [x[0] for x in ret_data], torch.vstack([y[1] for y in ret_data])


# In[ ]:


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
        self.num_hero = num_hero
        self.min_safe_distance = 5
        self.max_heros = 3
        self.hero_list = []
        self.active_actor = []
        self.max_field_reward = 0.1
        self.min_field_reward = 0.03
        self.time_penalty = 0.02
        self.g_rw = -0.08
        self.beta = 0.5
        self.MAX_REWARD = 10
        self.MIN_REWARD = -8
        self.resolution = 0.08
        self.reward_drop_rate = self.MAX_REWARD / (max_rw_distance ** self.beta)
        self.dt = self.dgen.dt
        self.MAX_TRY_STOP_SPEED = 2
        self.dist = self.compute_shortest_distance_matrix()
        self.sync_mode = None
        self.am = None
        self.world = world
        self.client = client
        self.blueprint_library = None
        self.camera_semseg = None
        self.is_initialized = False
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
        cached_map[6] = torch.from_numpy(through_target[s_pt[1]-self.h_cache_size_y : s_pt[1]+self.h_cache_size_y+1, s_pt[0]-self.h_cache_size_x : s_pt[0]+self.h_cache_size_x+1].toarray())
        cached_map[7 : 7 + config.embedding_size] = 0
        cached_map[7 + config.embedding_size :]  = 0
        offset = s_pt.copy() - np.array([self.h_cache_size_x, self.h_cache_size_y])
        state[2] = [cached_map, offset]

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
                
                rnn_features[config.lookback - t - 1, k] = np.array([(cur[0] - prev[0]) / (self.dt * config.v_scale),                                                           (cur[1] - prev[1]) / (self.dt * config.v_scale),                                                           (cur[2] - prev[2]) / (self.dt * config.w_scale),                                                           idx])
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
            vehicle = self.world.try_spawn_actor(self.blueprint_library.filter('vehicle.audi.a2')[0], s1.transform)
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
                if config.render and len(self.active_actor) == 0:
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
        else:
            print('warning : unable to find a decent point!')
        
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
        state = self.mgen.sparse_build_multiple_area_upwards(path, max_reward=1.0, min_reward=0.3,                                                             zero_padding=max(self.h_cache_size_y, self.h_cache_size_x) + 4,                                                             retain_road_label=True)

        if len(state[-2]) != len(path):
            j = len(state[-2])
            n = len(path)
            through = True
        else:
            through = False

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

        road = state[-2][-1][0]
        origin = state[-1]
        road_segment = state[0].copy()
        state[0][state[0] > 0] = 1
        
        if len(road) > 1:
            _last = -2
        else:
            _last = -1
        
        e_pt = np.array([road[_last].transform.location.x, road[_last].transform.location.y])
        e_pt = np.around((e_pt - origin) * self.map_ratio).astype(np.int32)
        qpoints = self.dgen.fix_end_line(state[1], state[0], e_pt, offset=-7*self.window, size=7*self.window)

        if not len(qpoints):
            if len(state[-2]) > 1:
                e_pt = np.array([state[-2][-2][0][-1].transform.location.x, state[-2][-2][0][-1].transform.location.y])
                e_pt = np.around((e_pt - origin) * self.map_ratio).astype(np.int32)
                qpoints = self.dgen.fix_end_line(state[1], state[0], e_pt, offset=-5*self.window, size=7*self.window)
                through = True
            else:
                print('ERROR : failed to find end point')
                return []

        target_pass = lil_matrix(state[0].shape, dtype=np.float32)
        target_stop = lil_matrix(state[0].shape, dtype=np.float32)

        if through:
            self.dgen.color_points_in_quadrilateral(qpoints, target_pass, val=1)
        else:
            self.dgen.color_points_in_quadrilateral(qpoints, target_stop, val=1)

        state.append([target_stop, target_pass])
        ref_point = (qpoints[0] + qpoints[1] + qpoints[2] + qpoints[3]) / 4
        hero_transform = self.hero_list[-1].get_transform()
        pos_t = np.array([hero_transform.location.x, hero_transform.location.y])
        pos_t = np.around((pos_t - origin) * self.map_ratio).astype(np.int32)
        yaw = hero_transform.rotation.yaw
        angle_t = np.array([np.cos(yaw * np.pi / 180), np.cos(yaw * np.pi / 180)])
        
        start_state = [state,                      [],                      [],                      [],                      e_pt,                      origin,                      len(self.hero_list) - 1,                      [road_segment, len(path), through],                      pos_t,                      angle_t,                      0,                      id,                      ref_point]
        
        if tick_once:
            dummy_action = np.zeros((1, 4), dtype=np.float32)
            self.step([start_state], dummy_action)

        return start_state

    def step(self, state, action):
        start_state = []
        
        for i in range(len(state)):
            start_state.append(state[i][10])
        
        for i in range(len(state)):
            if state[i][10] == 2:
                continue

            if state[i][10] == 1:
                action[i,2] = 1
                continue
            
            if self.is_quit_available(state[i]):
                p = np.exp(action[i,2])
                action[i,2] = np.random.choice([0, 1], p=[1-p, p])
                action[i,3] = 1
            else:
                action[i,2] = 0
                action[i,3] = 0

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
            else:
                steer = np.clip(action[i,0], -1, 1)
                throttle = np.clip(max(action[i,1], 0), 0, 1)
                brake = np.clip(-min(action[i,1], 0), 0, 1)
                control = carla.VehicleControl()
                control.steer = float(steer)
                control.throttle = float(throttle)
                control.brake = float(brake)
                hero.apply_control(control)
                print('.........')
                print('tried : ', steer, throttle, brake)
                print('applied : ', control.steer, control.throttle, control.brake)
                print('.........')

        sensor_data = self.sync_mode.tick(timeout=2.0)
        snapshot = sensor_data[0]

        if config.render:
            image_semseg = sensor_data[1]
            col_events = sensor_data[2:]
        else:
            image_semseg = None
            col_events = sensor_data[1:]

        reward = []
        j = 0

        for i in range(len(state)):
            if state[i][10] == 2:
                reward.append(0)
                continue

            r = self.process_step(state[i], col_events[j])
            reward.append(r)
            j += 1

        for i in range(len(state)):
            if state[i][10] == 2 and start_state[i] != 2:
                self.sync_mode.remove_sensor_queue(state[i][11])
                self.hero_list[i].destroy()
        
        return reward, start_state, image_semseg

    def process_step(self, state, col_event):
        if state[10] == 2:
            return 0

        if col_event:
            impulse = col_event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            print('\n...............\nCollision with at intensity %f\n...............\n' % (intensity))
            reward = self.MIN_REWARD
            state[10] = 2
            return reward
        
        hero = self.hero_list[state[6]]
        origin = state[5]
        e_pt = state[4]
        current_transform = hero.get_transform()
        yaw = current_transform.rotation.yaw
        angle = np.array([np.cos(yaw * np.pi / 180), np.cos(yaw * np.pi / 180)])
        x, y = current_transform.location.x, current_transform.location.y
        pos_t = np.array([x, y])
        s_pt = np.around((pos_t - origin) * self.map_ratio).astype(np.int32)

        if state[10] == 1:
            velocity = hero.get_velocity()
            v_x, v_y = velocity.x, velocity.y
            speed = np.sqrt(v_x * v_x + v_y * v_y)
            half_kernel = 2
            if speed <= self.MAX_TRY_STOP_SPEED:
                state[10] = 2
                D = state[0][-1][0][s_pt[1]-half_kernel : s_pt[1]+half_kernel+1, s_pt[0]-half_kernel : s_pt[0]+half_kernel+1].sum()
                
                if D > 1:
                    reward = self.MAX_REWARD
                else:
                    ref_point = state[12]
                    goal_distance = np.linalg.norm(ref_point - s_pt) / self.map_ratio
                    reward = max(self.MAX_REWARD - self.reward_drop_rate * (goal_distance ** self.beta), 0)
                
                return reward
            else:
                cached_map, offset = state[2]
                loc_y = s_pt[1] - offset[1]
                loc_x = s_pt[0] - offset[0]
                F = np.array([cached_map[0, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean(),                              cached_map[1, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean()])
                d = np.linalg.norm(F)
                
                if d < 1e-4:
                    state[10] = 2
                    return self.MIN_REWARD
                else:
                    return 0

        through = state[7][-1]

        if through:
            D = state[0][-1][1][s_pt[1]-half_kernel : s_pt[1]+half_kernel+1, s_pt[0]-half_kernel : s_pt[0]+half_kernel+1].sum()
            if D > 1:
                reward = self.MAX_REWARD
                state[10] = 2
                return reward
        
        velocity = hero.get_velocity()
        acc = hero.get_acceleration()
        v_x, v_y = velocity.x, velocity.y
        a_x, a_y = acc.x, acc.y
        speed = np.sqrt(v_x * v_x + v_y * v_y)

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
            print('crossed boundaries !')
            state[10] = 2
            return self.MIN_REWARD
        else:
            F = F / d

        c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)
        if c_y - self.h_dynamic_size_y < 0 or c_x - self.h_dynamic_size_x < 0 or           c_y + self.h_dynamic_size_y + 1 > config.cache_size_y or           c_x + self.h_dynamic_size_x + 1 > config.cache_size_x:
            self.__create_cached_map(state, s_pt, pos_t)
            cached_map, offset = state[2]
            loc_x = s_pt[0] - offset[0]
            loc_y = s_pt[1] - offset[1]
            c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)

        R = np.array([[F[0], -F[1]], [F[1], F[0]]])
        T = np.array([loc_x - c_x + self.h_dynamic_size_x,                      loc_y - c_y + self.h_dynamic_size_y]).reshape(2, 1)
        points = np.around(R.dot(self.hero_box) + T).astype(np.int32)
        dynamic_map = state[1]
        
        if len(dynamic_map):
            dynamic_map[7+config.embedding_size:] = 0
        
        state[1] = cached_map[:, c_y-self.h_dynamic_size_y : c_y+self.h_dynamic_size_y+1, c_x-self.h_dynamic_size_x : c_x+self.h_dynamic_size_x+1]
        dynamic_map = state[1]
        hero_start = 7 + config.embedding_size
        val = torch.Tensor([v_x / config.v_scale, v_y / config.v_scale, longitudinal_acc / config.a_scale, steering]).reshape(-1, 1)
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
        
        angle_tm1 = state[9]
        s_ptm1 = state[8]
        
        distance = np.linalg.norm(s_ptm1 - s_pt)
        n = 0
        accum = 0
        
        if distance < self.resolution:
            reward = 0
        else:
            for r in np.arange(distance, 0, -self.resolution):
                alpha = r / distance
                gradient = alpha * angle + (1 - alpha) * angle_tm1
                point = alpha * s_pt + (1 - alpha) * s_ptm1
                point = np.around(point - offset).astype(np.int32)
                accum += cached_map[:2, point[1], point[0]].numpy().dot(gradient) / np.linalg.norm(gradient)
                n += 1
            reward = (accum / n) * self.resolution * self.max_field_reward
        
        reward -= self.time_penalty
        state[9] = angle
        state[8] = s_pt
        acc_total = a_x * a_x + a_y * a_y
        lateral_g = np.sqrt(max(acc_total - longitudinal_acc ** 2, 0)) / 9.81
        reward += self.g_rw * lateral_g
        
        return reward

    def is_quit_available(self, state):
        if state[7][-1]:
            return False

        hero = self.hero_list[state[6]]
        v = hero.get_velocity()
        speed = np.sqrt(v.x * v.x + v.y * v.y)
        if speed > self.MAX_TRY_STOP_SPEED:
            return False
        
        x, y = state[8]
        half_kernel = 6
        D = state[0][-1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].sum()
        if D > 0:
            return True

        if state[7][0][y, x] != state[7][1]:
            return False
        
        half_kernel = 1
        D = np.array([state[0][1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),                      state[0][1][1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
        d = np.linalg.norm(D)
        
        if d < 1e-3:
            return False
        D = D / d
        angle = state[9]
        if angle.dot(D) < 0.86:
            return False
        
        return True

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
        self.sync_mode = CarlaSyncMode(self.client, self.world, not config.render, fps=config.fps)
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


# In[ ]:

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dynamics_encoder = nn.GRU(4, config.embedding_size)
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
         
        for i in range(1, len(config.conv_size) - 1):
            self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=config.kernel_size[i-1], padding=1))
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
            self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
        
        self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=config.kernel_size[-1]))
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.3)
        self.emlp_1 = nn.Linear(2, 8)
        self.emlp_2 = nn.Linear(2, 4, bias=False)
        self.mlp_1 = nn.Linear(config.conv_size[-1], 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(76, 32)
        self.mlp_4 = nn.Linear(32, 16)
        self.mlp_5 = nn.Linear(16, 1)
        self.flatten = nn.Flatten()

    def __encode_dynamics(self, features, state):
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
                bounds = np.logical_and(np.logical_and(pos[0] >= 0, pos[0] < config.dynamic_size_x),\
                                        np.logical_and(pos[1] >= 0, pos[1] < config.dynamic_size_y))
                pos = np.unique(pos[:, bounds], axis=1)
                features[7 : 7 + config.embedding_size, pos[1], pos[0]] += hn[i]
    
    def forward(self, state):
        batch_size = len(state)
        assert batch_size
        
        if batch_size == 1:
            x = state[0][0][1].to(device)
            if not use_cuda:
                x = x.clone()
            self.__encode_dynamics(x, state[0][0][0])
            x = x.unsqueeze(0)
            action = state[0][1].unsqueeze(0).to(device)
            return self.forward_pass(x, action)
        else:
            x = torch.stack([state[i][0][1] for i in range(batch_size)], axis=0).to(device)
            for i in range(batch_size):
                self.__encode_dynamics(x[i], state[i][0][0])
            action = torch.stack([state[i][1] for i in range(batch_size)], axis=0).to(device)
            return self.forward_pass(x, action)
    
    def forward_pass(self, x, action):        
        for i in range(len(self.conv) - 1):
            x = self.conv[i](x)
            x = self.leakyRelu(x)
            x = self.pool[i](x)
            x = self.batch_norm[i](x)
        
        x = self.conv[-1](x)
        x = self.leakyRelu(x)
        x = self.flatten(x)
        
        x = self.mlp_1(x)
        x = self.leakyRelu(x)
        x = self.dropout_1(x)
        
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        
        y = self.emlp_1(action[:, :2])
        e = self.emlp_2(action[:, 2:])
        z = torch.cat([x, y, e], dim=-1)

        z = self.mlp_3(z)
        z = self.leakyRelu(z)

        z = self.mlp_4(z)
        z = self.leakyRelu(z)
        z_ret = self.mlp_5(z)
        
        return z_ret.squeeze(-1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.dynamics_encoder = nn.GRU(4, config.embedding_size)
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        
        for i in range(1, len(config.conv_size) - 1):
            self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=config.kernel_size[i-1], padding=1))
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
            self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
        
        self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=config.kernel_size[-1]))
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.3)
        
        self.mlp_1 = nn.Linear(config.conv_size[-1], 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(64, 32)
        self.mlp_4 = nn.Linear(32, 16)
        self.mlp_5 = nn.Linear(16, 1)
        self.flatten = nn.Flatten()
    
    def __encode_dynamics(self, features, state):
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
                bounds = np.logical_and(np.logical_and(pos[0] >= 0, pos[0] < config.dynamic_size_x),                                        np.logical_and(pos[1] >= 0, pos[1] < config.dynamic_size_y))
                pos = np.unique(pos[:, bounds], axis=1)
                features[7 : 7 + config.embedding_size, pos[1], pos[0]] += hn[i]
    
    def forward(self, state):
        batch_size = len(state)
        assert batch_size
        
        if batch_size == 1:
            x = state[0][0][1].to(device)
            if not use_cuda:
                x = x.clone()
            self.__encode_dynamics(x, state[0][0][0])
            x = x.unsqueeze(0)
            return self.forward_pass(x)
        else:
            x = torch.stack([state[i][0][1] for i in range(batch_size)], axis=0).to(device)
            for i in range(batch_size):
                self.__encode_dynamics(x[i], state[i][0][0])
            return self.forward_pass(x)
    
    def forward_pass(self, x):        
        for i in range(len(self.conv) - 1):
            x = self.conv[i](x)
            x = self.leakyRelu(x)
            x = self.pool[i](x)
            x = self.batch_norm[i](x)
        
        x = self.conv[-1](x)
        x = self.leakyRelu(x)
        x = self.flatten(x)
        
        x = self.mlp_1(x)
        x = self.leakyRelu(x)
        x = self.dropout_1(x)
        
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        
        x = self.mlp_3(x)
        x = self.leakyRelu(x)

        x = self.mlp_4(x)
        x = self.leakyRelu(x)

        x_ret = self.mlp_5(x)
        
        return x_ret.squeeze(-1)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.dynamics_encoder = nn.GRU(4, config.embedding_size)
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        
        for i in range(1, len(config.conv_size) - 1):
            self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=config.kernel_size[i-1], padding=1))
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
            self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
        
        self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=config.kernel_size[-1]))
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.mlp_1 = nn.Linear(config.conv_size[-1], 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(64, 32)
        self.mlp_4 = nn.Linear(32, 16)
        self.mlp_5 = nn.Linear(16, 4)
        self.mlp_6 = nn.Linear(32, 8)
        self.mlp_7 = nn.Linear(8, 1)

        self.flatten = nn.Flatten()
        self.log_sigmoid = nn.LogSigmoid()

    def __encode_dynamics(self, features, state):
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
                bounds = np.logical_and(np.logical_and(pos[0] >= 0, pos[0] < config.dynamic_size_x),                                        np.logical_and(pos[1] >= 0, pos[1] < config.dynamic_size_y))
                pos = np.unique(pos[:, bounds], axis=1)
                features[7 : 7 + config.embedding_size, pos[1], pos[0]] += hn[i]
    
    def forward(self, state, return_embeddings=False):
        batch_size = len(state)
        assert batch_size
        
        if batch_size == 1:
            x = state[0][0][1].to(device)
            if not use_cuda:
                x = x.clone()
            self.__encode_dynamics(x, state[0][0][0])
            x = x.unsqueeze(0)
            if return_embeddings:
                return x, self.forward_pass(x)
            else:
                return self.forward_pass(x)
        else:
            x = torch.stack([state[i][0][1] for i in range(batch_size)], axis=0).to(device)
            for i in range(batch_size):
                self.__encode_dynamics(x[i], state[i][0][0])
            return self.forward_pass(x)

    def forward_pass(self, x):        
        for i in range(len(self.conv) - 1):
            x = self.conv[i](x)
            x = self.leakyRelu(x)
            x = self.pool[i](x)
            x = self.batch_norm[i](x)
        
        x = self.conv[-1](x)
        x = self.leakyRelu(x)
        x = self.flatten(x)
        
        x = self.mlp_1(x)
        x = self.leakyRelu(x)
        x = self.dropout_1(x)
        
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        x = self.dropout_2(x)
        
        x = self.mlp_3(x)
        x_mid = self.leakyRelu(x)

        x = self.mlp_4(x_mid)
        x = self.leakyRelu(x)
        out_1 = self.mlp_5(x)
        
        mu = out_1[:, :2]
        log_sigma = torch.clamp(out_1[:, 2:], config.sigma_min, config.sigma_max)

        x = self.mlp_6(x_mid)
        x = self.leakyRelu(x)
        log_signal = self.log_sigmoid(self.mlp_7(x))
        
        return mu, log_sigma, log_signal

    def log_pi(self, state):
        mu, log_sigma, log_signal = self.forward(state)
        batch_size = len(state)
        std = torch.exp(log_sigma)
        ndist = N.Normal(mu, std)
        x = torch.stack([state[i][1][:2] for i in range(batch_size)], axis=0).to(device)
        pi = ndist.log_prob(x).sum(axis=1)
        signal_prob = torch.clamp(log_signal, config.min_p).squeeze(-1)
        
        for i in range(len(state)):
            if state[i][1][3]:
                if state[i][1][2]:
                    pi[i] += signal_prob[i]
                else:
                    pi[i] += torch.log(1 - torch.exp(signal_prob[i]))

        return pi

    def sample_from_density_fn(self, mean, log_std, quit_signal, deterministic=False):
        if deterministic:
            pi = 0
            action = torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            ndist = N.Normal(mean, std)
            action = ndist.sample()
            pi = ndist.log_prob(action).sum(axis=1)
            # action = torch.tanh(a_inv)
            # pi = ndist.log_prob(a_inv) - (2 * (np.log(2) - a_inv - F.softplus(-2 * a_inv))).sum(axis=1)

        signal_prob = torch.clamp(quit_signal, config.min_p, config.max_p).squeeze(-1)
        return action, pi, signal_prob
    
    def get_action(self, state, mean, log_sigma, log_signal, deterministic=False):
        allowed = torch.Tensor([float(state[i][-1]) for i in range(len(mean))]).to(device)
        if torch.sum(allowed) > 0:
            p = allowed * torch.exp(log_signal.squeeze(-1))
        else:
            p = []
        
        if not deterministic:
            log_std = torch.clamp(log_sigma, config.sigma_min, config.sigma_max)
            std = torch.exp(log_std)
            ndist = N.Normal(mean, std)
            intend = ndist.rsample()
            action = torch.tanh(intend)
        else:
            action = torch.tanh(mean)
        
        return action, p
    
    def get_sample_action_with_entropy(self, state, mean, log_sigma, log_signal, deterministic=True):
        if not deterministic:
            log_std = torch.clamp(log_sigma, config.sigma_min, config.sigma_max)
            std = torch.exp(log_std)
            
            if not state[-1]:
                logp = -np.log(2)
                ndist_1 = N.Normal(mean[0], std[0])
                ndist_2 = N.Normal(mean[1], std[1])
                ndist_3 = N.Normal(mean[2], std[2])
                action_1 = ndist_1.rsample()
                action_2 = ndist_2.rsample()
                action_3 = ndist_3.rsample()
                if use_cuda:
                    action = torch.zeros(4).cuda()
                else:
                    action = torch.zeros(4)
                
                if action_2 >= action_3:
                    action[0] = torch.tanh(action_1)
                    action[1] = (torch.tanh(action_2) + 1) / 2
                    action[2] = 0
                    entropy = ndist_1.entropy() + ndist_2.entropy() - logp
                else:
                    action[0] = torch.tanh(action_1)
                    action[2] = (torch.tanh(action_3) + 1) / 2
                    action[1] = 0
                    entropy = ndist_1.entropy() + ndist_3.entropy() - logp
            else:
                logp_flag = torch.clamp(log_signal, config.min_p)
                p_flag = torch.exp(logp_flag)
                B = bn.Bernoulli(p_flag[0])
                sample = B.sample()
                if sample == 0:
                    logp = torch.log(1 - p_flag[0])
                    ndist_1 = N.Normal(mean[0], std[0])
                    ndist_2 = N.Normal(mean[1], std[1])
                    ndist_3 = N.Normal(mean[2], std[2])
                    action_1 = ndist_1.rsample()
                    action_2 = ndist_2.rsample()
                    action_3 = ndist_3.rsample()
                    if use_cuda:
                        action = torch.zeros(4).cuda()
                    else:
                        action = torch.zeros(4)
                    
                    if action_2 >= action_3:
                        action[0] = torch.tanh(action_1)
                        action[1] = (torch.tanh(action_2) + 1) / 2
                        action[2] = 0
                        entropy = ndist_1.entropy() + ndist_2.entropy() + B.entropy()
                    else:
                        action[0] = torch.tanh(action_1)
                        action[2] = (torch.tanh(action_3) + 1) / 2
                        action[1] = 0
                        entropy = ndist_1.entropy() + ndist_3.entropy() + B.entropy()
                else:
                    logp = B.log_prob(sample)
                    if use_cuda:
                        action = torch.Tensor(4).uniform_(-1, 1).cuda()
                    else:
                        action = torch.Tensor(4).uniform_(-1, 1)
                    action[3] = 1
                    entropy = B.entropy() + 3 * np.log(2)
        else:
            if use_cuda:
                action = torch.zeros(4).cuda()
            else:
                action = torch.zeros(4)
            
            action[0] = mean[0]
            if mean[1] >= mean[2]:
                action[1] = mean[1]
            else:
                action[2] = mean[2]
            
            if state[-1] and log_signal[0] > -np.log(2):
                action[3] = 1
            logp = 0
            entropy = 0
        
        return action, logp, entropy
    
    # use it for having seperate output node for brake and throttle with minimum between the two of them always zero(as seen in expert data)
    # requires env or dataset access to further divide density function on basis of full state and log_signal 
    def get_sample_log_density_1(self, env, state, mean, log_sigma, log_signal, deterministic=True):
        if not deterministic:
            log_std = torch.clamp(log_sigma, config.sigma_min, config.sigma_max)
            std = torch.exp(log_std)
            if env.is_quit_available(state):
                logp = -np.log(2)
                ndist_1 = N.Normal(mean[0], std[0])
                ndist_2 = N.Normal(mean[1], std[1])
                ndist_3 = N.Normal(mean[2], std[2])
                action_1 = ndist_1.rsample()
                action_2 = ndist_2.rsample()
                action_3 = ndist_3.rsample()
                action = torch.zeros(4)
                a_inv = torch.zeros(3)
                if action_2 >= action_3:
                    a_inv[0] = action_1
                    a_inv[1] = action_2
                    action[0] = torch.tanh(action_1)
                    action[1] = (torch.tanh(action_2) + 1) / 2
                    action[2] = 0
                    logp_y = ndist_1.log_prob(a_inv[0]) + ndist_2.log_prob(a_inv[1]) + torch.log(ndist_3.cdf(a_inv[1]))
                    logp += logp_y - 2 * (2 * np.log(2) - a_inv[0] - F.softplus(-2 * a_inv[0]) - a_inv[1] - F.softplus(-2 * a_inv[1]))
                else:
                    a_inv[0] = action_1
                    a_inv[2] = action_3
                    action[0] = torch.tanh(action_1)
                    action[2] = (torch.tanh(action_3) + 1) / 1.85
                    action[1] = 0
                    logp_y = ndist_1.log_prob(a_inv[0]) + ndist_3.log_prob(a_inv[2]) + torch.log(ndist_2.cdf(a_inv[2])) 
                    logp += logp_y - 2 * (2 * np.log(2) - a_inv[0] - F.softplus(-2 * a_inv[0]) - a_inv[2] - F.softplus(-2 * a_inv[2]))
            else:
                logp_flag = torch.clamp(log_signal, config.min_p)
                p_flag = torch.exp(logp_flag)
                B = bn.Bernoulli(p_flag[0])
                sample = B.sample()
                if sample == 0:
                    logp = torch.log(1 - p_flag[0])
                    ndist_1 = N.Normal(mean[0], std[0])
                    ndist_2 = N.Normal(mean[1], std[1])
                    ndist_3 = N.Normal(mean[2], std[2])
                    action_1 = ndist_1.rsample()
                    action_2 = ndist_2.rsample()
                    action_3 = ndist_3.rsample()
                    action = torch.zeros(4)
                    a_inv = torch.zeros(3)
                    if action_2 >= action_3:
                        a_inv[0] = action_1
                        a_inv[1] = action_2
                        action[0] = torch.tanh(action_1)
                        action[1] = (torch.tanh(action_2) + 1) / 2
                        action[2] = 0
                        logp_y = ndist_1.log_prob(a_inv[0]) + ndist_2.log_prob(a_inv[1]) + torch.log(ndist_3.cdf(a_inv[1]))
                        logp += logp_y - 2 * (2 * np.log(2) - a_inv[0] - F.softplus(-2 * a_inv[0]) - a_inv[1] - F.softplus(-2 * a_inv[1]))
                    else:
                        a_inv[0] = action_1
                        a_inv[2] = action_3
                        action[0] = torch.tanh(action_1)
                        action[2] = (torch.tanh(action_3) + 1) / 1.85
                        action[1] = -1
                        logp_y = ndist_1.log_prob(a_inv[0]) + ndist_3.log_prob(a_inv[2]) + torch.log(ndist_2.cdf(a_inv[2]))
                        logp += logp_y - 2 * (2 * np.log(2) - a_inv[0] - F.softplus(-2 * a_inv[0]) - a_inv[2] - F.softplus(-2 * a_inv[2]))
                else:
                    logp = B.log_prob(sample) - 3 * np.log(2)
                    action = torch.Tensor(4).uniform_(-1, 1)
                    action[3] = 1
        else:
            action = torch.zeros(4)
            action[0] = mean[0]
            if mean[1] >= mean[2]:
                action[1] = mean[1]
            else:
                action[2] = mean[2]
            
            if env.has_quit_available(state) and log_signal[0] > -np.log(2):
                action[3] = 1
            logp = 0
        
        return action, logp


# In[ ]:


def load_models(checkpoint_path, actor, critic, discriminator, actor_optimizer, critic_optimizer, discriminator_optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['actor_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['num_trajectory'], checkpoint['num_grad_steps']


# In[ ]:


class trainUtils(object):
    def __init__(self, **kwargs):
        self.gamma = kwargs.get('gamma', 0.99)
        self._lambda = kwargs.get('_lambda', 0.95)
        self.temperature = kwargs.get('temperature', 0.01)
        self.randomized_control_std = kwargs.get('randomized_control_std', 0.01)
        self.delta = kwargs.get('delta', 0.02)
        self.cg_damping = kwargs.get('cg_damping', 0.01)
        self.max_conjugate_error = kwargs.get('max_conjugate_error', 1e-9)
        self.max_conjugate_it = kwargs.get('max_conjugate_it', 10)
        self.drop_rate = kwargs.get('drop_rate', 0.5)
        self.min_improvment = kwargs.get('min_improvment', 0.1)
        self.max_search_it = kwargs.get('max_line_search_it', 10)
        self.saved_expert_state = []
        self.game_ = []
        self.prev_not_used = []
        self.not_used = []
        self.previous_direction = None
        self.param_lengths = []
    
    def get_used_parameters(self, net):
            return [param for i, param in enumerate(net.parameters()) if not self.not_used[i]]

    def get_grad_vector(self, out, net, v, retain_graph=False, create_graph=False):
        if len(self.not_used):
            grads = torch.cat([g.flatten() for g in torch.autograd.grad(out, self.get_used_parameters(net),                               grad_outputs=v, retain_graph=retain_graph, create_graph=create_graph)])
            return grads
        else:
            param = list(net.parameters())
            grads = [g for g in torch.autograd.grad(out, param, grad_outputs=v,                                                    allow_unused=True, retain_graph=retain_graph,                                                    create_graph=create_graph)]
            self.not_used = np.array([True if g is None else False for g in grads])
            grads = torch.cat([g.flatten() for g in grads if g is not None])
        
        return grads
    
    def copy_vector(self, b):
        if all(self.prev_not_used == self.not_used):
            return self.previous_direction
        
        x = torch.zeros_like(b).to(device)
        a_ptr = 0
        b_ptr = 0
        param_lengths = self.param_lengths
        
        for i in range(len(self.not_used)):                
            if not self.not_used[i]:
                if not self.prev_not_used[i]:
                    x[a_ptr : a_ptr + param_lengths[i]] = self.previous_direction[b_ptr : b_ptr + param_lengths[i]]
                    b_ptr += param_lengths[i]
                a_ptr += param_lengths[i]   
            elif not self.prev_not_used[i]:
                b_ptr += param_lengths[i]
        
        return x

    def conjugate_solve(self, Av, b, use_previous_estimate=True):
        if not use_previous_estimate or self.previous_direction is None:
            x = torch.zeros_like(b).to(device)
            r = b
        else:
            x = self.copy_vector(b)
            r = b - Av(x)
        
        p = r
        print('p : ', p)
        print('p : ', p.requires_grad)
        print('pdtype : ', p.dtype)
        rk_norm = r.norm() ** 2

        for k in range(self.max_conjugate_it):
            Ap = Av(p)
            print('Ap : ', Ap)
            alpha = rk_norm / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rk1_norm = r.norm()
            print('rk1_norm' , rk1_norm)
            if rk1_norm < self.max_conjugate_error:
                break
            rk1_norm = rk1_norm ** 2
            beta = rk1_norm / rk_norm
            p = r + beta * p
            rk_norm = rk1_norm
        
        print('conjugate loss = ', r.norm())
        self.previous_direction = x
        self.prev_not_used = self.not_used
        return x

    def update_net(self, net, flat_params):
        end_index = 0
        
        for i, param in enumerate(net.parameters()):
            if self.not_used[i]:
                continue
            start_index = end_index
            end_index = start_index + np.prod(list(param.shape))
            param.data = flat_params[start_index:end_index].reshape(param.shape)

    def line_search(self, surrogate, KL, actor, L_t, gk, search_direction, max_step):
        beta = max_step
        param_t = torch.cat([param.flatten().detach() for i, param in enumerate(actor.parameters())                             if not self.not_used[i]])
        param = param_t

        for k in range(self.max_search_it):
            param = param + beta * search_direction
            self.update_net(actor, param)
            L_t1 = surrogate(actor)
            first_order_est = torch.dot(gk, beta * search_direction)
            print('estimate : ', first_order_est)
            
            if L_t1 > self.min_improvment * first_order_est:
                if L_t1 - L_t > 0:
                    kl_val = KL(actor)
                    print('kl : ', kl_val)
                    if kl_val < self.delta:
                        return param, L_t1 - L_t
            
            beta *= self.drop_rate

        print('search failed! returning old_param')
        return param_t, 0

    def clear_saved_expert_data(self):
        self.saved_expert_state = None

    def __train_trajectory(self, actor, critic, discriminator, critic_optimizer, discriminator_optimizer, data, normalize=True, infinite_horizon=True, use_fisher_product=True):
        batch_size = data['batch_size']
        _gamma_ = data['_gamma_']
        policy_data = data['policy_data']
        mu_t1 = data['mu_t1']
        log_var_t1 = data['log_var_t1']
        log_signal_t1 = data['log_signal_t1']
        signal_sample_idx = data['signal_sample_idx']
        log_pi_old = data['log_pi_old']
        effective_batch_size = data['effective_batch_size']
        extra_states = data['extra_states']
        trajectory_lengths = data['trajectory_lengths']
        expert_data = data['expert_data']
        
        self.not_used = []
        mu_t = mu_t1.detach()
        log_var_t = log_var_t1.detach()
        log_signal_t = log_signal_t1.detach()
        print(effective_batch_size, len(policy_data), int(not infinite_horizon), batch_size)
        self.param_lengths = [np.prod(param.shape) for param in actor.parameters()]
        assert effective_batch_size == len(policy_data)
        get_log_probs = nn.LogSigmoid()
        discriminator.eval()
        critic.eval()
        
        with torch.no_grad():
            if infinite_horizon:
                values = critic.forward(policy_data)
                extra_values = None
            else:
                assert len(extra_states)
                values = critic.forward(policy_data + extra_states)
                extra_values = values[-batch_size:]
                values = values[:effective_batch_size]
            
            rewards = -get_log_probs(discriminator.forward(policy_data))
            advantage = torch.zeros(effective_batch_size).to(device)
            returns = torch.zeros(effective_batch_size).to(device)
            end = 0

            for j in range(batch_size):
                start = end
                end = start + trajectory_lengths[j]
                print(trajectory_lengths[j])

                if infinite_horizon:
                    gae_lambda = 0

                    for i in range(end-1, start-1, -1):
                        gae_0 = rewards[i] + (self.gamma * values[i+1] if i < end - 1 else 0) - values[i]
                        gae_lambda = gae_0 + self.gamma * self._lambda * gae_lambda
                        advantage[i] = gae_lambda
                        returns[i] = gae_lambda + values[i]
                else:
                    running_lambda = self._lambda
                    returns[end - 1] = rewards[end - 1] + self.gamma * extra_values[j]
                    advantage[end - 1] = returns[end - 1] - values[end - 1]
                    gae_lambda = advantage[end - 1]

                    for i in range(end-2, start-1, -1):
                        running_lambda *= self._lambda
                        returns[i] = rewards[i] + (self.gamma / (1 - running_lambda)) * (self._lambda * gae_lambda + values[i+1] - running_lambda * returns[i+1])
                        gae_lambda = returns[i] - values[i]
                        advantage[i] = gae_lambda

            if normalize:
                advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)
        
        print('advantage : ', advantage)
        print('returns : ', returns)
        
        causal_entropy = -(_gamma_ * log_pi_old).mean()
        Lpolicy = (advantage * log_pi_old).mean() + self.temperature * causal_entropy
        gk = self.get_grad_vector(Lpolicy, actor, None, retain_graph=True)
        print('gk : ', gk)
        gk = gk.detach()
        log_pi_old = log_pi_old.detach()

        def KL_policy():
            var_t1 = torch.exp(log_var_t1)
            var_t = torch.exp(log_var_t)
            size = len(mu_t)
            kl_continious = 0.5 * ((log_var_t - log_var_t1 + (var_t1 / var_t) + torch.pow(mu_t - mu_t1, 2) / var_t).sum(-1).mean() - len(mu_t[0]))
            
            if len(log_signal_t):
                signal_t1 = torch.exp(log_signal_t1)
                signal_t = torch.exp(log_signal_t)
                kl_discrete = (signal_t1 * (log_signal_t1 - log_signal_t) + (1 - signal_t1) * (torch.log(1 - signal_t1) - torch.log(1 - signal_t))).sum() / size
            else:
                kl_discrete = 0

            return kl_continious + kl_discrete

        def get_grad_kl(kl_policy):
            return self.get_grad_vector(kl_policy, actor, None, create_graph=True, retain_graph=True)

        def surrogate(net):
            with torch.no_grad():
                log_pi = net.log_pi(policy_data)
                return (torch.exp(log_pi - log_pi_old) * advantage).mean().item()

        def KL_eval(actor):
            with torch.no_grad():
                eval_mu_t, eval_log_sigma_t, eval_log_signal_t = actor.forward(policy_data)
                eval_log_signal_t = eval_log_signal_t[signal_sample_idx]
                size = len(mu_t)
                eval_log_var_t = 2 * eval_log_sigma_t
                eval_var_t = torch.exp(eval_log_var_t)
                var_t = torch.exp(log_var_t)
                
                kl_continious = 0.5 * ((log_var_t - eval_log_var_t + (eval_var_t / var_t) + torch.pow(mu_t - eval_mu_t, 2) / var_t).sum(-1).mean() - len(mu_t[0]))
                
                if len(log_signal_t):
                    eval_signal_t = torch.exp(eval_log_signal_t)
                    signal_t = torch.exp(log_signal_t)
                    kl_discrete = (eval_signal_t * (eval_log_signal_t - log_signal_t) + (1 - eval_signal_t) * (torch.log(1 - eval_signal_t) - torch.log(1 - signal_t))).sum() / size
                else:
                    kl_discrete = 0

                return kl_continious + kl_discrete
        
        def Hv(v):
            return self.get_grad_vector(torch.dot(d_kl, v), actor, None, retain_graph=True) + self.cg_damping * v
        
        def create_stacked_parameters():
            outs = torch.cat([mu_t1.view(-1), log_var_t1.view(-1) / 2, log_signal_t1])
            signal_t = torch.exp(log_signal_t)
            var_t = torch.exp(log_var_t)
            diag_param_hessian = torch.cat([(1 / var_t).view(-1), 2 * torch.ones(np.prod(var_t.shape)).to(device), signal_t / (1 - signal_t)])
            return outs, diag_param_hessian
        
        def save_jvp_graph():
            self.const_v = torch.zeros_like(flat_out).to(device).requires_grad_()
            self.const_vjp = self.get_grad_vector(flat_out, actor, self.const_v, create_graph=True, retain_graph=True)
        
        def compute_fisher_product(v):
            jvp = torch.autograd.grad(self.const_vjp, self.const_v, grad_outputs=v, retain_graph=True)[0]
            Mjvp = diag_kl_hessian * jvp
            return self.get_grad_vector(flat_out, actor, Mjvp, retain_graph=True) / effective_batch_size + self.cg_damping * v
        
        diag_kl_hessian = None
        flat_out = None
        
        if not use_fisher_product:
            kl_policy = KL_policy()
            print('kl_policy : ', kl_policy)
            print(kl_policy.dtype)
            d_kl = get_grad_kl(kl_policy)
            print('d_kl : ', d_kl)  
            print(d_kl.dtype)
            s = self.conjugate_solve(Hv, gk)
        else:
            flat_out, diag_kl_hessian =  create_stacked_parameters()
            print('flat_outs : ', flat_out)
            print('diag_kl_hessian : ', diag_kl_hessian)
            save_jvp_graph()
            s = self.conjugate_solve(compute_fisher_product, gk)
        
        print('s : ', s)
        max_step = torch.sqrt(2 * self.delta / torch.dot(gk.T, s))
        if use_fisher_product:
            del self.const_vjp
            del self.const_v
        else:
            del d_kl
        
        print('max_step : ', max_step)
        L_0 = 0 if normalize else torch.mean(advantage).item()
        param, L_n = self.line_search(surrogate, KL_eval, actor, L_0, gk, s, max_step)
        print('L_n :', L_n)
        self.update_net(actor, param)

        mse_loss = nn.MSELoss()
        critic.train()
        critic_optimizer.zero_grad()
        values = critic.forward(policy_data)
        value_loss = mse_loss(values, returns)
        value_loss.backward()
        critic_optimizer.step()

        discriminator.train()
        discriminator_optimizer.zero_grad()
        expert_logits = discriminator.forward(expert_data)
        policy_logits = discriminator.forward(policy_data)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.zeros(len(expert_data)).to(device))
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.ones(len(policy_data)).to(device))
        discriminator_loss = expert_loss + policy_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        return L_n, discriminator_loss.item(), value_loss.item() 


    def train_broken_trajectory(self, env, expert, actor, critic, discriminator, critic_optimizer, discriminator_optimizer,                                max_expert_trajectory_length=32, max_policy_trajectory_length=32, max_attempt=3,                                max_gradient_steps=1, expert_save=True, normalize=True, game=[]):
        policy_state = []
        
        for j in range(max_attempt):
            game = env.start_new_game(max_num_roads=np.random.choice([3, 4, 5]), tick_once=True)
            if len(game):
                self.game_ = game
                policy_state.append(game)
                break
        if not len(policy_state):
            print('failed to start a new game')
            return -1

        if not len(self.saved_expert_state):
            expert_state = expert.extract_trajectory()
        else:
            expert_state = self.saved_expert_state

        gradient_steps = 0
        done = False
        expert_done = False
        sum_pi = 0
        sum_l = 0
        sum_v = 0

        while gradient_steps < max_gradient_steps and not done:
            timestep = 0
            policy_data = []
            _gamma_ = []
            mu_t = []
            log_sigma_t = []
            log_signal_t = []
            signal_sample_idx = []
            effective_batch_size = 0
            log_pi_old = []
            data = {}

            while timestep < max_policy_trajectory_length + 1 and not done:
                batch_action = np.zeros((1, 4), dtype=np.float32)

                if policy_state[0][10] == 0:
                    if timestep == max_policy_trajectory_length:
                        with torch.no_grad():
                            mu, log_sigma, signal = actor.forward([[[policy_state[0][3], policy_state[0][1]]]])
                            action, pi, log_signal = actor.sample_from_density_fn(mu, log_sigma, signal)
                    else:
                        mu, log_sigma, signal = actor.forward([[[policy_state[0][3], policy_state[0][1]]]])
                        action, pi, log_signal = actor.sample_from_density_fn(mu, log_sigma, signal)
                        mu_t.append(mu[0])
                        log_sigma_t.append(log_sigma[0])
                        effective_batch_size += 1
                    
                    batch_action[0, :-1] = np.concatenate([action.detach().cpu().squeeze(0).numpy(), log_signal.detach().cpu().numpy()], axis=-1)

                with torch.no_grad():
                    if policy_state[0][10] == 0:
                        policy_data.append([[deepcopy(policy_state[0][3]), policy_state[0][1].clone()], None, None])
                    
                    step_reward, start_state, image_semseg = env.step(policy_state, batch_action)
                    print('step_reward : ', step_reward)
                    
                    if start_state[0] == 0:
                        policy_data[-1][1] = torch.from_numpy(batch_action[0])
                        policy_data[-1][2] = step_reward[0]
                    elif start_state[0] == 1:
                        policy_data[-1][3] += step_reward[0]

                    if config.render:
                        env.render(image_semseg)
                
                if start_state[0] == 0 and timestep < max_policy_trajectory_length:
                    if batch_action[0][-1]:
                        if batch_action[0][-2]:
                            log_signal_t.append(log_signal[0])
                            pi[0] += log_signal_t[-1]
                        else:
                            log_signal_t.append(torch.log(1 - torch.exp(log_signal[0])))
                            pi[0] += log_signal_t[-1]
                        
                        signal_sample_idx.append(len(policy_data) - 1)
                    
                    log_pi_old.append(pi[0])
                    _gamma_.append(self.gamma ** timestep)

                done = policy_state[0][10] != 0
                timestep += 1

            expert_timestep = 0
            expert_data = []

            while expert_timestep < max_expert_trajectory_length:
                if expert_done:
                    expert_state = expert.extract_trajectory()
                    expert_done = False

                while not expert_done and expert_timestep < max_expert_trajectory_length:
                    try:
                        control, expert_done = expert.trajectory_step(expert_state)
                    except:
                        traceback.print_exception(*sys.exc_info())
                        expert_done = True
                        break

                    if not len(control):
                        expert_done = True
                        break

                    action = torch.zeros(4)
                    action[0] = control[0]
                    action[1] = np.clip(control[1] - (control[2] * config.brake_scale) + np.random.normal(scale=self.randomized_control_std), -1, 1)
                    action[2] = control[3]
                    action[3] = control[4]
                    print('expert_action : ' , action)
                    expert_data.append([[deepcopy(expert_state[10]), expert_state[1].clone()], action])

                    expert_timestep += 1

            if timestep > max_policy_trajectory_length:
                infinite_horizon = False
                data['extra_states'] = [policy_data[-1]]
                policy_data.pop()
            else:
                infinite_horizon = True
                data['extra_states'] = []

            batch_size = 1
            trajectory_lengths = [len(policy_data)]
            _gamma_ = torch.Tensor(_gamma_).to(device)
            mu_t1 = torch.stack(mu_t)
            log_var_t1 = 2 * torch.stack(log_sigma_t)
            log_signal_t1 = torch.stack(log_signal_t) if len(log_signal_t) else torch.Tensor([]).to(device)
            log_pi_old = torch.stack(log_pi_old)
            
            print('log_signal_t1 : ' , log_signal_t1)
            print('log_pi_old : ' , log_pi_old)
            print('log_var_t1 : ', log_var_t1)
            print('mu_t1 : ', mu_t1)

            data['batch_size'] = batch_size
            data['_gamma_'] = _gamma_
            data['policy_data'] = policy_data
            data['mu_t1'] = mu_t1
            data['log_var_t1'] = log_var_t1
            data['log_signal_t1'] = log_signal_t1
            data['signal_sample_idx'] = signal_sample_idx
            data['log_pi_old'] = log_pi_old
            data['effective_batch_size'] = effective_batch_size
            data['trajectory_lengths'] = trajectory_lengths
            data['expert_data'] = expert_data
            
            print(infinite_horizon, effective_batch_size, len(policy_data), len(data['extra_states']))

            policy_improvement, discriminator_loss, value_loss = self.__train_trajectory(actor, critic, discriminator, critic_optimizer, discriminator_optimizer,                                                                                         data, normalize=normalize, infinite_horizon=infinite_horizon)

            sum_pi += policy_improvement
            sum_l += discriminator_loss
            sum_v += value_loss
            gradient_steps += 1

            del data
            del log_var_t1
            del log_signal_t1
            del log_pi_old

        env.reset()

        if not expert_done and expert_save:
            self.saved_expert_state = expert_state
        else:
            self.saved_expert_state = []

        return gradient_steps, sum_pi, sum_l / gradient_steps, sum_v / gradient_steps
    

    def train(self, env, actor, critic, discriminator, critic_optimizer, discriminator_optimizer, num_expert_trajectory=1,              num_policy_trajectoy=1, max_trajectory_length=1000, max_expert_trajectory_length=1000, max_attempt=3, normalize=True):
        policy_state = []
        
        for i in range(max_num_policy_trajectoy):
            for j in range(max_attempt):
                game = env.start_new_game(max_num_roads=np.random.choice([3, 4, 5]))
                if len(game):
                    policy_state.append(game)
                    break

        if not len(policy_state):
            print('failed to start a new game')
            return -1

        batch_size = len(policy_state)
        timestep = 0
        self.clear_saved_expert_data()

        policy_data = [[] for _ in range(batch_size)]
        mu_t = [[] for _ in range(batch_size)]
        log_sigma_t = [[] for _ in range(batch_size)]
        log_signal_t = [[] for _ in range(batch_size)]
        signal_sample_idx = [[] for _ in range(batch_size)]
        log_pi_old = [[] for _ in range(batch_size)]
        _gamma_ = [[] for _ in range(batch_size)]
        effective_batch_size = 0
        count = 0

        while timestep < max_trajectory_length and count < batch_size:
            this_batch = []
            index = []
            batch_action = np.zeros((batch_size, 4), dtype=np.float32)

            for i in range(batch_size):
                if policy_state[i][10] == 0:
                    this_batch.append(policy_state[i])
                    index.append(i)

            pi = None
            log_signal = None

            if len(index):
                effective_batch_size += 1
                mu, log_sigma, signal = actor.forward(this_batch)
                action, pi, log_signal = actor.sample_from_density_fn(mu, log_sigma, signal)
                batch_action[:,:-1][index] = np.concatenate([action.detach().cpu().numpy(), log_signal.detach().cpu().numpy()], axis=-1)

            with torch.no_grad():
                step_rewards, start_state, image_semseg = env.step(policy_state, batch_action)
                
                for i in range(batch_size):
                    if start_state[i] == 0:
                        policy_data[i].append([[deepcopy(policy_state[i][3]), policy_state[i][1].clone()], torch.from_numpy(batch_action[i]), (i, len(policy_data[i])),                                               step_reward[i], transitions[i][0] != transitions[i][1]])
                    
                    if start_state[i] == 1:
                        policy_data[i][-1][3] += step_reward[i]

                if config.render:
                    env.render(image_semseg)

            count = 0
            for i in range(batch_size):                
                if start_state[i] == 0:
                    if batch_action[i][-1]:
                        if batch_action[i][-2]:
                            log_signal_t[i].append(log_signal[i])
                            pi[i] += log_signal_t[i][-1]
                        else:
                            log_signal_t[i].append(torch.log(1 - torch.exp(log_signal[i])))
                            pi[i] += log_signal_t[i][-1]
                        
                        signal_sample_idx[i].append(len(policy_data[i]) - 1)
                    
                    log_pi_old[i].append(pi[i])
                    _gamma_[i].append(self.gamma ** timestep)

                if policy_state[i][10] == 2:
                    count += 1

            timestep += 1

        env.reset()

        offset = 0
        for i in range(1, batch_size):
            offset += len(policy_data[i-1])
            for j in range(len(signal_sample_idx[i])):
                signal_sample_idx[i][j] = offset + signal_sample_idx[i][j]

        expert_data = []
        for i in range(num_expert_trajectory):
            expert_state = expert.extract_trajectory()
            done = False
            data.append([])

            while not done:
                try:
                    control, done = expert.trajectory_step(expert_state)
                except:
                    traceback.print_exception(*sys.exc_info())
                    break

                if not len(control):
                    break

                action = torch.zeros(4)
                action[0] = control[0]
                action[1] = np.clip(control[1] - (control[2] * config.brake_scale) + np.random.normal(self.randomized_control_std), -1, 1)
                action[2] = control[3]
                action[3] = control[4]

                expert_data.append(([deepcopy(expert_state[3]), expert_state[1].clone()], action))

        batch_size = len(policy_state)
        signal_sample_idx = [item for sublist in signal_sample_idx for item in sublist]
        trajectory_lengths = [len(policy_data[i]) for i in range(batch_size)]
        policy_data = [item for sublist in policy_data for item in sublist]
        _gamma_ = torch.Tensor([item for sublist in _gamma_ for item in sublist]).to(device)
        mu_t1 = torch.stack([item for sublist in mu_t for item in sublist])
        log_var_t1 = torch.stack([2 * item for sublist in log_sigma_t for item in sublist])
        log_signal_t1 = torch.stack([item for sublist in log_signal_t for item in sublist])
        log_pi_old = torch.stack([item for sublist in log_pi_old for item in sublist])

        data = {}
        data['batch_size'] = batch_size
        data['_gamma_'] = _gamma_
        data['policy_data'] = policy_data
        data['mu_t1'] = mu_t1
        data['log_var_t1'] = log_var_t1
        data['log_signal_t1'] = log_signal_t1
        data['signal_sample_idx'] = signal_sample_idx
        data['log_pi_old'] = log_pi_old
        data['effective_batch_size'] = effective_batch_size
        data['trajectory_lengths'] = trajectory_lengths
        data['extra_states'] = []

        return self.__train_trajectory(actor, critic, discriminator, critic_optimizer, discriminator_optimizer, data, normalize=normalize)

