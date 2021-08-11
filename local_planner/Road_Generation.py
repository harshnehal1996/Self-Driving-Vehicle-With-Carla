#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
import carla
import pygame
import numpy as np
import queue
import random
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
m = world.get_map()


# In[3]:


top = m.get_topology()


# In[22]:


class MapGenerator(object):
    def __init__(self, size_x, size_y, map_ratio):
        self.map_ratio = map_ratio
        self.size_x = size_x
        self.size_y = size_y
    
    def color_points_in_quadrilateral(self, P, img, origin):
        for i in range(4):
            P[i] = self.map_ratio * (P[i] - origin)

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
                    img[y][x] = 1.0
                    continue

                s, t = A2.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    img[y][x] = 1.0
    
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
                if lanes[i][j].road_id != waypoint.road_id:
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

        return plots
    
    def build_multiple_area_upwards(self, waypoints):
        num_roads = len(waypoints)
        offset = [self.size_x // 4, self.size_y // 4]
        plots = [None for _ in range(num_roads)]
        img = np.zeros((self.size_x, self.size_y))
        
        for i in range(num_roads):
            plots[i] = self.generate_drivable_area(waypoints[i])
        
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
                    x_req = round((x_max_ - x_min_ + 2) * self.map_ratio)
                    y_req = round((y_max_ - y_min_ + 2) * self.map_ratio)
                    
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
        
        xreq = round((x_max - x_min + 2) * self.map_ratio)
        yreq = round((y_max - y_min + 2) * self.map_ratio)
        max_offset_x = self.size_x - xreq
        max_offset_y = self.size_y - yreq
        offset[0] = min(max_offset_x, offset[0]) / self.map_ratio
        offset[1] = min(max_offset_y, offset[1]) / self.map_ratio
        origin = np.array([x_min - offset[0], y_min - offset[1]])
        
        for i in range(num_roads):
            for j in range(0, len(plots[i]), 2):
                for k in range(1, len(plots[i][j][0])):
                    points = [plots[i][j][:, k], plots[i][j+1][:, k], plots[i][j+1][:, k-1], plots[i][j][:, k-1]]
                    self.color_points_in_quadrilateral(points, img, origin)
        
        matches = []
        for i in range(num_roads - 1):
            matches.append(self.join_roads(plots[i], plots[i+1]))

        for i in range(num_roads - 1):
            keys = matches[i].keys()
            mx,mn = max(keys), min(keys)
            tx,tn = matches[i][mx], matches[i][mn]
            points = [np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                      np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]]),                      np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                      np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]])]
            self.color_points_in_quadrilateral(points, img, origin)
        
        return img
    
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
        print(M)
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
                if lanes[i][j].road_id != waypoint.road_id:
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


# In[71]:


mgen = MapGenerator(320, 320, 4)


# In[24]:


img = mgen.build_multiple_area_upwards([top[0][0], top[0][1], top[114][1], top[63][1]])


# In[25]:


plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img, cmap='gray')


# In[26]:


img = mgen.build_multiple_area_upwards([top[16][0], top[16][1], top[338][1], top[418][1]])


# In[27]:


plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img)


# In[7]:


top[20][0].road_id, top[20][1].road_id


# In[279]:


for i in range(len(top)):
    print(i, top[i][0].road_id, top[i][1].road_id)


# In[64]:


top[82][1].lane_id, top[227][0].lane_id


# In[70]:


mgen.build_multiple_upwards([top[30][0], top[30][1], top[212][1], top[457][1], top[353][1], top[82][1], top[227][0]])


# In[72]:


img = mgen.build_multiple_area_upwards([top[30][0], top[30][1], top[212][1], top[457][1], top[353][1], top[82][1], top[227][0]])


# In[73]:


plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img)


# In[ ]:




