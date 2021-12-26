#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import carla
import numpy as np
import random
import matplotlib.pyplot as plt
import traceback


# In[2]:


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
m = world.get_map()


# In[3]:


top = m.get_topology()


# In[182]:


class MapGenerator(object):
    def __init__(self, size_x, size_y, map_ratio):
        self.map_ratio = map_ratio
        self.size_x = size_x
        self.size_y = size_y
    
    def color_points_in_quadrilateral(self, P, img, origin, reward_field, max_reward, min_reward):
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
                    img[y][x] = 0
                    _y = y / self.map_ratio
                    _x = x / self.map_ratio
                    distance = abs((_x - x0_mid[0]) * normal_vec[0] + (_y - x0_mid[1]) * normal_vec[1])
                    reward = drop_factor * distance + max_reward
                    reward_field[y, x] = reward * tangent
                    continue

                s, t = A2.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    img[y][x] = 0
                    _y = y / self.map_ratio
                    _x = x / self.map_ratio
                    distance = abs((_x - x0_mid[0]) * normal_vec[0] + (_y - x0_mid[1]) * normal_vec[1]) 
                    reward = drop_factor * distance + max_reward
                    reward_field[y, x] = reward * tangent
            
    
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
    
    def build_multiple_area_upwards(self, waypoints, max_reward=1, min_reward=0, padding=1):
        num_roads = len(waypoints)
        offset = [self.size_x // 4, self.size_y // 4]
        plots = [None for _ in range(num_roads)]
        lanes = [None for _ in range(num_roads)]
        img = np.ones((self.size_y, self.size_x))
        reward_field = np.zeros((self.size_y, self.size_x, 2))
        
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
                        self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward)
                    
                    k += 1
                
                traversable_wp[i][j // 2] = lanes[i][j // 2][:k]
                
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
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward)
                continue
            
            for j in range(0, len(keys), 2):
                mx,mn = keys[j+1], keys[j]
                tx,tn = matches[i][mx], matches[i][mn]
                points = [np.array([plots[i+1][tn][0][0], plots[i+1][tn][1][0]]),                          np.array([plots[i+1][tx][0][0], plots[i+1][tx][1][0]]),                          np.array([plots[i][mx][0][-1], plots[i][mx][1][-1]]),                          np.array([plots[i][mn][0][-1], plots[i][mn][1][-1]])]
                self.color_points_in_quadrilateral(points, img, origin, reward_field, max_reward, min_reward)
        
        return img, reward_field, traversable_wp, origin
    
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


# In[230]:


class Environment(object):
    def __init__(self, topology, size_x, size_y, map_ratio, window=3, out_threshold=0.3):
        self.mgen = MapGenerator(size_x, size_y, map_ratio)
        self.size_x = size_x
        self.size_y = size_y
        self.size = min(size_x, size_y)
        self.top = topology
        self.map_ratio = map_ratio
        adj = {}
        for i in range(len(self.top)):
            s2 = self.top[i][0].lane_id > 0
            s1 = self.top[i][0].road_id
            if top[i][0].road_id == top[i][1].road_id:
                continue
            if adj.__contains__((s1, s2)):
                adj[(s1, s2)].append((top[i][0], top[i][1]))
            else:
                adj[(s1, s2)] = [(top[i][0], top[i][1])]
        self.keys = list(adj.keys())
        self.adj = adj
        
        self.window = window
        self.footprint = np.zeros((2*window + 1, 2*window + 1))
        for y in range(2*window+1):
            for x in range(2*window+1):
                d = np.sqrt((x - window)**2 + (y - window)**2)
                if d <= window:
                    self.footprint[y, x] = 1.0
        self.random_theta_variation = np.pi / 54
        self.max_field_reward = 0.01
        self.min_field_reward = 0.003
        self.swath_resolution = self.window / self.map_ratio
        self.out_threshold = out_threshold
        self.expected = np.sum(self.footprint)
        self.steps = 0.01
        self.s = np.arange(0, 1 + self.steps / 2, self.steps)
        self.num_points = int(1 / self.steps) + 1
        self.time_penalty = self.max_field_reward * self.num_points
        self.MIN_REWARD = -3
        self.MAX_REWARD = 5
    
    def generate_random_road_idx(self, num_roads):
        r, l = self.keys[np.random.randint(len(self.keys))]
        s1, s2 = self.adj[(r, l)][np.random.choice(len(self.adj[(r, l)]))]
        res = [s1]
        
        if num_roads == 1:
            return res
        res.append(s2)
        
        for i in range(num_roads-1):
            k1 = res[-1].road_id
            k2 = res[-1].lane_id > 0
            if self.adj.__contains__((k1, k2)):
                _, t2 = self.adj[(k1, k2)][np.random.choice(len(self.adj[(k1, k2)]))]
                res.append(t2)
            else:
                return res
        
        return res
    
    def create_new_game(self, sample_per_new_games, new_games, max_num_roads=6, trajectories=None, random=True):
        if random:
            waypoints = []
            num_roads = np.random.randint(1, max_num_roads+1, size=new_games)
            trajectories = []
            for i in range(new_games):
                trajectories.append(self.generate_random_road_idx(num_roads[i]))
        
        states = []
        for i in range(new_games):
            try:
                img, flow_field, wp, origin = self.mgen.build_multiple_area_upwards(trajectories[i],                                                                            max_reward=self.max_field_reward,                                                                            min_reward=self.min_field_reward,                                                                            padding=2)
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
                
                n = len(wp)
                for j in range(sample_per_new_games):
                    start_road = np.random.randint(0, n)
                    end_road = np.random.randint(start_road, n)
                    start_lane = np.random.randint(0, len(wp[start_road]))
                    end_lane = np.random.randint(0, len(wp[end_road]))

                    if start_road == end_road:
                        end_pos = len(wp[end_road][end_lane]) - 1
                        start_pos = 0
                        start_loc = wp[start_road][start_lane][0].transform.location
                        end_loc = wp[end_road][end_lane][-1].transform.location
                    else:
                        _n = len(wp[start_road][start_lane])
                        _m = len(wp[end_road][end_lane])
                        if _n > 1:
                            start_pos = np.random.randint(_n // 2)
                        else:
                            start_pos = 0

                        if _m > 1:
                            end_pos = _m - np.random.randint(_m // 2) - 1
                        else:
                            end_pos = 0

                        start_loc = wp[start_road][start_lane][start_pos].transform.location
                        end_loc = wp[end_road][end_lane][end_pos].transform.location

                    start_pt = np.array([start_loc.x, start_loc.y])
                    end_pt = np.array([end_loc.x, end_loc.y])
                    s_pt = (start_pt - origin) * self.map_ratio
                    e_pt = (end_pt - origin) * self.map_ratio
                    rs_pt = round(s_pt[0]), round(s_pt[1])
                    re_pt = round(e_pt[0]), round(e_pt[1])
                    features = np.zeros((3, self.size_y, self.size_y))
                    features[0] = img.copy()

                    ld = re_pt[1] - self.window
                    lu = re_pt[1] + self.window + 1
                    lr = re_pt[0] + self.window + 1
                    ll = re_pt[0] - self.window

                    height = lu - ld
                    width = lr - ll

                    g = wp[end_road][end_lane][end_pos].transform.rotation.get_forward_vector()
                    G = np.array([g.x, g.y])
                    cos_et, sin_et = G / np.linalg.norm(G)
#                     features[1, ld : lu, ll : lr] = self.footprint * cos_et
#                     features[2, ld : lu, ll : lr] = self.footprint * sin_et

                    ld = rs_pt[1] - self.window
                    lu = rs_pt[1] + self.window + 1
                    lr = rs_pt[0] + self.window + 1
                    ll = rs_pt[0] - self.window
                    height = lu - ld
                    width = lr - ll

                    g = wp[start_road][start_lane][start_pos].transform.rotation.get_forward_vector()
                    G = np.array([g.x, g.y])
                    cos_t, sin_t = G / np.linalg.norm(G)
                    add_theta = np.random.randn() * self.random_theta_variation
                    cos_p = np.cos(add_theta)
                    sin_p = np.sin(add_theta)
                    sin_rt = cos_t * sin_p + sin_t * cos_p
                    cos_rt = cos_t * cos_p - sin_t * sin_p
#                     features[3, ld : lu, ll : lr] = self.footprint * cos_rt
#                     features[4, ld : lu, ll : lr] = self.footprint * sin_rt
#                     features[5:] = flow_field.transpose(2, 0, 1) / self.max_field_reward  
                    features[1:] = flow_field.transpose(2, 0, 1) / self.max_field_reward
                    states.append([features, np.array([[cos_rt, sin_rt], [cos_et, sin_et],                                   [s_pt[0] / self.map_ratio, s_pt[1] / self.map_ratio],                                   [e_pt[0], e_pt[1]]])])#, flow_field])
            except:
                traceback.print_exception(*sys.exc_info())
                return states, trajectories[i]
        
        return states, trajectories
    
    def energy(self, a, b):
        c = - (a + b)
        return (a**2 / 7) + (b**2 / 5) + (c**2 / 3) + 2 * ((b * c / 4) + (a * b / 6) + (a * c / 5))
    
    def theta(self, x, a, b):
        c = a + b
        return (a / 4) * np.power(x, 4) + (b / 3) * np.power(x, 3) - (c / 2) * np.power(x, 2)
    
    def generate_path(self, angle, a, length):
        b = (-12 * angle - 3 * a) / 2
        val = self.theta(self.s, a, b)
        tangent_x = np.cos(val)
        tangent_y = np.sin(val)
        x = self.steps * tangent_x
        y = self.steps * tangent_y
        X = np.cumsum(2 * x) - x - x[0]
        Y = np.cumsum(2 * y) - y - y[0]
        x = X / 2
        y = Y / 2
        return [np.array([x * length, y * length]), np.vstack((tangent_x, tangent_y)),                self.energy(a, b) / length, np.cos(angle), np.sin(angle)]
    
    def step(self, state, angle, f_y, length, trace_map=[], plot_trace=False, min_length=1):        
        Map = state[0]
        start_angle = state[1]
        end_angle = state[2]
        start_pt = state[3]
        end_pt = state[4]
        flow_field = state[5]
        length += min_length
        curve, gradients, K_energy, cos_p, sin_p = self.generate_path(angle, f_y, length)
        
        cos_t = start_angle[0]
        sin_t = start_angle[1]
        x = curve[0][-1]
        y = curve[1][-1]
        x_t1 = cos_t * x - sin_t * y + start_pt[0]
        y_t1 = sin_t * x + cos_t * y + start_pt[1]
        sin_t1 = cos_t * sin_p + sin_t * cos_p
        cos_t1 = cos_t * cos_p - sin_t * sin_p
        start_angle[0] = cos_t1
        start_angle[1] = sin_t1
        
        x_t = start_pt[0]
        y_t = start_pt[1]
        start_pt[0] = x_t1
        start_pt[1] = y_t1
        
        R = np.array([[cos_t , -sin_t], [sin_t, cos_t]])
        T = np.array([[x_t], [y_t]])
        curve_point = R.dot(curve) + T
        curve_point = np.around(curve_point * self.map_ratio).astype(np.int32)
        starts = curve_point - self.window
        ends = curve_point + self.window + 1
        stop = False
        reward = 0
        
        if np.sum(ends >= self.size) or np.sum(starts < 1):
            reward += self.MIN_REWARD
            stop = True
        else:
            step_size = int(max((self.num_points - 1) * self.swath_resolution / length, 1))
            for i in range(self.num_points - 1, step_size - 1, -step_size):
                mean = np.sum(Map[0, starts[1][i] : ends[1][i], starts[0][i] : ends[0][i]] * self.footprint) / self.expected
                if mean >= self.out_threshold:
                    reward += self.MIN_REWARD
                    stop = True
                    break
        if not stop:
            gradients = R.dot(gradients).T.reshape(-1, 1)
            potentials = flow_field[curve_point[1], curve_point[0]].reshape(1, -1)
            trace_map[curve_point[1], curve_point[0]] = 1.0
            reward += potentials.dot(gradients).item()
            reward -= K_energy
            reward -= self.time_penalty
            Map[3] = Map[3] * 0
            Map[4] = Map[4] * 0
            xmap = curve_point[:, -1].reshape(-1)
            ld = xmap[1] - self.window
            lu = xmap[1] + self.window + 1
            lr = xmap[0] + self.window + 1
            ll = xmap[0] - self.window
            height = lu - ld
            width = lr - ll   
            Map[3, ld : lu, ll : lr] = self.footprint * cos_t1
            Map[4, ld : lu, ll : lr] = self.footprint * sin_t1
            
            if np.linalg.norm(xmap - end_pt) <= self.window:
                cos_diff = cos_t1 * end_angle[0] + sin_t1 * end_angle[1]
                reward += self.MAX_REWARD * cos_diff
                stop = True
        
        return reward, stop


# In[252]:


400 * 12


# In[231]:


env = Environment(top, 328, 328, 4)


# In[91]:


states, trajectories = env.create_new_game(2, 4)


# In[253]:


batches = 500
sample_per_new_games = 3
new_games = 4
traj = []
sample_per_batch = sample_per_new_games * new_games
for i in range(batches):
    print('batch %d ' % i)
    states, trajectories = env.create_new_game(sample_per_new_games, new_games)
    traj.append(trajectories)
    for j in range(sample_per_batch):
        index = i * sample_per_batch + j
        if j % sample_per_new_games == 0:
            Map = states[j][0][0]
            field = states[j][0][1:]
            tups = np.where(field != 0)
            compressed_field = np.array([*tups, field[tups]])
            cv2.imwrite('maps/' + str(index) + '.png', (Map * 255).astype(np.uint8))
            np.save('fields/' + str(index) + '.npy', compressed_field)
        np.save('init_state/' + str(index) + '.npy', states[j][1])


# In[242]:


cd state_dumps


# In[244]:


# state = env.mgen.build_multiple_area_upwards(traj[153 // sample_per_batch][(153 % sample_per_batch) // sample_per_new_games])
# Map = state[0]
# # field = state[1:]
# # tups = np.where(field != 0)
# # compressed_field = np.array([*tups, field[tups]])
# cv2.imwrite(str(1) + '.png', (Map * 255).astype(np.uint8))
# # np.save(str(1) + '.npy', compressed_field)
# np.save(str(1) + '.npy', states[j][1])


# In[245]:


# traj[200 // sample_per_batch][(200 % sample_per_batch) // sample_per_new_games]


# In[246]:


# state = env.mgen.build_multiple_upwards(traj[12][7 // sample_per_new_games])


# In[ ]:


trace_map = states[0][0][0].copy()


# In[157]:


env.step(states[0], 0, 0, 3, trace_map=trace_map, plot_trace=True)


# In[158]:


plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(trace_map, cmap='gray')


# In[137]:


plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(states[0][0][6], cmap='gray')


# In[13]:


# for i in range(100):
#     angle = np.random.randn()
#     a = np.random.uniform(-30, 30)
#     print(env.step(states[i % len(states)], 0, 0, 3))


# In[368]:


# plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
# plt.imshow(np.sqrt(np.power(states[0][0][6], 2) + np.power(states[0][0][5], 2)), cmap='gray')


# In[292]:


# plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
# plt.imshow(states[2][0][6], cmap='gray')


# In[293]:


# plt.figure(figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
# plt.imshow(img)


# In[294]:


# plt.figure(figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
# plt.imshow(image)


# In[369]:


# img, wp, origin = mgen.build_multiple_area_upwards(trajectories[0][1:3], padding=2)


# In[368]:


# image = img.copy()
# for i in range(len(wp)):
#     for j in range(len(wp[i])):
#         for k in range(len(wp[i][j])):
#             loc = wp[i][j][k].transform.location
#             start_pt = np.array([loc.x, loc.y])
#             s_pt = (start_pt - origin) * 4
#             rs_pt = round(s_pt[0]), round(s_pt[1])
#             ld = rs_pt[1] - 3
#             lu = rs_pt[1] + 3 + 1
#             lr = rs_pt[0] + 3 + 1
#             ll = rs_pt[0] - 3
#             print(wp[i][j][k].road_id, wp[i][j][k].lane_id)
#             image[ld : lu, ll : lr] = env.footprint


# In[465]:


def energy(a, b):
    c = - (a + b)
    return (a**2 / 7) + (b**2 / 5) + (c**2 / 3) + 2 * ((b * c / 4) + (a * b / 6) + (a * c / 5))
  
def theta(x, a, b):
    c = a + b
    return (a / 4) * np.power(x, 4) + (b / 3) * np.power(x, 3) - (c / 2) * np.power(x, 2)


# In[466]:


steps = 0.01
plots = []
s = np.arange(0, 1 + steps / 2, steps)

curves = [(0, 0), (0, 30), (0, -30),          (-np.pi / 6, 30), (-np.pi / 6, -30),          (-np.pi / 4, 30), (-np.pi / 4, -30),          (np.pi / 6, 30), (np.pi / 6, -30),          (np.pi / 4, 30), (np.pi / 4, -30),          (np.pi / 4, 0), (np.pi / 6, 0),          (np.pi / 4, -30), (-np.pi / 4, 0), (-np.pi / 6, 0)]

for angle, a in curves:
    b = (-12 * angle - 3 * a) / 2
    val = theta(s, a, b)
    tangent_x = np.cos(val)
    tangent_y = np.sin(val)
    x = steps * tangent_x
    y = steps * tangent_y
    X = np.cumsum(2 * x) - x - x[0]
    Y = np.cumsum(2 * y) - y - y[0]
    x = X / 2
    y = Y / 2
    plots.append([np.array([x, y]), energy(a, b), angle, np.vstack((tangent_x, tangent_y))])


# In[473]:


plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
# for i in range(len(plots)):
plt.plot(plots[3][0][0], plots[3][0][1])
plt.plot(plots[4][0][0], plots[4][0][1])
plt.plot(plots[-1][0][0], plots[-1][0][1])
plt.show()


# In[ ]:


# env.mgen.build_multiple_upwards(trajectories[1])


# In[153]:


plots = env.generate_path(0, 30, 1)


# In[151]:


# plot = env.generate_path(-1, -30, 1)


# In[154]:


plt.plot(plots[0][0], plots[0][1])
# plt.plot(plot[0][0], plot[0][1])


# In[3]:


import torch
from torch.distributions.normal import Normal


# In[8]:


x = Normal(torch.randn(5, 3), 10 + torch.randn(5, 3))


# In[10]:


z = x.rsample()


# In[12]:


x.log_prob(z).sum(axis=1)


# In[13]:


mean = torch.randn(5)
sigma = torch.ones(5)
mean.requires_grad=True
sigma.requires_grad=True


# In[14]:


mean


# In[15]:


x = Normal(mean, sigma)


# In[16]:


z = x.rsample()


# In[17]:


z


# In[18]:


z.mean().backward()


# In[23]:


sigma.grad


# In[25]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as N


# In[27]:


class config:
    img_size = 128
    actual_size = 32
    conv_size = [4, 8, 16, 32, 64, 128, 256]
    num_actions = [16, 3]


# In[30]:


# add action embedding
class QCritic(nn.Module):
    def __init__(self):
        super(QCritic, self).__init__()
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        for i in range(1, len(config.conv_size) - 1):
            self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=3, padding=1))
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))

        self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=4))
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.4)
        self.mlp_1 = nn.Linear(config.conv_size[-1], config.conv_size[-1] // 2)
        self.mlp_2 = nn.Linear(config.conv_size[-1] // 2, 32)
        self.mlp_3 = nn.Linear(32, 1)
        self.flatten = nn.Flatten()

    def forward(self, state):
        x = state[0].to(device)

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

        return x


# In[78]:


import numpy as np
import math


# In[79]:


math.log(2)


# In[42]:


t = QCritic()


# In[43]:


l = QCritic()


# In[52]:


for x, y in zip(t.parameters(), l.parameters()):
    y.data = 0.99 * y.data + 0.01 * x.data


# In[53]:


for x in l.parameters():
    print(x.data)


# In[44]:


for x in t.parameters():
    print(x.data)


# In[68]:


x = torch.zeros(5)


# In[69]:


y = torch.ones(5)


# In[70]:


x = 1 * y


# In[71]:


x[0] = 0


# In[72]:


x


# In[73]:


y


# In[ ]:




