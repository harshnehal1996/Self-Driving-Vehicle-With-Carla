import numpy as np
import sys
import os
import pygame
import carla
import traceback
import cv2
import time
from carla import VehicleLightState as vls
from RL_local_planner.Config import config
import torch


class Environment(object):
    def __init__(self, num_hero, dataset, client, world, max_rw_distance=20):
        self.dgen = dataset
        self.mgen = self.dgen.mgen
        self.carla_map = self.dgen.carla_map
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
        self.random_theta_variation = 0
        self.cos_max_theta_stop = 1 / np.sqrt(2)
        self.resolution = 0.09
        self.pedestrian_safe_distance = 5 * self.map_ratio
        self.MAX_TRY_STOP_SPEED = 0.8 / 3.6
        self.MIN_TRY_STOP_SPEED = 5 / 3.6
        self.dist = self.compute_shortest_distance_matrix()
        self.SPEED_PENALTY_UPPER_LIMIT = 60 / 3.6
        self.SPEED_PENALTY_LOWER_LIMIT = 1
        self.max_ray_distance = 20 * self.map_ratio
        self.min_section_speed = 1
        self.const_dist_scale = 25
        self.stop_distance = 80
        self.start_bonus_time = 36
        self.const_timescale = self.const_dist_scale * \
            (config.fps * 1.0 / (config.tick_per_action * self.min_section_speed))
        self.reward_factor = 5
        self.rw_jerk = -0.08
        self.rw_long = -0.001
        self.max_field_reward = 0.6 / self.reward_factor
        self.boundary_cross_reward = -8 / self.reward_factor
        self.max_time_penalty = 0.1 / self.reward_factor
        self.high_speed_penalty = 1 / self.reward_factor
        self.MAX_REWARD = 15 / self.reward_factor
        self.MIN_REWARD = -10 / self.reward_factor
        self.collision_reward = -10 / self.reward_factor
        self.max_rw_speed = 14
        self.toggle_ratio = 0.5
        self.cutoff_speed = 5
        self.reward_above = 5.5
        self.brake_reward = [-0.02, -0.04]
        self.p_cross_th = 0.77
        self.positive_turnout_speed = 30
        self.reward_drop_rate = self.MAX_REWARD / \
            (max_rw_distance ** self.beta)
        self.ray_angles = np.concatenate(
            [np.arange(-90, 91, 10), np.array([120, 150, 170, 180, 190, 210, 240])])
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
        self.override_landmark_check = False
        self.throttle_map = dict([(i, y)
                                 for i, y in enumerate(config.throttle_pos)])
        self.steer_map = dict([(i, y) for i, y in enumerate(config.steer_pos)])

    def __has_affecting_landmark(self, waypoint, search_distance):
        if self.override_landmark_check:
            return False
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
            direction = np.array(
                [np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)])
            pos = np.zeros(2)
            distance = 0
            self.track_points.append([])
            while distance < self.max_ray_distance:
                self.track_points[-1].append(pos)
                pos = pos + step * direction
                distance += step
            self.track_points[-1] = np.around(
                self.track_points[-1]).astype(np.int32)
            index = np.unique(
                self.track_points[-1], axis=0, return_index=True)[1]
            self.track_points[-1] = [self.track_points[-1][i]
                                     for i in sorted(index)]
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
                road_ids[(self.top[i][0].road_id,
                          self.top[i][0].lane_id > 0)] = n
                self.keys.append(
                    (self.top[i][0].road_id, self.top[i][0].lane_id > 0))
                n += 1

            if not road_ids.__contains__((self.top[i][1].road_id, self.top[i][1].lane_id > 0)):
                road_ids[(self.top[i][1].road_id,
                          self.top[i][1].lane_id > 0)] = n
                self.keys.append(
                    (self.top[i][0].road_id, self.top[i][0].lane_id > 0))
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
            transform.rotation.yaw += np.clip(np.random.randn(), -1,
                                              1) * self.random_theta_variation
            vehicle = self.world.try_spawn_actor(
                self.blueprint_library.filter('vehicle.audi.a2')[0], transform)
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
                        self.blueprint_library.find('sensor.camera.rgb'),
                        carla.Transform(carla.Location(
                            x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                        attach_to=vehicle, attachment_type=carla.AttachmentType.SpringArm)
                    ret = self.sync_mode.add_render_queue(self.camera_semseg)
                    if ret == -1:  # stale camera
                        self.sync_mode.reset(hard_reset=False)
                        self.sync_mode.add_render_queue(self.camera_semseg)

                collision_sensor = self.world.spawn_actor(
                    self.blueprint_library.find('sensor.other.collision'),
                    carla.Transform(), attach_to=vehicle)
                self.hero_list.append(vehicle)
                id = self.sync_mode.add_sensor_queue(collision_sensor)
                # should not be done during a trajectory
                self.sync_mode.tick(timeout=2)

                return s1, id

        return None, -1

    def try_spawn_random_trajectory(self, num_roads, search_distance=30):
        if len(self.active_actor) > 0:
            if len(self.active_actor) >= self.max_heros:
                return [], -1

            w = self.get_safe_start_point(
                max(num_roads + 2, self.min_safe_distance))

            if w == -1:
                print('unable to find node with distance safe_distance=%d',
                      max(num_roads + 2, self.min_safe_distance))
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

    def color_tiles(self, state, num_tiles, first_val=0.75):
        roads = state[-2]
        origin = state[-1]
        target_stop = np.zeros(state[0].shape, dtype=np.float32)
        qpoints = None
        stop_road = -1
        counter = 0
        last_points = None
        first = True

        for i in reversed(range(len(roads))):
            left = roads[i][-1]
            right = roads[i][0]
            stop_road = i+1

            if len(left) != len(right):
                print('Warning : extra points detected l=%d, r=%d' %
                      (len(left), len(right)))
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
                x_l1 = self.map_ratio * \
                    (diff_l1 * rv_l1 +
                     np.array([g_l1.location.x, g_l1.location.y]) - origin)
                x_r1 = self.map_ratio * \
                    (diff_r1 * rv_r1 +
                     np.array([g_r1.location.x, g_r1.location.y]) - origin)
                x_l2 = self.map_ratio * \
                    (diff_l2 * rv_l2 +
                     np.array([g_l2.location.x, g_l2.location.y]) - origin)
                x_r2 = self.map_ratio * \
                    (diff_r2 * rv_r2 +
                     np.array([g_r2.location.x, g_r2.location.y]) - origin)

                try:
                    if first:
                        self.dgen.color_points_in_quadrilateral(
                            [x_l1, x_r1, x_r2, x_l2], target_stop, val=first_val)
                        first = False
                    else:
                        self.dgen.color_points_in_quadrilateral(
                            [x_l1, x_r1, x_r2, x_l2], target_stop, val=1)

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
        road_no = set()

        for i in reversed(range(last_road)):
            if i == last_road - 1:
                size = lp[1]
            else:
                size = len(roads[i][ref_lane]) - 1

            for j in reversed(range(size)):
                road_no.add(roads[i][ref_lane][j].road_id)
                loc_1 = roads[i][ref_lane][j+1].transform.location
                loc_2 = roads[i][ref_lane][j].transform.location
                delta = np.sqrt((loc_1.x - loc_2.x) ** 2 +
                                (loc_1.y - loc_2.y) ** 2)
                distance += delta
                reference_points.append(
                    [distance, np.array([loc_2.x, loc_2.y])])
                if spawn_wp.road_id == roads[i][ref_lane][j].road_id:
                    start_dist = np.sqrt(
                        (start_loc.x - loc_2.x) ** 2 + (start_loc.y - loc_2.y) ** 2)
                    if start_dist < min_dist:
                        min_dist = start_dist
                        ref_loc = len(reference_points) - 1

            if i != 0:
                loc_1 = roads[i][ref_lane][0].transform.location
                loc_2 = roads[i-1][ref_lane][-1].transform.location
                delta = np.sqrt((loc_1.x - loc_2.x) ** 2 +
                                (loc_1.y - loc_2.y) ** 2)
                distance += delta
                reference_points.append(
                    [distance, np.array([loc_2.x, loc_2.y])])
                if spawn_wp.road_id == roads[i-1][ref_lane][-1].road_id:
                    start_dist = np.sqrt(
                        (start_loc.x - loc_2.x) ** 2 + (start_loc.y - loc_2.y) ** 2)
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
                print('delta too large!!', delta, i,
                      section_status, reference_points, section_id)
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
            assert section_id[ranges[-2][0]
                              ] == len(ranges) - 2, print(section_id, len(ranges) - 2)
            section_id[ranges[-1][0]:ranges[-1][1]] = len(ranges) - 2
            normalizer[-2] += normalizer[-1]
            normalizer.pop()
            ranges[-2][1] = ref_len
            ranges.pop()

        if const_scale:
            for j in range(len(normalizer)):
                section_status[ranges[j][0]:ranges[j][1]] = normalizer[j] - \
                    section_status[ranges[j][0]:ranges[j][1]]
            section_status /= self.const_dist_scale
        else:
            for j in range(len(normalizer)):
                section_status[ranges[j][0]:ranges[j][1]] /= normalizer[j]

        normalizer = np.array(normalizer)
        timestep_per_section = np.ceil(normalizer * (config.fps * 1.0 / (
            config.tick_per_action * self.min_section_speed))).astype(np.int32)
        timestep_per_section[0] += self.start_bonus_time
        return [reference_points, section_status, section_id, timestep_per_section, road_no], ref_loc

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
            state = self.mgen.var_build_multiple_area_upwards(
                path, max_reward=1.0, min_reward=0.3, zero_padding=16, retain_road_label=False, use_sparse_mat=False)
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
        pos_t = np.array([hero_transform.location.x,
                         hero_transform.location.y])
        pos_t = pos_t - origin
        s_pt = np.around(pos_t * self.map_ratio).astype(np.int32)

        if state[-1][0][s_pt[1] - 6: s_pt[1] + 7, s_pt[0] - 6: s_pt[0] + 7].sum() > 0:
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
        angle_t = np.array(
            [np.cos(yaw * np.pi / 180), np.sin(yaw * np.pi / 180)])
        start_state = [state, None, None, [], e_pt, origin, len(self.hero_list) - 1, 0, pos_t, angle_t, 0,  id, ref_point, [
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [0, ref_loc, partitions[2][ref_loc]], partitions, [0, 0]]

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
                action[i, 2] = 1
                continue

            if not override and self.is_quit_available(state[i]):
                action[i, 2] = 1
            else:
                action[i, 2] = 0

        sensor_data = None
        for tick in range(config.tick_per_action):
            for i in range(len(state)):
                if state[i][10] == 2:
                    continue

                hero = self.hero_list[state[i][6]]
                if action[i, 2]:
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
                    target_steer = action[i, 0]
                    mix_action = action[i, 1]
                    steer = state[i][13][0][-1]
                    steer = (1 - config.steering_gain) * steer + \
                        config.steering_gain * target_steer
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

    def shoot_rays(self, state, current_transform, road_set, trace=False):
        relative_angle = self.ray_angles
        hero = self.hero_list[state[6]]
        yaw = current_transform.rotation.yaw
        absolute_angle = relative_angle + yaw
        origin = state[5]

        x, y = current_transform.location.x, current_transform.location.y
        pos_t = np.array([x, y])
        s_pt = (pos_t - origin) * self.map_ratio
        angle = np.array([np.cos(yaw * np.pi / 180),
                         np.sin(yaw * np.pi / 180)])
        alpha = 1
        half_kernel = 1

        R = np.array([[angle[0], -angle[1]], [angle[1], angle[0]]])
        T = s_pt.reshape(-1, 1)
        pos = np.around(R @ self.track_points + T).astype(np.int32)

        img = state[0][0]
        size_y, size_x = img.shape
        flat_pos = pos.transpose(1, 0, 2)
        bounds = np.logical_and(np.logical_and(flat_pos[0] >= 0, flat_pos[0] < size_x), np.logical_and(
            flat_pos[1] >= 0, flat_pos[1] < size_y))
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
                        start = np.around(
                            (state[15][0][i][1] - origin) * self.map_ratio).astype(np.int32)
                        qp = self.dgen.fix_end_line(
                            state[0][1], img, start, offset=-self.window, size=2*self.window)
                        if not len(qp):
                            continue
                        try:
                            self.dgen.color_points_in_quadrilateral(
                                qp, new_img, val=1)
                        except:
                            pass
                trace_map = np.concatenate([new_img[np.newaxis, :, :], np.zeros(
                    (2, size_y, size_x), dtype=np.float32)], axis=0)
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

        def get_crossing_probability(direction, start, max_distance=2, step_size=1):
            e_pos = start
            max_it = max(round(self.map_ratio * max_distance / step_size), 1)

            for i in range(max_it):
                e_pos = e_pos + direction * step_size
                x, y = e_pos.astype(np.int32)
                if x < 0 or x >= size_x or y < 0 or y >= size_y:
                    return 0
                D = state[0][1][:, y, x]
                d = np.linalg.norm(D)
                if d > 1e-2:
                    D = D / d
                    return 1 - abs(D.dot(direction))

            return 0

        def get_intersection_distance(boundary_points, line_angle, local_point):
            cos_t, sin_t = np.cos(line_angle), np.sin(line_angle)
            h_width = abs(boundary_points[0][0])
            h_height = abs(boundary_points[1][0])

            min_distance = np.inf
            point = [0, 0]

            if abs(sin_t) > 1e-3:
                inv_slope = cos_t / sin_t
                for y in boundary_points[1]:
                    x = local_point[0, 0] + inv_slope * (y - local_point[1, 0])
                    if abs(x) <= h_width:
                        distance = np.sqrt(
                            (x - local_point[0, 0]) ** 2 + (y - local_point[1, 0]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            point = [x, y]

            if abs(cos_t) > 1e-3:
                slope = sin_t / cos_t
                for x in boundary_points[0]:
                    y = local_point[1, 0] + slope * (x - local_point[0, 0])
                    if abs(y) <= h_height:
                        distance = np.sqrt(
                            (x - local_point[0, 0]) ** 2 + (y - local_point[1, 0]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            point = [x, y]

            if (point[0] - local_point[0, 0]) * cos_t + (point[1] - local_point[1, 0]) * sin_t >= 0:
                return min_distance
            else:
                return np.inf

            return min_distance

        delta = 100
        ret = False

        for i in range(len(self.am.np_pedestrian_objects)):
            transform = self.am.np_pedestrian_objects[i].get_transform()
            x, y = transform.location.x, transform.location.y
            vehicle_yaw = transform.rotation.yaw
            pos = [x, y, vehicle_yaw]
            x, y = (pos[0] - origin[0]) * \
                self.map_ratio, (pos[1] - origin[1]) * self.map_ratio
            if trace:
                _x, _y = np.around(x).astype(
                    np.int32), np.around(y).astype(np.int32)
                if _x >= 16 and _y >= 16 and _x < size_x - 16 and _y < size_y - 16:
                    cos_t, sin_t = np.cos(
                        pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    T = np.array([[x], [y]])
                    box = (R.dot(self.am.box[1][i]) + T).astype(np.int32)
                    trace_map[1, box[1], box[0]] = 0.2
                    trace_map[2, box[1], box[0]] = 0.8

            if (x - s_pt[0]) ** 2 + (y - s_pt[1]) ** 2 < max_squared_distance + delta:
                cos_t, sin_t = np.cos(
                    pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                T = np.array([[x], [y]])
                local_point = R.T.dot(s_pt.reshape(-1, 1) - T)
                for j in range(config.num_rays):
                    line_angle = (absolute_angle[j] - pos[2]) * np.pi / 180
                    distance = get_intersection_distance(
                        self.am.boundary_points[1][i], line_angle, local_point)
                    if distance != np.inf:
                        if dynamic_features.__contains__(j):
                            features = dynamic_features[j]
                            if features[1] < distance:
                                continue
                        else:
                            features = np.zeros(16, dtype=np.float32)
                            dynamic_features[j] = features

                        if j >= 6 and j <= 12:
                            if distance < self.pedestrian_safe_distance:
                                proximity = False
                                _x, _y = round(x), round(y)
                                if distance <= self.toggle_ratio * self.max_ray_distance:
                                    if img[_y-half_kernel: _y+half_kernel+1, _x-half_kernel: _x+half_kernel+1].sum() > 0:
                                        proximity = True
                                        ret = True

                                if proximity:
                                    mod_vel = np.sqrt(
                                        self_velocity.x ** 2 + self_velocity.y ** 2)
                                    if mod_vel >= self.MAX_TRY_STOP_SPEED:
                                        return True, True

                        if not ret and j >= 2 and j <= 16:
                            _x, _y = round(x), round(y)
                            if distance <= self.toggle_ratio * self.max_ray_distance:
                                if img[_y-half_kernel: _y+half_kernel+1, _x-half_kernel: _x+half_kernel+1].sum() > 0:
                                    ret = True
                                else:
                                    vel = self.am.np_pedestrian_objects[i].get_velocity(
                                    )
                                    velocity = np.array([vel.x, vel.y])
                                    nm = np.linalg.norm(velocity)
                                    if nm > 1e-3:
                                        search_direction = velocity / nm
                                        probability = get_crossing_probability(
                                            search_direction, np.array([_x, _y]))
                                        if probability > self.p_cross_th:
                                            ret = True

                        features[0] = 1
                        features[1] = distance
                        features[2] = i
                        features[12], features[13] = np.cos(
                            (pos[2] - yaw) * np.pi / 180), np.sin((pos[2] - yaw) * np.pi / 180)

        for i in range(len(self.am.np_vehicle_objects)):
            transform = self.am.np_vehicle_objects[i].get_transform()
            x, y = transform.location.x, transform.location.y
            vehicle_yaw = transform.rotation.yaw
            pos = [x, y, vehicle_yaw]
            x, y = (pos[0] - origin[0]) * \
                self.map_ratio, (pos[1] - origin[1]) * self.map_ratio
            if trace:
                _x, _y = np.around(x).astype(
                    np.int32), np.around(y).astype(np.int32)
                if _x >= 16 and _y >= 16 and _x < size_x - 16 and _y < size_y - 16:
                    cos_t, sin_t = np.cos(
                        pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    T = np.array([[x], [y]])
                    box = (R.dot(self.am.box[0][i]) + T).astype(np.int32)
                    trace_map[2, box[1], box[0]] = 1.0
            if (x - s_pt[0]) ** 2 + (y - s_pt[1]) ** 2 < max_squared_distance + delta:
                cos_t, sin_t = np.cos(
                    pos[2] * np.pi / 180), np.sin(pos[2] * np.pi / 180)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                T = np.array([[x], [y]])
                local_point = R.T.dot(s_pt.reshape(-1, 1) - T)
                for j in range(config.num_rays):
                    line_angle = (absolute_angle[j] - pos[2]) * np.pi / 180
                    distance = get_intersection_distance(
                        self.am.boundary_points[0][i], line_angle, local_point)
                    if distance != np.inf:
                        if dynamic_features.__contains__(j):
                            features = dynamic_features[j]
                            if features[1] < distance:
                                continue
                        else:
                            features = np.zeros(16, dtype=np.float32)
                            dynamic_features[j] = features

                        if not ret and j >= 2 and j <= 16:
                            wp = self.carla_map.get_waypoint(
                                transform.location)
                            if wp.road_id not in road_set and distance <= self.toggle_ratio * self.max_ray_distance:
                                ret = True

                        features[0] = 0
                        features[1] = distance
                        features[2] = i
                        features[12], features[13] = np.cos(
                            (pos[2] - yaw) * np.pi / 180), np.sin((pos[2] - yaw) * np.pi / 180)

        R = np.array([[angle[0], -angle[1]], [angle[1], angle[0]]])
        loc_x, loc_y = np.around(s_pt).astype(np.int32)

        for j in range(config.num_rays):
            intersection = None
            for i in range(self.points_per_ray):
                if not bounds[j, i]:
                    break

                x = flat_pos[0][j][i]
                y = flat_pos[1][j][i]
                if points[j, i] == 0:
                    if img[y-half_kernel: y+half_kernel+1, x-half_kernel: x+half_kernel+1].sum() == 0:
                        intersection = np.array([x, y])
                        break
                if state[0][-1][0][y, x] > 0 and state[0][-1][0][y, x] < 1:
                    break
                if trace:
                    if np.linalg.norm(np.array([x, y]) - s_pt) <= self.pedestrian_safe_distance:
                        trace_map[1, y, x] = 0.75
                        trace_map[2, y, x] = 0.5
                    else:
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
                    vehicle_speed = self.am.np_pedestrian_objects[index].get_velocity(
                    )
                    vehicle_acc = self.am.np_pedestrian_objects[index].get_acceleration(
                    )
                    vehicle_omega = self.am.np_pedestrian_objects[index].get_angular_velocity(
                    ).z
                    features[14] = 1
                    features[15] = 0
                else:
                    vehicle_speed = self.am.np_vehicle_objects[index].get_velocity(
                    )
                    vehicle_acc = self.am.np_vehicle_objects[index].get_acceleration(
                    )
                    vehicle_omega = self.am.np_vehicle_objects[index].get_angular_velocity(
                    ).z
                    features[14] = 0
                    features[15] = 1
                    if not ret and j >= 2 and j <= 16:
                        if features[1] < distance and features[1] <= self.toggle_ratio * self.max_ray_distance:
                            ret = True

                features[0] = steering
                features[1] /= self.max_ray_distance
                features[1] = 1 - features[1]
                features[2], features[3] = (
                    vehicle_acc.x - self_acc.x) / config.a_scale, (vehicle_acc.y - self_acc.y) / config.a_scale
                features[4], features[5] = (vehicle_speed.x - self_velocity.x) / \
                    config.v_scale, (vehicle_speed.y -
                                     self_velocity.y) / config.v_scale
                features[6], features[7] = vehicle_speed.x / \
                    config.v_scale, vehicle_speed.y / config.v_scale
                features[8], features[9] = vehicle_acc.x / \
                    config.a_scale, vehicle_acc.y / config.a_scale
                features[10], features[11] = vehicle_omega / \
                    config.w_scale,  (vehicle_omega -
                                      self_omega) / config.w_scale
                features[2:10] = features[2:10].reshape(
                    4, 2).dot(R).reshape(-1)

            static_features[j] = distance / self.max_ray_distance
            static_features[j] = 1 - static_features[j]

        potential = (state[0][1][:, loc_y, loc_x]).copy()
        d = np.linalg.norm(potential)
        if d > 1e-3:
            potential /= d
        static_features[config.num_rays], static_features[config.num_rays +
                                                          1] = self_velocity.x / config.v_scale, self_velocity.y / config.v_scale
        static_features[config.num_rays + 2], static_features[config.num_rays +
                                                              3] = self_acc.x / config.a_scale, self_acc.y / config.a_scale
        static_features[config.num_rays +
                        4], static_features[config.num_rays + 5] = potential[0], potential[1]
        static_features[config.num_rays + 6] = self_omega / config.w_scale
        static_features[config.num_rays +
                        7] = potential.dot(angle) * self.max_field_reward * d
        static_features[config.num_rays:config.num_rays +
                        6] = static_features[config.num_rays:config.num_rays + 6].reshape(3, 2).dot(R).reshape(-1)
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

        return ret, False

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
        angle = np.array([np.cos(yaw * np.pi / 180),
                         np.sin(yaw * np.pi / 180)])
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
            F = np.array([state[0][1][0][loc_y-half_kernel: loc_y+half_kernel+1, loc_x-half_kernel: loc_x+half_kernel+1].mean().item(),
                         state[0][1][1][loc_y-half_kernel: loc_y+half_kernel+1, loc_x-half_kernel: loc_x+half_kernel+1].mean().item()])
            d = np.linalg.norm(F)

            if d < 1e-3:
                print('out of range with state 1')
                state[10] = 2
                return self.MIN_REWARD
            field_vector = F / d

            if speed <= self.MAX_TRY_STOP_SPEED:
                state[10] = 2
                print('stopping.....')
                D = state[0][-1][0][s_pt[1]-half_kernel: s_pt[1]+half_kernel +
                                    1, s_pt[0]-half_kernel: s_pt[0]+half_kernel+1].sum()
                cosine = field_vector.dot(angle)

                if D > 1:
                    reward = self.MAX_REWARD * cosine
                else:
                    ref_point = state[12]
                    goal_distance = np.linalg.norm(
                        ref_point - s_pt) / self.map_ratio
                    reward = max(self.MAX_REWARD - self.reward_drop_rate *
                                 (goal_distance ** self.beta), 0) * cosine

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
            lateral_acc = -a_x * angle[1] + a_y * angle[0]
        else:
            longitudinal_acc = (a_x * v_x + a_y * v_y) / speed
            lateral_acc = (-a_x * v_y + a_y * v_x) / speed

        half_kernel = 2
        loc_y = s_pt[1]
        loc_x = s_pt[0]
        d = state[0][0][loc_y-half_kernel: loc_y+half_kernel +
                        1, loc_x-half_kernel: loc_x+half_kernel+1].sum()

        if d < 1e-4:
            state[10] = 2
            return self.boundary_cross_reward

        self.update_closest_neighbor(state, pos_t)
        reference_points = state[15][0]
        ref_index = state[14][1]
        vehicle_inrange, proximity_collision = self.shoot_rays(
            state, current_transform, state[15][4], trace)

        if proximity_collision:
            reward = self.collision_reward
            state[10] = 2
            return reward

        features = state[1]
        features[config.num_rays + 15] = speed / config.v_scale
        features[config.num_rays +
                 8] = reference_points[ref_index][0] / self.stop_distance
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

        # if state[13][1][-1] < 0:
        #     br = abs(state[13][1][-1]) / 0.9
        #     reward += (1 - br) * self.brake_reward[0] + br * self.brake_reward[1]

        if distance < self.resolution:
            reward = 0
        elif speed > self.reward_above or vehicle_inrange:
            for r in np.arange(distance, 0, -self.resolution):
                alpha = r / distance
                gradient = alpha * angle + (1 - alpha) * angle_tm1
                point = alpha * pos_t + (1 - alpha) * pos_tm1
                point = np.around(point * self.map_ratio).astype(np.int32)
                accum += state[0][1][:2, point[1], point[0]
                                     ].dot(gradient) / np.linalg.norm(gradient)

            reward = accum * self.resolution * self.max_field_reward

            if reward < -1e-3:
                state[10] = 2
                print('vehicle moving in opposite direction!')
                return self.MIN_REWARD

        if speed > self.SPEED_PENALTY_UPPER_LIMIT:
            reward -= self.high_speed_penalty

        coeff = self.max_time_penalty * 3.6 / self.positive_turnout_speed
        time_penalty = self.max_time_penalty - \
            coeff * min(speed, self.max_rw_speed)
        reward -= time_penalty
        state[9] = angle
        state[8] = pos_t

        reward += self.rw_jerk * ((lateral_acc - state[16]) * 7.5 / 90) ** 2
        reward += self.rw_long * (longitudinal_acc ** 2)
        # reward += self.rw_jerk * (lateral_acc - state[16]) ** 2
        # reward += self.rw_lat * (lateral_acc ** 2)

        if self.print_:
            print('linear reward', self.rw_long * (longitudinal_acc ** 2))
            print('lateral reward', self.rw_jerk *
                  ((lateral_acc - state[16]) * 7.5 / 90) ** 2)
            print('linear acceleration : %f' % longitudinal_acc)
            print('lateral acceleration : %f' % lateral_acc)

        state[16] = lateral_acc

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
        D = state[0][-1][0][y-half_kernel: y+half_kernel +
                            1, x-half_kernel: x+half_kernel+1].sum()
        if D >= 1:
            return True

        return False

    def draw_image(self, array, blend=False):
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        self.display.blit(image_surface, (0, 0))

    def render(self, state, reward, evaluation, pdist, cam_img=None):
        self.clock.tick()
        image = state[2].transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        if cam_img is not None:
            cam_img.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(cam_img.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (cam_img.height, cam_img.width, 4))
            array = array[:, :, :3]
            image = cv2.resize(image, (360, 360))
            array = cv2.resize(array, (1080, 1080))
            array[-360:, -360:] = image
        else:
            array = cv2.resize(image, (1080, 1080))
        self.draw_image(array)

        steer = state[13][0][-1]
        mix_action = state[13][1][-1]
        throttle = max(mix_action, 0)
        brake = -min(mix_action, 0)

        self.display.blit(
            self.font.render('%f accumulated reward' %
                             reward, True, (255, 255, 255)),
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

        return array

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
        throttle = max(state[13][1][-1], 0)
        brake = -min(state[13][1][-1], 0)

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = min(throttle + 0.01, 0.8)
        else:
            throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = min(brake + 0.2, 1)
        else:
            brake = 0

        steer_increment = milliseconds
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
        action[0][1] = throttle - brake

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
        self.sync_mode = CarlaSyncMode(
            self.client, self.world, not config.camera_render, fps=config.fps)
        self.sync_mode.enter()
        self.am = ActorManager(self.client, self.world,
                               self.traffic_manager, self.map_ratio)
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
        self.sync_mode.tick(timeout=2.0)
