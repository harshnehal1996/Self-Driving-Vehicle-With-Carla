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
            self.df_time = pd.DataFrame(np.array([np.array(self.orientations[d][0])[:,0],\
                                                 [i for i in range(len(self.orientations[d][0]))]]).T,\
                                                 columns=['time', 'idx'])
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
                
                rnn_features[config.lookback - t - 1, k] = np.array([(cur_x - prev_x) / (self.dt * config.v_scale),\
                                                           (cur_y - prev_y) / (self.dt * config.v_scale),\
                                                           (cur_yaw - prev_yaw) / (self.dt * config.w_scale),\
                                                           idx])
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
            state = self.mgen.sparse_build_multiple_area_upwards(path, max_reward=1.0, min_reward=0.3,\
                                                                 zero_padding=max(self.h_cache_size_y, self.h_cache_size_x) + 1,\
                                                                 retain_road_label=True)        
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
                e_pt = np.array([self.orientations[sim][hero][int(row['idx'])][4] - origin[0],\
                                 self.orientations[sim][hero][int(row['idx'])][5] - origin[1]])
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
        F = np.array([cached_map[0, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean(),\
                      cached_map[1, loc_y-half_kernel : loc_y+half_kernel+1, loc_x-half_kernel : loc_x+half_kernel+1].mean()])
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
        if c_y - self.h_dynamic_size_y < 0 or c_x - self.h_dynamic_size_x < 0 or\
           c_y + self.h_dynamic_size_y + 1 > config.cache_size_y or\
           c_x + self.h_dynamic_size_x + 1 > config.cache_size_x:
            self.__create_cached_map(state)
            cached_map, offset = state[2]
            loc_x = s_pt[0] - offset[0]
            loc_y = s_pt[1] - offset[1]
            c_x, c_y = round(self.radius * F[0] + loc_x), round(self.radius * F[1] + loc_y)
        
        R = np.array([[F[0], -F[1]], [F[1], F[0]]])
        T = np.array([loc_x - c_x + self.h_dynamic_size_x,\
                      loc_y - c_y + self.h_dynamic_size_y]).reshape(2, 1)
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
            
            action_t = np.array([df.iloc[step_index]['steering'], df.iloc[step_index]['throttle'], df.iloc[step_index]['brake'],\
                                     signal & can_quit, can_quit])
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
        F = np.array([field[0][start[1]-half_kernel : start[1]+half_kernel+1, start[0]-half_kernel : start[0]+half_kernel+1].mean(),\
                      field[1][start[1]-half_kernel : start[1]+half_kernel+1, start[0]-half_kernel : start[0]+half_kernel+1].mean()])
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
            D = np.array([field[0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),\
                          field[1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
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
            D = np.array([field[0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),\
                          field[1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
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
        
        D = np.array([state[0][1][0][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean(),\
                      state[0][1][1][y-half_kernel : y+half_kernel+1, x-half_kernel : x+half_kernel+1].mean()])
        d = np.linalg.norm(D)
        if d < 1e-3:
            return False
        D = D / d
        angle = state[3]
        if angle.dot(D) < 0.86:
            return False
        
        return True
    
    @staticmethod
    def color_points_in_quadrilateral(P, img, val=1):        
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







