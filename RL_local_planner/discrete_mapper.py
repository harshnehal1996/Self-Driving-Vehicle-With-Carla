# -*- coding: utf-8 -*-
"""discrete_mapper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DnWpzkUa0FcRbWUNXqIZev7xrvOoJXYY
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as N

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import sys
import traceback

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

!mkdir Data

!unzip -q "/content/drive/My Drive/RL_PF/state_dumps.zip" -d "/content/Data" && echo "success" || echo "failure"

cd /content/Data/

!nvidia-smi

def energy(a, b):
  c = - (a + b)
  return (a**2 / 7) + (b**2 / 5) + (c**2 / 3) + 2 * ((b * c / 4) + (a * b / 6) + (a * c / 5))
  
def theta(x, a, b):
  c = a + b
  return (a / 4) * np.power(x, 4) + (b / 3) * np.power(x, 3) - (c / 2) * np.power(x, 2)

steps = 0.01
plots = []
s = np.arange(0, 1 + steps / 2, steps)

# curves = [(0, 0), (0, 30), (0, -30),\
#           (-np.pi / 6, 30), (-np.pi / 6, -30),\
#           (-np.pi / 4, 30), (-np.pi / 4, -30),\
#           (np.pi / 6, 30), (np.pi / 6, -30),\
#           (np.pi / 4, 30), (np.pi / 4, -30),\
#           (np.pi / 4, 0), (np.pi / 6, 0),\
#           (np.pi / 4, -30), (-np.pi / 4, 0), (-np.pi / 6, 0),
#           (np.pi / 12, 0), (-np.pi / 12, 0)]#, (-np.pi / 12, -30)]

curves = [(-np.pi / 6, -10), (-np.pi / 6, 30),\
          (-np.pi / 9, -30), (-np.pi / 9, 30),\
          (-np.pi / 18, -30), (0, 0),\
          (0, 30), (0, -30),\
          (np.pi / 6, 10), (np.pi / 6, -30),\
          (np.pi / 9, -30), (np.pi / 9, 30),\
          (np.pi / 18, 30), (np.pi / 4, 0),\
          (-np.pi / 4, 0), (np.pi / 36, 20), (-np.pi / 36, -20)]

for angle, a in curves:
    b = (-12 * angle - 3 * a) / 2
    val = theta(s, a, b)
    x = steps * np.cos(val)
    y = steps * np.sin(val)
    X = np.cumsum(2 * x) - x - x[0]
    Y = np.cumsum(2 * y) - y - y[0]
    x = X / 2
    y = Y / 2
    plots.append([np.array([x, y]), energy(a, b), angle])

plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
for i in range(len(plots)):
  plt.plot(plots[i][0][0], plots[i][0][1])
plt.show()

class Environment(object):
    def __init__(self, size_x, size_y, map_ratio, sample_per_new_games, window=3, out_threshold=0.3):
        self.size_x = size_x
        self.size_y = size_y
        self.size = min(size_x, size_y)
        self.map_ratio = map_ratio
        
        self.window = window
        self.footprint = torch.zeros((2*window + 1, 2*window + 1))
        for y in range(2*window+1):
            for x in range(2*window+1):
                d = np.sqrt((x - window)**2 + (y - window)**2)
                if d <= window:
                    self.footprint[y, x] = 1.0
        self.random_theta_variation = np.pi / 54
        self.max_field_reward = 0.5
        self.min_field_reward = 0.15
        self.swath_resolution = self.window / self.map_ratio
        self.out_threshold = out_threshold
        self.expected = torch.sum(self.footprint).item()
        self.steps = 0.01
        self.s = np.arange(0, 1 + self.steps / 2, self.steps)
        self.num_points = int(1 / self.steps) + 1
        self.time_penalty = 0
        self.HIT_WALL_REWARD = -10
        self.WRONG_DIR_REWARD = -10
        self.MAX_REWARD = 10
        self.sample_per_new_games = sample_per_new_games
        self.Init_Data = None
        self.Road_Maps = None
        self.Flow_Fields = None
        self.hpad = 64
        self.curve_params = [(-np.pi / 6, -10), (-np.pi / 6, 30),\
          (-np.pi / 9, -30), (-np.pi / 9, 30),\
          (-np.pi / 18, -30), (0, 0),\
          (0, 30), (0, -30),\
          (np.pi / 6, 10), (np.pi / 6, -30),\
          (np.pi / 9, -30), (np.pi / 9, 30),\
          (np.pi / 18, 30), (np.pi / 4, 0),\
          (-np.pi / 4, 0), (np.pi / 36, 20), (-np.pi / 36, -20)]

        # self.curve_params = [(0, 0), (0, 30), (0, -30),\
        #   (-np.pi / 6, 30), (-np.pi / 6, -30),\
        #   (-np.pi / 4, 30), (-np.pi / 4, -30),\
        #   (np.pi / 6, 30), (np.pi / 6, -30),\
        #   (np.pi / 4, 30), (np.pi / 4, -30),\
        #   (np.pi / 4, 0), (np.pi / 6, 0),\
        #   (np.pi / 4, -30), (-np.pi / 4, 0), (-np.pi / 6, 0),
        #   (np.pi / 12, )]
        self.length_map = [1, 2, 3, 5]
    
    def fill_state_data(self, path):
        f_field = [os.path.join(path + '/fields/', files) for files in os.listdir(path + '/fields/')]
        f_init_state = [os.path.join(path + '/init_state/', files) for files in os.listdir(path + '/init_state/')]
        f_maps = [os.path.join(path + '/maps/', files) for files in os.listdir(path + '/maps/')]
        self.data_size = len(f_init_state)
        
        if self.data_size % self.sample_per_new_games != 0:
            print('ERROR : uneven sampling!')
            return
        
        self.num_img = self.data_size // self.sample_per_new_games
        if self.num_img != len(f_field) or self.num_img != len(f_maps):
            print('ERROR : missing maps or field files!')
            return
        
        self.Init_Data = np.empty((self.data_size, 4, 2))
        self.Road_Maps = np.empty((self.num_img, self.size_y, self.size_x))
        self.Flow_Fields = [None for _ in range(self.num_img)]

        for i in range(self.num_img):
            index = int(f_maps[i].split('/')[-1].split('.')[0])
            if index % self.sample_per_new_games != 0:
                print('ERROR : uneven sampling! %d' % i)
                return
            idx = index // self.sample_per_new_games
            image = cv2.imread(f_maps[i])
            self.Road_Maps[idx] =  1 - (image[:, :, 0].astype(np.float32) / 255)
        
        for i in range(self.num_img):
            index = int(f_field[i].split('/')[-1].split('.')[0])
            if index % self.sample_per_new_games != 0:
                print('ERROR : uneven sampling! %d' % i)
                return
            idx = index // self.sample_per_new_games
            field = np.zeros((2, self.size_y, self.size_x))
            arr = np.load(f_field[i])
            I, J, K = arr[:3].astype(np.int32)
            I = I + 5
            self.Flow_Fields[idx] = (I, J, K, arr[3])
        
        for i in range(self.data_size):
            index = int(f_init_state[i].split('/')[-1].split('.')[0])
            arr = np.load(f_init_state[i])
            self.Init_Data[index] = arr
    
    def expand_to_state(self, init_states, done):
        batch_size = len(init_states)
        idx = init_states[:, 0].astype(np.int32)
        Maps = self.Road_Maps[idx]

        data_index = self.sample_per_new_games * idx
        data = self.Init_Data[data_index]
        e_angles = data[:, 1]
        re_pt = np.around(data[:, 3]).astype(np.int32)
        features = torch.zeros(batch_size, 7, self.size_y, self.size_x)
        features[:, 0] = torch.from_numpy(Maps)
        
        for i in range(batch_size):
            if len(done) and done[i]:
              continue
            
            I, J, K, val = self.Flow_Fields[idx[i]]
            features[i][I, J, K] = torch.from_numpy(val.astype(np.float32))

            ld = re_pt[i][1] - self.window
            lu = re_pt[i][1] + self.window + 1
            lr = re_pt[i][0] + self.window + 1
            ll = re_pt[i][0] - self.window

            height = lu - ld
            width = lr - ll
            features[i, 1, ld : lu, ll : lr] = self.footprint * e_angles[i][0]
            features[i, 2, ld : lu, ll : lr] = self.footprint * e_angles[i][1]

            ld = int(init_states[i][4]) - self.window
            lu = int(init_states[i][4]) + self.window + 1
            lr = int(init_states[i][3]) + self.window + 1
            ll = int(init_states[i][3]) - self.window
            height = lu - ld
            width = lr - ll

            features[i, 3, ld : lu, ll : lr] = self.footprint * init_states[i][1]
            features[i, 4, ld : lu, ll : lr] = self.footprint * init_states[i][2]
        
        return features
    
    def start_new_game(self, initial_state=[], game_index=-1):
        if not len(initial_state):
            data_size = len(self.Init_Data)
            index_1 = np.random.randint(data_size) if game_index == -1 else game_index
            index_2 = index_1 // self.sample_per_new_games
            s_pt = self.Init_Data[index_1][2] * self.map_ratio
            cos_rt, sin_rt = self.Init_Data[index_1][0]
        else:
            index_1 = int(initial_state[0])
            index_2 = index_1 // self.sample_per_new_games
            s_pt = initial_state[-2:] * self.map_ratio
            cos_rt, sin_rt = initial_state[1:3]
        
        features = torch.zeros(7, self.size_y, self.size_x)
        e_pt = self.Init_Data[index_1][3]
        cos_et, sin_et = self.Init_Data[index_1][1]
        rs_pt = round(s_pt[0]), round(s_pt[1])
        re_pt = round(e_pt[0]), round(e_pt[1])
        
        features[0] = torch.from_numpy(self.Road_Maps[index_2])
        I, J, K, val = self.Flow_Fields[index_2]
        features[I, J, K] = torch.from_numpy(val.astype(np.float32))
        field = features[5:].permute(1, 2, 0).numpy()

        ld = re_pt[1] - self.window
        lu = re_pt[1] + self.window + 1
        lr = re_pt[0] + self.window + 1
        ll = re_pt[0] - self.window

        height = lu - ld
        width = lr - ll
        features[1, ld : lu, ll : lr] = self.footprint * cos_et
        features[2, ld : lu, ll : lr] = self.footprint * sin_et

        ld = rs_pt[1] - self.window
        lu = rs_pt[1] + self.window + 1
        lr = rs_pt[0] + self.window + 1
        ll = rs_pt[0] - self.window
        height = lu - ld
        width = lr - ll

        features[3, ld : lu, ll : lr] = self.footprint * cos_rt
        features[4, ld : lu, ll : lr] = self.footprint * sin_rt
        quad, points = self.fix_end_line(field, self.Road_Maps[index_2], np.around(e_pt).astype(np.int32), offset=self.window, size=3*self.window)
        features = nn.functional.pad(features, (self.hpad, self.hpad, self.hpad, self.hpad))
        dynamic_features = features[:, rs_pt[1] : rs_pt[1] + 2*self.hpad + 1, rs_pt[0] : rs_pt[0] + 2*self.hpad + 1]

        states = [dynamic_features , features, np.array([cos_rt, sin_rt]), np.array([cos_et, sin_et]),\
                 np.array([s_pt[0] / self.map_ratio, s_pt[1] / self.map_ratio]),\
                 np.array([e_pt[0], e_pt[1]]), quad, points, index_1]
        
        return states
    
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
        return [np.array([x * length, y * length]), np.vstack((tangent_x, tangent_y)),\
                self.energy(a, b) / length, np.cos(angle), np.sin(angle)]
    
    def is_inside_quadrilateral(self, P, loc):
        A1 = P[1]
        A2 = P[2]
        b = np.zeros((2, 1))
        x, y = loc
        
        b[0][0] = x - P[0][0]
        b[1][0] = y - P[0][1]
        
        s, t = A1.dot(b)
        if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
            return True
        
        s, t = A2.dot(b)
        if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
            return True
        
        return False
    
    def color_points_in_quadrilateral(self, P, img, shift=True):
        if shift:
            for i in range(4):
                P[i] = P[i] + self.hpad
        
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
                    img[y][x][1] = 1
                    continue

                s, t = A2.dot(b)
                if s[0] >= 0 and t[0] >= 0 and s[0] + t[0] <= 1:
                    img[y][x][1] = 1
    
    def fix_end_line(self, field, img, start, delta=1, offset=0, step_size=1, size=2, max_it=100):
        F = field[start[1], start[0]].copy()
        d = np.linalg.norm(F)
        if d < 1e-5:
            return [], []
        lpos = start.copy().astype(np.float32)
        rpos = start.copy().astype(np.float32)
        new_dir = np.zeros(2)

        for i in range(max_it):
            new_dir[0] = -F[1]
            new_dir[1] = F[0]
            norm = np.linalg.norm(new_dir)
            lpos = lpos + (new_dir * step_size) / norm
            x, y = np.around(lpos).astype(np.int32)
            out_of_bound = y < 0 or y >= self.size_y or x < 0 or x >= self.size_x
            if norm < 1e-5 or out_of_bound or img[y][x] == 0 or field[y][x].dot(F) <= 0:
                break
            F = field[y][x]
        F = field[start[1], start[0]].copy()
        d = np.linalg.norm(F)
      
        lpos = lpos + (F * offset) / d
        l1pos = lpos + (F * size) / d

        for i in range(max_it):
            new_dir[0] = -F[1]
            new_dir[1] = F[0]
            norm = np.linalg.norm(new_dir)
            rpos = rpos - (new_dir * step_size) / norm
            x, y = np.around(rpos).astype(np.int32)
            out_of_bound = y < 0 or y >= self.size_y or x < 0 or x >= self.size_x
            if norm < 1e-5 or out_of_bound or img[y][x] == 0 or field[y][x].dot(F) <= 0:
                break
            F = field[y][x]
        
        F = field[start[1], start[0]]
        rpos = rpos + (F * offset) / d
        r1pos = rpos + (F * size) / d
        A1 = np.linalg.inv(np.vstack((r1pos - l1pos, rpos - l1pos)).T)
        A2 = np.linalg.inv(np.vstack((lpos - l1pos, rpos - l1pos)).T)

        return [l1pos, A1, A2], [l1pos, r1pos, rpos, lpos]
    
    def step(self, state, curve_choice, length_choice, trace_map=[], plot_trace=False):        
        dynamic_map = state[0]
        static_map = state[1]
        start_angle = state[2]
        end_angle = state[3]
        start_pt = state[4]
        end_pt = state[5]
        Qpoint = state[6]

        # length += min_length + config.action_range[2]
        # length = self.length_map[length_choice]
        length = 2
        angle, f_y = self.curve_params[curve_choice]
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
        is_inside = False
        
        if np.sum(ends >= self.size) or np.sum(starts < 1):
            reward += self.HIT_WALL_REWARD
            stop = True
            return reward, stop
        else:
            step_size = int(max((self.num_points - 1) * self.swath_resolution / length, 1))
            pts = np.arange(self.num_points - 1, step_size - 1, -step_size)
            for i in reversed(pts):
                mean = torch.sum(static_map[0, starts[1][i] + self.hpad : ends[1][i] + self.hpad, starts[0][i] + self.hpad : ends[0][i] + self.hpad] * self.footprint) / self.expected
                if mean <= 1 - self.out_threshold:
                    reward += self.HIT_WALL_REWARD
                    stop = True
                    if plot_trace:
                        trace_map[curve_point[1] + self.hpad, curve_point[0] + self.hpad] = 1.0
                    return reward, stop
                xmap = curve_point[:, i]
                if np.linalg.norm(xmap - end_pt) <= 2.0 * self.window:
                    cos_diff = cos_t1 * end_angle[0] + sin_t1 * end_angle[1]
                    reward += self.MAX_REWARD * cos_diff
                    stop = True
                    break
                if len(Qpoint) != 0:
                    is_inside = self.is_inside_quadrilateral(Qpoint, xmap)
                    if is_inside:
                        reward += self.HIT_WALL_REWARD
                        stop = True
                        break

        gradients = torch.Tensor(R.dot(gradients).T.reshape(-1))
        potentials = static_map[5:].permute(1, 2, 0)[curve_point[1] + self.hpad, curve_point[0] + self.hpad].reshape(-1)
        
        if plot_trace:
            trace_map[curve_point[1] + self.hpad, curve_point[0] + self.hpad] = 1.0
        
        field_reward = length * self.max_field_reward * (potentials.dot(gradients).item()) / self.num_points

        if field_reward < 0:
            reward += self.WRONG_DIR_REWARD
            reward -= K_energy
            reward -= self.time_penalty
            return reward, True
        else:
            reward += field_reward
        
        reward -= K_energy
        reward -= self.time_penalty
        
        if stop:
          return reward, stop
        
        xmap = curve_point[:, -1]
        dynamic_map[3, self.hpad-self.window:self.hpad+self.window+1, self.hpad-self.window:self.hpad+self.window+1] = 0
        dynamic_map[4, self.hpad-self.window:self.hpad+self.window+1, self.hpad-self.window:self.hpad+self.window+1] = 0
        state[0] = static_map[:, xmap[1] : xmap[1] + 2*self.hpad + 1, xmap[0] : xmap[0] + 2*self.hpad + 1]
        state[0][3, self.hpad-self.window:self.hpad+self.window+1, self.hpad-self.window:self.hpad+self.window+1] = self.footprint * cos_t1
        state[0][4, self.hpad-self.window:self.hpad+self.window+1, self.hpad-self.window:self.hpad+self.window+1] = self.footprint * sin_t1
        
        return reward, stop

class CircularQueue(object):
  def __init__(self, remove_size=500, max_repetition=70):
    self.buffer_size = config.buffer_size
    self.state_buffer = np.zeros((self.buffer_size, config.compressed_state_space))
    self.rm_size = remove_size
    self.start = 0
    self.end = 0
    self.size = 0
    self.times = [[] for _ in range(len(env.Init_Data))]
    self.max_allowed = max_repetition
  
  def load_state(self):
    try:
      dbfile = open(config.path_to_save + '/buffer/dict', 'rb')     
    except:
      print('file not found!')
      return
    
    exp_state = pickle.load(dbfile)
    self.state_buffer = exp_state['state_buffer']
    self.size = exp_state['size']
    self.start = exp_state['start']
    self.end = exp_state['end']
    self.times = exp_state['times']
    dbfile.close()
  
  def save_state(self):
    exp_state = {}
    exp_state['state_buffer'] = self.state_buffer
    exp_state['size'] = self.size
    exp_state['start'] = self.start
    exp_state['end'] = self.end
    exp_state['times'] = self.times
    dbfile = open(config.path_to_save + '/buffer/dict' , 'wb')
    pickle.dump(exp_state, dbfile)
    dbfile.close()
  
  def remove(self, size):
    size = min(size, self.size)
    
    for i in range(size):
      index_1 = int(self.state_buffer[self.end][0])
      self.times[index_1].pop(0)
      self.end = (self.end + 1) % self.buffer_size
    self.size -= size
  
  def push_back(self, state):
    if self.start == self.end and self.size:
      print('queue is full! freeing %d data', self.rm_size)
      self.remove(self.rm_size)
    
    index_1 = int(state[0])
    index_list = self.times[index_1]
    
    if len(index_list) > self.max_allowed:
      idx = index_list[np.random.randint(len(index_list))]
      self.state_buffer[idx] = state
      return
    
    index_list.append(self.start)
    self.state_buffer[self.start] = state
    self.size += 1
    self.start = (self.start + 1) % self.buffer_size
  
  def get_init_state(self):
    if not self.size:
      return None
    
    idx = np.random.randint(self.end, self.end + self.size) % self.buffer_size
    return self.state_buffer[idx]

class config:
  img_size = 328
  actual_size = 82
  map_ratio = img_size / actual_size
  conv_size = [7, 14, 28, 56, 112, 224, 448]
  padding = [1, 0, 0, 0, 0, 0]
  num_action = [17, 4]
  action_range = [np.pi/4, 30.0, 3.0]
  state_space = (conv_size[0], img_size, img_size)
  compressed_state_space = 5
  min_buffer_size = 500
  p_min = -40
  save_every = 3
  buffer_size = 10000
  path_to_save = "/content/drive/My Drive/RL_PF/A2C/checkpoint"

env = Environment(328, 328, 4, 3)

env.fill_state_data('./state_dumps')

# saved_init_data = env.Init_Data
# saved_flow_data = env.Flow_Fields
# saved_road_data = env.Road_Maps

# env.Init_Data = saved_init_data
# env.Flow_Fields = saved_flow_data 
# env.Road_Maps = saved_road_data

class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.conv = nn.ModuleList()
    self.pool = nn.ModuleList()
    self.batch_norm = nn.ModuleList()
    
    for i in range(1, len(config.conv_size) - 1):
      self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=3, padding=1-config.padding[i-1]))
      self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
      self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
    
    self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=4))
    self.leakyRelu = nn.LeakyReLU(0.2)

    self.dropout_1 = nn.Dropout(0.3)
    
    self.mlp_1 = nn.Linear(config.conv_size[-1], 128)
    self.mlp_2 = nn.Linear(128, 64)
    self.mlp_3 = nn.Linear(64, 32)
    self.mlp_4 = nn.Linear(32, config.num_action[0])
    # self.mlp_5 = nn.Linear(32, 4)
    self.flatten = nn.Flatten()
    self.lsoft = nn.LogSoftmax()

  def forward(self, state):
    x = state.to(device)
    
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
    xs = self.leakyRelu(x)

    x_ret = self.mlp_4(xs)
    # y_ret = self.mlp_5(xs)

    log_p1 = torch.clamp(self.lsoft(x_ret), config.p_min)
    # log_p2 = torch.clamp(self.lsoft(y_ret), config.p_min)

    return log_p1

class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.conv = nn.ModuleList()
    self.pool = nn.ModuleList()
    self.batch_norm = nn.ModuleList()
    
    for i in range(1, len(config.conv_size) - 1):
      self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=3, padding=1-config.padding[i-1]))
      self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
      self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
    
    self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=4))
    self.leakyRelu = nn.LeakyReLU(0.2)

    self.dropout_0 = nn.Dropout(0.3)
    self.dropout_1 = nn.Dropout(0.3)
    self.mlp_1 = nn.Linear(config.conv_size[-1], 96)
    self.mlp_2 = nn.Linear(96, 32)
    self.mlp_3 = nn.Linear(32, 8)
    self.mlp_4 = nn.Linear(8, 1)
    self.flatten = nn.Flatten()
  
  def forward(self, state):
    x = state.to(device)
    
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
    x_ret = self.mlp_4(x)

    return x_ret

actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001, betas=(0.5, 0.999))
critic_optimizer = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.999))

# # uses reinforce stratergy to generate trajectory and update current policy for A2C
# def train_trajectory(env, state, actor, critic, actor_optim, critic_optim, alpha=0.2, gamma=0.99, max_steps=100):
#   actor.train()
#   critic.train()
  
#   actor_optim.zero_grad()
#   critic_optim.zero_grad()
#   advantages = []
#   logp = []
#   values = []
#   rewards = []
#   entropies = []
#   num_steps = 0
#   for i in range(max_steps):
#     mu_action, log_sigma_action = actor.forward(state[0].unsqueeze(0))
#     actions, log_prob = actor.tanh_squish(mu_action, log_sigma_action)
#     action_sample = actions.detach().squeeze(0).cpu().numpy()
#     value = critic.forward(state[0].unsqueeze(0))
#     values.append(value[0][0])

#     with torch.no_grad():
#       reward, stop = env.step(state, action_sample[0], action_sample[1], action_sample[2])
#       rewards.append(reward)
    
#     logp.append(log_prob[0])
#     num_steps += 1
    
#     if stop:
#       break
  
#   Val = torch.zeros(num_steps)
#   Val[-1] = rewards[-1] - alpha * logp[-1]
#   for i in range(num_steps-2, -1, -1):
#     Val[i] = rewards[i] - alpha * logp[i] + gamma * Val[i+1]
  
#   values = torch.stack(values, axis=0)
#   values = values.to(device)
#   Val = Val.to(device)
#   advantage = Val - values
#   d_advantage = advantage.detach() 
#   actor_loss = -(torch.stack(logp, axis=0) * d_advantage).mean()
#   critic_loss = torch.pow(advantage, 2).mean()
#   cumm_loss = actor_loss + critic_loss
#   cumm_loss.backward()
#   actor_optim.step()
#   critic_optim.step()
  
#   a_loss = actor_loss.item()
#   c_loss = critic_loss.item()

#   print('trajectory stat : actor_loss %f  critic_loss %f G_reward %f length %d' % (a_loss, c_loss, Val[0].item(), num_steps))
#   return a_loss, c_loss

def train_trajectory(env, state, actor, critic, actor_optim, critic_optim, cq, alpha=0.46, gamma=0.98, max_steps=400, min_dist_for_sample=5):
  actor.train()
  critic.train()
  
  actor_optim.zero_grad()
  critic_optim.zero_grad()
  advantages = []
  logp = []
  values = []
  rewards = []
  entropies = []
  encountered = []
  num_steps = 0
  index = state[-1]
  pos_e = state[5] / config.map_ratio

  for i in range(max_steps):
    # log_prob_curve, log_prob_length = actor.forward(state[0].unsqueeze(0))
    log_prob_curve = actor.forward(state[0].unsqueeze(0))
    value = critic.forward(state[0].unsqueeze(0))
    values.append(value[0][0])
    # log_prob = (log_prob_curve.T + log_prob_length).view(-1)
    log_prob = log_prob_curve.squeeze(0)
    absolute_prob = torch.exp(log_prob)
    entropies.append((absolute_prob * log_prob).sum())

    with torch.no_grad():
      try:
        # prob_curve = torch.exp(log_prob_curve)
        # prob_length = torch.exp(log_prob_length)
        # c1 = np.random.choice(config.num_action[0], p=prob_curve.squeeze(0).cpu().numpy())
        # c2 = np.random.choice(config.num_action[1], p=prob_length.squeeze(0).cpu().numpy())
        c1 = np.random.choice(config.num_action[0], p=absolute_prob.cpu().numpy())
      except:
        traceback.print_exception(*sys.exc_info())
        print('error nan vals!!!')
        print(log_prob_curve)
        print(entropies)
        # print(log_prob_length)
        return
      
      reward, stop = env.step(state, c1, 0)
      rewards.append(reward)
      if not stop:
        pos_s = state[4].copy()
        distance = np.linalg.norm(pos_s - pos_e)
        if distance >= min_dist_for_sample:
            encountered.append([state[2].copy(), pos_s])
    
    # logp.append(log_prob_curve[0, c1] + log_prob_length[0, c2])
    logp.append(log_prob[c1])
    num_steps += 1
    
    if stop:
      break
  
  Val = torch.zeros(num_steps)
  Val[-1] = rewards[-1]
  for i in range(num_steps-2, -1, -1):
    Val[i] = rewards[i] + gamma * Val[i+1]
  
  values = torch.stack(values, axis=0)
  values = values.to(device)
  Val = Val.to(device)
  advantage = Val - values
  d_advantage = advantage.detach() 
  actor_loss = -(torch.stack(logp, axis=0) * d_advantage).mean()
  critic_loss = torch.pow(advantage, 2).mean()
  entropies = torch.stack(entropies, axis=0)
  entropy_loss = alpha * entropies.mean()
  cumm_loss = actor_loss + critic_loss + entropy_loss
  cumm_loss.backward()
  actor_optim.step()
  critic_optim.step()

  if rewards[-1] <= 2 and len(encountered):
    random_state = encountered[np.random.randint(len(encountered))]
    cq.push_back(np.concatenate([[index], *random_state]))
  
  a_loss = actor_loss.item()
  c_loss = critic_loss.item()

  print('trajectory stat : actor_loss %f  critic_loss %f  entropy_loss %f G_reward %f length %d' % (a_loss, c_loss, entropy_loss.item(), Val[0].item(), num_steps))
  return a_loss, c_loss

def train(env, actor, critic, cq, save_prefix='', training_epoch=1, sample_per_epoch=1000):
  if use_cuda:
    actor.cuda()
    critic.cuda()
  
  for i in range(training_epoch):
    for j in range(sample_per_epoch):
      if np.random.randint(2) or cq.size < config.min_buffer_size:
        state = env.start_new_game()
      else:
        state = env.start_new_game(cq.get_init_state())
      a_loss, c_loss = train_trajectory(env, state, actor, critic, actor_optimizer, critic_optimizer, cq)
    
    print('\n completed Epoch : ', str(i), '\n.......................\n')

    if (i+1) % config.save_every == 0 and config.path_to_save != '':
      torch.save({
            'epoch': i,
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict' : critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'actor_loss': a_loss,
            'critic_loss' : c_loss
            }, config.path_to_save + '/' + 'checkpoint_' + save_prefix + str(i) + '.pth')

def load_models(checkpoint_path):
  checkpoint = torch.load(checkpoint_path, map_location=device)
  actor.load_state_dict(checkpoint['actor_state_dict'])
  critic.load_state_dict(checkpoint['critic_state_dict'])
  actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
  critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

def test_game(env, actor, cq, max_steps=1000, sample=True, print_tensor=False, use_played_sample=False):
  actor.eval()

  if use_played_sample:
    init_state = cq.get_init_state()
    print(init_state[0])
    state = env.start_new_game(init_state)
  else:
    state = env.start_new_game()
  
  trace_map = state[1][0].numpy()
  end = torch.sqrt(torch.pow(state[1][1], 2) + torch.pow(state[1][2], 2)).numpy()
  start = torch.sqrt(torch.pow(state[1][3], 2) + torch.pow(state[1][4], 2)).numpy()
  trace_map = np.stack([start, end, trace_map], axis=2)
  if len(state[-2]):
    env.color_points_in_quadrilateral(state[-2], trace_map)
  
  i = 0
  with torch.no_grad():
    stop = False
    total_reward = 0
    while i < max_steps and not stop:
      # log_p1, log_p2 = actor.forward(state[0].unsqueeze(0))
      log_p1 = actor.forward(state[0].unsqueeze(0))
      if print_tensor:
        print(log_p1)
      # print(log_p2)
      if sample:
        p1 = torch.exp(log_p1)
        # p2 = torch.exp(log_p2)
        c1 = np.random.choice(config.num_action[0], p=p1.squeeze(0).cpu().numpy())
        # c2 = np.random.choice(config.num_action[1], p=p2.squeeze(0).cpu().numpy())
      else:
        c1 = torch.argmax(log_p1.squeeze(0)).item()
        # c2 = torch.argmax(log_p2.squeeze(0)).item()
      print(c1)
      reward, stop = env.step(state, c1, 0, trace_map=trace_map, plot_trace=True)
      total_reward += reward
      i += 1
  
  print('reward_earned %f, reward_per_step %f, total_step %d, last_reward %f' % (total_reward, total_reward / i, i, reward))
  plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
  plt.imshow(trace_map)

cq = CircularQueue()

load_models(config.path_to_save + '/checkpoint_train_epochs__2.pth')

train(env, actor, critic, cq, training_epoch=30, save_prefix='train_epochs__')

cq.load_state()

cq.save_state()

cq.size

test_game(env, actor, cq, sample=False)#, use_played_sample=True)

# Testing

#  self.curve_params = [(-np.pi / 6, -10), (-np.pi / 6, 30),\
#           (-np.pi / 9, -30), (-np.pi / 9, 30),\
#           (-np.pi / 18, -30), (0, 0),\
#           (0, 30), (0, -30),\
#           (np.pi / 6, 10), (np.pi / 6, -30),\
#           (np.pi / 9, -30), (np.pi / 9, 30),\
#           (np.pi / 18, 30), (np.pi / 4, 0),\
#           (-np.pi / 4, 0), (np.pi / 36, 20), (-np.pi / 36, -20)]

state[-1]

state = env.start_new_game(game_index=1267)

state = env.start_new_game(cq.state_buffer[4071])

trace_map = state[1][0].numpy()
end = torch.sqrt(torch.pow(state[1][1], 2) + torch.pow(state[1][2], 2)).numpy()
start = torch.sqrt(torch.pow(state[1][3], 2) + torch.pow(state[1][4], 2)).numpy()
trace_map = np.stack([start, end, trace_map], axis=2)
if len(state[-2]):
  env.color_points_in_quadrilateral(state[-2], trace_map)
plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

print(env.step(state, 0, 2, trace_map=trace_map, plot_trace=True))

state[-1]

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(state[0][5].cpu().numpy(), cmap='gray')
# state[0][5:]

# state = env.start_new_game()
trace_map = state[1][0].numpy()
end = torch.sqrt(torch.pow(state[1][1], 2) + torch.pow(state[1][2], 2)).numpy()
start = torch.sqrt(torch.pow(state[1][3], 2) + torch.pow(state[1][4], 2)).numpy()
trace_map = np.stack([start, end, trace_map], axis=2)
if len(state[-2]):
  env.color_points_in_quadrilateral(state[-2], trace_map)
plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

print(env.step(state, -3, 2, trace_map=trace_map, plot_trace=True))#, trace_map=trace_map, plot_trace=True)
# trace_map = state[0][0].numpy()
# end = torch.sqrt(torch.pow(state[0][1], 2) + torch.pow(state[0][2], 2)).numpy()
# start = torch.sqrt(torch.pow(state[0][3], 2) + torch.pow(state[0][4], 2)).numpy()
# trace_map = np.stack([start, end, trace_map], axis=2)

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

# Takeaways

# exploring was suboptimal. Even with 20 dimentional action space it fails to learn simple turning maneuver
# dimentionality of state space is very big 7 * 328 * 328
# reducing dimentionality by taking a subframes of size 7 * 56 * 56. As this problem can take advantage of its local planning optimality approximates global optimality
# using ppo this time
# we use A2C with reduced params. entropy_param = 0.5 seems to work well
# experiment with state sampling : Before fuckups happens just sample any states before that and play from there. This way we can explore the map and new states better.