# -*- coding: utf-8 -*-
"""SAC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EHpgtz7mjYu3GhPBUDP2HusGKtLLmMOo
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
from copy import deepcopy
import cv2
import os
import pickle

!mkdir Data

!unzip "/content/drive/My Drive/RL_PF/state_dumps.zip" -d "/content/Data"

cd /content/Data/

# Commented out IPython magic to ensure Python compatibility.
# %ls



# outputs distribution parameter mean and sigma given states
# tanh(mean + sigma * N)
# action : [angle, y_coeff, length]
# -1 <= angle < 1  -> [-45*, 45*]
# -1  <= a <= 1    -> [-30, 30]
# length -> [0, 6]
# sizes = [328, 164, 82, 41, 20, 10, 5, 1]

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

class config:
  img_size = 328
  actual_size = 82
  map_ratio = img_size / actual_size
  conv_size = [7, 14, 25, 40, 80, 160, 320, 640]
  padding = [0, 0, 0, 1, 0, 0]
  action_space = 3
  action_range = [np.pi/4, 30.0, 3.0]
  state_space = (conv_size[0], img_size, img_size)
  compressed_state_space = 5
  buffer_size = int(1e6)
  temperature = 0.2
  batch_size = 16
  random_explore = 4000
  polyak = 0.995
  gamma = 0.99
  min_buffer_size = 1200
  update_every = 128
  save_every = 2
  path_to_save = "/content/drive/My Drive/RL_PF/"

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
    
    self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=5))
    self.leakyRelu = nn.LeakyReLU(0.2)

    self.dropout_0 = nn.Dropout(0.4)
    self.dropout_1 = nn.Dropout(0.3)
    self.dropout_2 = nn.Dropout(0.2)
    self.dropout_3 = nn.Dropout(0.2)
    
    self.mlp_1 = nn.Linear(config.conv_size[-1], 144)
    self.mlp_2 = nn.Linear(144, 64)
    self.mlp_3 = nn.Linear(64, 16)
    self.mlp_4 = nn.Linear(16, 3)
    self.mlp_5 = nn.Linear(144, 64)
    self.mlp_6 = nn.Linear(64, 16)
    self.mlp_7 = nn.Linear(16, 3)
    self.flatten = nn.Flatten()
    self.action_range = torch.Tensor(config.action_range).to(device).unsqueeze(0)
    self.sigma_min = -40
    self.sigma_max = 3

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
    # x = self.dropout_0(x)
    
    x = self.mlp_1(x)
    x = self.leakyRelu(x)
    xs = self.dropout_1(x)
    
    x = self.mlp_2(xs)
    x = self.leakyRelu(x)
    x = self.dropout_2(x)

    x = self.mlp_3(x)
    x = self.leakyRelu(x)
    x_ret = self.mlp_4(x)

    x = self.mlp_5(xs)
    x = self.leakyRelu(x)
    x = self.dropout_3(x)

    x = self.mlp_6(x)
    x = self.leakyRelu(x)
    y_ret = self.mlp_7(x)

    return x_ret, y_ret
  
  # log(tanh'(x)) = log(1 - tanh(x)^2) = 2 * log(2 / (exp(x) + exp(-x)) = 2 * (log2 - x - softplus(-2x))
  def tanh_squish(self, mean, log_sigma, train=True):
    if train:
      log_std = torch.clamp(log_sigma, self.sigma_min, self.sigma_max)
      std = torch.exp(log_std)
      ndist = N.Normal(mean, std)
      a_inv = ndist.rsample()
      logp = ndist.log_prob(a_inv).sum(axis=1) - (2 * (np.log(2) - a_inv - F.softplus(-2 * a_inv))).sum(axis=1)
    else:
      a_inv = mean
      logp = 0
    
    norm_action = torch.tanh(a_inv)
    return self.action_range * norm_action, logp

class QCritic(nn.Module):
  def __init__(self):
    super(QCritic, self).__init__()
    self.conv = nn.ModuleList()
    self.pool = nn.ModuleList()
    self.batch_norm = nn.ModuleList()
    
    for i in range(1, len(config.conv_size) - 1):
      self.conv.append(nn.Conv2d(config.conv_size[i-1], config.conv_size[i], kernel_size=3, padding=1-config.padding[i-1]))
      self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=config.padding[i-1]))
      self.batch_norm.append(nn.BatchNorm2d(config.conv_size[i]))
    
    self.conv.append(nn.Conv2d(config.conv_size[-2], config.conv_size[-1], kernel_size=5))
    self.leakyRelu = nn.LeakyReLU(0.2)

    self.dropout_0 = nn.Dropout(0.4)
    self.dropout_1 = nn.Dropout(0.3)
    self.dropout_2 = nn.Dropout(0.3)
    self.mlp_1 = nn.Linear(config.conv_size[-1], 139)
    self.mlp_2 = nn.Linear(154, 64)
    self.mlp_3 = nn.Linear(64, 32)
    self.mlp_4 = nn.Linear(32, 8)
    self.mlp_5 = nn.Linear(8, 1)
    self.emb_1 = nn.Linear(1, 6, bias=False)
    self.emb_2 = nn.Linear(1, 6, bias=False)
    self.emb_3 = nn.Linear(1, 3, bias=False)
    self.flatten = nn.Flatten()
  
  def forward(self, state, action):
    x = state.to(device)
    
    for i in range(len(self.conv) - 1):
      x = self.conv[i](x)
      x = self.leakyRelu(x)
      x = self.pool[i](x)
      x = self.batch_norm[i](x)
    
    x = self.conv[-1](x)
    x = self.leakyRelu(x)
    x = self.flatten(x)
    # x = self.dropout_0(x)

    x = self.mlp_1(x)
    x = self.leakyRelu(x)
    x = self.dropout_1(x)

    e1 = self.emb_1(action[:, 0].unsqueeze(-1))
    e2 = self.emb_2(action[:, 1].unsqueeze(-1))
    e3 = self.emb_3(action[:, 2].unsqueeze(-1))
    x = torch.cat([x, e1, e2, e3], axis=1)

    x = self.mlp_2(x)
    x = self.leakyRelu(x) 
    x = self.dropout_2(x)
    
    x = self.mlp_3(x)
    x = self.leakyRelu(x)
    x = self.mlp_4(x)
    x = self.leakyRelu(x)

    x_ret = self.mlp_5(x)
    
    return x_ret

# Circular queue based buffer
class Experience_Replay(object):
  def __init__(self, remove_size=25000):
    self.buffer_size = config.buffer_size
    self.state_buffer = np.zeros((self.buffer_size, config.compressed_state_space))
    self.reward_buffer = np.zeros(self.buffer_size)
    self.action_buffer = np.zeros((self.buffer_size, config.action_space))
    self.next_state_buffer = np.zeros((self.buffer_size, config.compressed_state_space))
    self.done_buffer = np.zeros(self.buffer_size)
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
    self.size += 1
    self.start = (self.start + 1) % self.buffer_size
  
  def generate_batch(self, env, batch_size):
    batch_size = min(batch_size, self.size)
    if not batch_size:
      return None
    
    idx = np.random.randint(self.end, self.end + self.size, size=batch_size) % self.buffer_size
    done = self.done_buffer[idx]

    batch = { 'state' : env.expand_to_state(self.state_buffer[idx], []),\
              'action' : torch.Tensor(self.action_buffer[idx]).to(device),\
              'reward' : torch.Tensor(self.reward_buffer[idx]).to(device),\
              'next_state' : env.expand_to_state(self.next_state_buffer[idx], done),\
              'done' : torch.Tensor(done).to(device)}
    
    return batch

def generate_new_episodes(env, buffer, actor, max_steps=100, game_index=-1, random_explore=False):
  state = env.start_new_game(game_index)  
  i = 0
  with torch.no_grad():
    stop = False
    while i < max_steps and not stop:
      prev_state_param = np.concatenate([[state[-1]], state[1], np.around(state[3] * config.map_ratio)])
      if random_explore:
        action_sample = [np.random.uniform(low=-config.action_range[0], high=config.action_range[0]),\
                         np.random.uniform(low=-config.action_range[1], high=config.action_range[1]),\
                         np.random.uniform(low=-config.action_range[2], high=config.action_range[2])]
      else:
        mu_action, log_sigma_action = actor.forward(state[0].unsqueeze(0))
        actions, _ = actor.tanh_squish(mu_action, log_sigma_action)
        action_sample = actions.squeeze(0).cpu().numpy()
      reward, stop = env.step(state, action_sample[0], action_sample[1], action_sample[2])
      new_state_param = np.concatenate([[state[-1]], state[1], np.around(state[3] * config.map_ratio)])
      buffer.add_observation(prev_state_param, action_sample, reward, new_state_param, stop)
      i += 1
  
  return i

def train_from_buffer(env, buffer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss, num_steps=1):
  q1.train()
  q2.train()
  actor.train()
  ac_loss, q1_loss, q2_loss = 0, 0, 0

  for i in range(num_steps):
    batch = buffer.generate_batch(env, config.batch_size)
    actor_optimizer.zero_grad()
    q1_optimizer.zero_grad()
    q2_optimizer.zero_grad()

    with torch.no_grad():
      mu_act, log_sigma_act = actor.forward(batch['next_state'])
      actions, logp = actor.tanh_squish(mu_act, log_sigma_act)
      tval1 = q1_target.forward(batch['next_state'], actions)
      tval2 = q2_target.forward(batch['next_state'], actions)
      tmin_val, _ = torch.min(torch.hstack([tval1, tval2]), axis=1)
      target = batch['reward'] + config.gamma * (1 - batch['done']) * (tmin_val - config.temperature * logp)
    
    val1 = q1.forward(batch['state'], batch['action']).squeeze(-1)
    val2 = q2.forward(batch['state'], batch['action']).squeeze(-1)
    loss_1 = loss(val1, target)
    loss_2 = loss(val2, target)
    q_losses = loss_1 + loss_2
    q_losses.backward()
    q1_optimizer.step()
    q2_optimizer.step()

    q1.requires_grad = False
    q2.requires_grad = False
    mu_act, log_sigma_act = actor.forward(batch['state'])
    actions, logp = actor.tanh_squish(mu_act, log_sigma_act)
    val1_ = q1.forward(batch['state'], actions)
    val2_ = q2.forward(batch['state'], actions)
    min_val, _ = torch.min(torch.hstack([val1_, val2_]), axis=1)
    state_loss = (-min_val + config.temperature * logp).mean()
    state_loss.backward()
    actor_optimizer.step()

    for param, param_t in zip(q1.parameters(), q1_target.parameters()):
      param_t.data = config.polyak * param_t.data + (1 - config.polyak) * param.data
    
    for param, param_t in zip(q2.parameters(), q2_target.parameters()):
      param_t.data = config.polyak * param_t.data + (1 - config.polyak) * param.data
    
    q1.requires_grad = True
    q2.requires_grad = True

    ac_loss += state_loss.item()
    q1_loss += loss_1.item()
    q2_loss += loss_2.item()
  
  print('loss : actor %f q1 %f q2 %f' % (ac_loss / num_steps, q1_loss / num_steps, q2_loss / num_steps))  
  
  return ac_loss / num_steps, q1_loss / num_steps, q2_loss / num_steps

def load_models(checkpoint_path):
  checkpoint = torch.load(checkpoint_path, map_location=device)
  actor.load_state_dict(checkpoint['actor_state_dict'])
  q1.load_state_dict(checkpoint['q1_state_dict'])
  q2.load_state_dict(checkpoint['q2_state_dict'])
  q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
  q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
  actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
  q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
  q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])

def trainAI(env, buffer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, epochs=100, save_prefix=''):
  if use_cuda:
    q1.cuda()
    q2.cuda()
    q1_target.cuda()
    q2_target.cuda()
    actor.cuda()

  q1_target.requires_grad = False
  q2_target.requires_grad = False
  loss_fn = nn.MSELoss()
  num_batches = max(round(config.update_every / config.batch_size), 1)
  num_batches = 1

  for i in range(config.random_explore):
    generate_new_episodes(env, buffer, actor, random_explore=True)
  
  counter = 0
  data_size = len(env.Init_Data)
  current_epoch = 0
  a_loss, q1_loss, q2_loss = np.inf, np.inf, np.inf
  train_steps = 0

  for i in range(epochs):
    for step in range(data_size):
      counter += generate_new_episodes(env, buffer, actor)
      if counter >= config.update_every and buffer.size > config.min_buffer_size:
        counter = 0
        train_steps += num_batches
        a_loss, q1_loss, q2_loss = train_from_buffer(env, buffer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, num_steps=num_batches)
    
    print('\n..............\n epoch %d completed : total train steps = %d \n..............\n' % (i, train_steps))

    if (i+1) % config.save_every == 0 and config.path_to_save != '':
      torch.save({
            'epoch': i,
            'actor_state_dict': actor.state_dict(),
            'q1_state_dict' : q1.state_dict(),
            'q2_state_dict' : q2.state_dict(),
            'q1_target_state_dict' : q1_target.state_dict(),
            'q2_target_state_dict' : q2_target.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'q1_optimizer_state_dict': q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': q2_optimizer.state_dict(),
            'actor_loss': a_loss,
            'q1_loss' : q1_loss,
            'q2_loss' : q2_loss,
            }, config.path_to_save + 'checkpoints/' + 'checkpoint_' + save_prefix + str(i) + '.pth')
      buffer.save_state()

def testAI(env, actor, game_index, max_steps=1000, sampled=True):
  q1.eval()
  q2.eval()
  actor.eval()

  state = env.start_new_game(game_index)
  trace_map = state[0][0].numpy()
  end = torch.sqrt(torch.pow(state[0][1], 2) + torch.pow(state[0][2], 2)).numpy()
  start = torch.sqrt(torch.pow(state[0][3], 2) + torch.pow(state[0][4], 2)).numpy()
  trace_map = np.stack([start, end, trace_map], axis=2)
  if len(state[-2]):
    env.color_points_in_quadrilateral(state[-2], trace_map)
  
  i = 0
  with torch.no_grad():
    stop = False
    total_reward = 0
    while i < max_steps and not stop:
      mu_action, log_sigma_action = actor.forward(state[0].unsqueeze(0))
      actions, _ = actor.tanh_squish(mu_action, log_sigma_action, train=sampled)
      print(actions)
      print(_)
      action_sample = actions.squeeze(0).cpu().numpy()
      print(action_sample)
      reward, stop = env.step(state, action_sample[0], action_sample[1], action_sample[2], trace_map=trace_map, plot_trace=True)
      total_reward += reward
      i += 1
  
  print('reward_earned %f, reward_per_step %f, total_step %d, last_reward %f' % (total_reward, total_reward / i, i, reward))
  plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
  plt.imshow(trace_map)

exp = Experience_Replay()
exp.load_state()

q1 = QCritic()
q2 = QCritic()

actor = Actor()

q1_target = deepcopy(q1)
q2_target = deepcopy(q2)

actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001, betas=(0.5, 0.999))
q1_optimizer = optim.Adam(q1.parameters(), lr=0.0001, betas=(0.5, 0.999))
q2_optimizer = optim.Adam(q2.parameters(), lr=0.0001, betas=(0.5, 0.999))

trainAI(env, exp, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, epochs=5, save_prefix='p_')

cd "/content/drive/My Drive/RL_PF/"

# Commented out IPython magic to ensure Python compatibility.
# %cd checkpoints/

# Commented out IPython magic to ensure Python compatibility.
# %ls

exp.size

np.exp(-1.0199)

exp.size

testAI(env, actor, -1, sampled=False)

# load_models(config.path_to_save + 'checkpoints/checkpoint_0.pth')

# train_from_buffer(env, exp, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, nn.MSELoss(), num_steps=1)

# trace_map = state[0][0].numpy()
# end = torch.sqrt(torch.pow(state[0][1], 2) + torch.pow(state[0][2], 2)).numpy()
# start = torch.sqrt(torch.pow(state[0][3], 2) + torch.pow(state[0][4], 2)).numpy()
# trace_map = np.stack([start, end, trace_map], axis=2)
# # if len(state[-2]):
#     # env.color_points_in_quadrilateral(state[-2], trace_map)
# plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
# plt.imshow(trace_map)

"""

reward_downlane = -10
reward_uplane = 1
reward_mid = 0.3
time_penalty = -0.5
wall_hit = -10
goal_reached = 10

"""

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
        self.max_field_reward = 1
        self.min_field_reward = 0.3
        self.swath_resolution = self.window / self.map_ratio
        self.out_threshold = out_threshold
        self.expected = torch.sum(self.footprint).item()
        self.steps = 0.01
        self.s = np.arange(0, 1 + self.steps / 2, self.steps)
        self.num_points = int(1 / self.steps) + 1
        self.time_penalty = 0.5
        self.HIT_WALL_REWARD = -5
        self.WRONG_DIR_REWARD = -10
        self.MAX_REWARD = 10
        self.sample_per_new_games = sample_per_new_games
        self.Init_Data = None
        self.Road_Maps = None
        self.Flow_Fields = None
    
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
            self.Road_Maps[idx] = image[:, :, 0].astype(np.float32) / 255
        
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
    
    def start_new_game(self, index_1=-1):
        if index_1 == -1:
            data_size = len(self.Init_Data)
            index_1 = np.random.randint(data_size)
        
        index_2 = index_1 // self.sample_per_new_games
        features = torch.zeros(7, self.size_y, self.size_x)
        s_pt = self.Init_Data[index_1][2] * self.map_ratio
        e_pt = self.Init_Data[index_1][3]
        cos_et, sin_et = self.Init_Data[index_1][1]
        cos_rt, sin_rt = self.Init_Data[index_1][0]
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

        states = [features, np.array([cos_rt, sin_rt]), np.array([cos_et, sin_et]),\
                 np.array([s_pt[0] / self.map_ratio, s_pt[1] / self.map_ratio]),\
                 np.array([e_pt[0], e_pt[1]]), quad, points, index_2]
        
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
    
    def color_points_in_quadrilateral(self, P, img):
        A1 = np.linalg.inv(np.vstack((P[1] - P[0], P[2] - P[0])).T)
        A2 = np.linalg.inv(np.vstack((P[3] - P[0], P[2] - P[0])).T)
        pnts = np.vstack(P).T
        
        x_max = round(min(np.max(pnts[0]) + 1, self.size_x))
        y_max = round(min(np.max(pnts[1]) + 1, self.size_y))
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
            if norm < 1e-5 or out_of_bound or img[y][x] == 1 or field[y][x].dot(F) <= 0:
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
            if norm < 1e-5 or out_of_bound or img[y][x] == 1 or field[y][x].dot(F) <= 0:
                break
            F = field[y][x]
        
        F = field[start[1], start[0]]
        rpos = rpos + (F * offset) / d
        r1pos = rpos + (F * size) / d
        A1 = np.linalg.inv(np.vstack((r1pos - l1pos, rpos - l1pos)).T)
        A2 = np.linalg.inv(np.vstack((lpos - l1pos, rpos - l1pos)).T)

        return [l1pos, A1, A2], [l1pos, r1pos, rpos, lpos]

    def step(self, state, angle, f_y, length, trace_map=[], plot_trace=False, min_length=0.25):        
        Map = state[0]
        start_angle = state[1]
        end_angle = state[2]
        start_pt = state[3]
        end_pt = state[4]
        Qpoint = state[5]

        length += min_length + config.action_range[2]
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
        else:
            step_size = int(max((self.num_points - 1) * self.swath_resolution / length, 1))
            for i in range(self.num_points - 1, step_size - 1, -step_size):
                mean = torch.sum(Map[0, starts[1][i] : ends[1][i], starts[0][i] : ends[0][i]] * self.footprint) / self.expected
                if mean >= self.out_threshold:
                    reward += self.HIT_WALL_REWARD
                    stop = True
                    break
                if not is_inside and len(Qpoint) != 0:
                    is_inside = self.is_inside_quadrilateral(Qpoint, curve_point[:, i])

        if not stop:
            gradients = torch.Tensor(R.dot(gradients).T.reshape(-1))
            potentials = Map[5:].permute(1, 2, 0)[curve_point[1], curve_point[0]].reshape(-1)
            
            if plot_trace:
                trace_map[curve_point[1], curve_point[0]] = 1.0
            
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
            elif is_inside:
                reward += self.HIT_WALL_REWARD
                stop = True
        
        return reward, stop

env = Environment(328, 328, 4, 3)

# env.fill_state_data('./state_dumps')

env.Init_Data = saved_init_data
env.Flow_Fields = saved_flow_data 
env.Road_Maps = saved_road_data

# saved_init_data = env.Init_Data
# saved_flow_data = env.Flow_Fields
# saved_road_data = env.Road_Maps

state = env.start_new_game()
trace_map = state[0][0].numpy()
end = torch.sqrt(torch.pow(state[0][1], 2) + torch.pow(state[0][2], 2)).numpy()
start = torch.sqrt(torch.pow(state[0][3], 2) + torch.pow(state[0][4], 2)).numpy()
trace_map = np.stack([start, end, trace_map], axis=2)
if len(state[-2]):
    env.color_points_in_quadrilateral(state[-2], trace_map)
plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(trace_map)

