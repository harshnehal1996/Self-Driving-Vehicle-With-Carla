#!/usr/bin/env python
# coding: utf-8

import carla
import sys
import os
import glob
import pygame
import queue
import pickle
import traceback
import cv2
import time
import imports
import RL_local_planner.utils.Dataset as Dataset
from copy import deepcopy
import numpy as np
from carla import VehicleLightState as vls
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as N
import torch.distributions.categorical as C
from torch.utils.tensorboard import SummaryWriter
from imports import use_cuda, device
from RL_local_planner.Config import config

# when v < 2.5 m/s and no vehicle in range
# throttle range becomes ~ [0.2 < 0.4 > 0.6]
# else
# throttle range becomes ~ [-0.5, 1]
# steer range is always [-0.85, 0.85]


# In[5]:


def test_network(actor=None, max_frames=5000, filename=''):
    setting = config.render
    config.render = True
    state = env.start_new_game(max_num_roads=5, tick_once=True)
    total_reward = 0
    long_jerk = [0]
    lat_jerk = [0]
    lo_acc = [0]
    la_acc = [0]
    vel = [0]
    rewards = [0]
    reward_jerk = [0]
    reward_long = [0]
    if filename != '':
        size = (1080, 1080)
        vid_fd = cv2.VideoWriter(filename,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 7, size)
    else:
        vid_fd = None

    def get_data(this_batch):
        x_static_t = torch.stack([this_batch[i][0]
                                 for i in range(len(this_batch))]).to(device)
        index_t = [this_batch[i][1].to(device) for i in range(len(this_batch))]
        batch_t = [this_batch[i][2].to(device) for i in range(len(this_batch))]
        input_lengths_t = np.array([len(index_t[i])
                                   for i in range(len(index_t))], dtype=np.int32)
        mask = torch.LongTensor([this_batch[i][3]
                                for i in range(len(this_batch))]).to(device)
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

                action, _ = actor.sample_from_density_fn(*actor(*get_data([[torch.from_numpy(
                    state[1]), state[3][0], state[3][1], state[7]]])), return_pi=False, deterministic=True)
                evaluation = q1(
                    *get_data([[torch.from_numpy(state[1]), state[3][0], state[3][1], state[7]]]), action)
                prob_dist = state[1][-1] * config.v_scale
                evaluation = evaluation.squeeze(0).cpu().item()

                hero = env.hero_list[state[6]]
                velocity = hero.get_velocity()
                acc = hero.get_acceleration()
                v_x, v_y = velocity.x, velocity.y
                a_x, a_y = acc.x, acc.y
                speed = np.sqrt(v_x * v_x + v_y * v_y)
                angle = state[9]

                if speed < 1e-3:
                    longitudinal_acc = a_x * angle[0] + a_y * angle[1]
                    lateral_acc = -a_x * angle[1] + a_y * angle[0]
                else:
                    longitudinal_acc = (a_x * v_x + a_y * v_y) / speed
                    lateral_acc = (-a_x * v_y + a_y * v_x) / speed

                la_acc.append(lateral_acc)
                lo_acc.append(longitudinal_acc)
                long_jerk.append(abs(lo_acc[-2] - lo_acc[-1]) * 7.5)
                lat_jerk.append(abs(la_acc[-2] - la_acc[-1]) * 7.5)
                vel.append(prob_dist)
                reward_jerk.append(
                    env.rw_jerk * ((la_acc[-2] - la_acc[-1]) * 7.5 / 90) ** 2)
                reward_long.append(env.rw_long * (longitudinal_acc ** 2))
            else:
                action = env.parse_inputs(state)
                if action is None:
                    break
                action = torch.Tensor(action)
                # action[:, 1] = np.clip(np.random.randn() * 0.3 + 0.39, 0.36, 0.42) * np.random.choice([0,1], p=[0.1, 0.9])
                evaluation = state[7]
                prob_dist = state[1][-1] * config.v_scale

            batch_action = np.zeros((1, 3), dtype=np.float32)
            index = [0]
            batch_action[:, :2][index] = action.detach().cpu().numpy()
            # batch_action[0, 1] = 0.4#np.clip(np.random.randn() * 0.3 + 0.39, 0.36, 0.42) * np.random.choice([0,1], p=[0.1, 0.9])
            reward, _, img = env.step([state], batch_action, trace=True)
            rewards.append(reward[0] - reward_jerk[-1] - reward_long[-1])
            total_reward += reward[0]
            image = env.render(state, total_reward, evaluation, prob_dist, img)

            if vid_fd is not None:
                vid_fd.write(image)

            if i % 100 == 0:
                print('completed %f percent of trajectory' %
                      round(i * 100 / max_frames, 2))

    print('total_reward....', total_reward)
    env.reset()
    config.render = setting
    if vid_fd is not None:
        vid_fd.release()

    return long_jerk, lat_jerk, lo_acc, la_acc, vel, rewards, reward_jerk, reward_long


class Experience_Buffer(object):
    def __init__(self, remove_size=25000):
        self.buffer_size = config.buffer_size
        self.state_buffer = [None for _ in range(self.buffer_size)]
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_buffer = np.zeros(
            (self.buffer_size, config.num_action), dtype=np.float32)
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
        dbfile = open(config.path_to_save + 'buffer/dict', 'wb')
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
        idx = np.random.randint(
            self.end + offset, self.end + self.size, size=batch_size) % self.buffer_size
        x_static_t = torch.stack([self.state_buffer[i][0]
                                 for i in idx]).to(device)
        index_t = [self.state_buffer[i][1].to(device) for i in idx]
        batch_t = [self.state_buffer[i][2].to(device) for i in idx]
        mask_t = torch.LongTensor([self.state_buffer[i][3]
                                  for i in idx]).to(device)
        input_lengths_t = np.array([len(index_t[i])
                                   for i in range(len(index_t))], dtype=np.int32)
        index_t = nn.utils.rnn.pad_sequence(index_t)
        batch_t = nn.utils.rnn.pad_sequence(batch_t)

        x_static_t1 = torch.stack(
            [self.next_state_buffer[i][0] for i in idx]).to(device)
        index_t1 = [self.next_state_buffer[i][1].to(device) for i in idx]
        batch_t1 = [self.next_state_buffer[i][2].to(device) for i in idx]
        mask_t1 = torch.LongTensor(
            [self.next_state_buffer[i][3] for i in idx]).to(device)
        input_lengths_t1 = np.array([len(index_t1[i])
                                    for i in range(len(index_t1))], dtype=np.int32)
        index_t1 = nn.utils.rnn.pad_sequence(index_t1)
        batch_t1 = nn.utils.rnn.pad_sequence(batch_t1)

        batch = {'state': (x_static_t, index_t, batch_t, input_lengths_t, mask_t), 'action': torch.Tensor(self.action_buffer[idx]).to(device), 'reward': torch.Tensor(
            self.reward_buffer[idx]).to(device), 'next_state': (x_static_t1, index_t1, batch_t1, input_lengths_t1, mask_t1),  'done': torch.Tensor(self.done_buffer[idx]).to(device)}

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
        x_static_t = torch.stack([this_batch[i][0]
                                 for i in range(len(this_batch))]).to(device)
        index_t = [this_batch[i][1].to(device) for i in range(len(this_batch))]
        batch_t = [this_batch[i][2].to(device) for i in range(len(this_batch))]
        input_lengths_t = np.array([len(index_t[i])
                                   for i in range(len(index_t))], dtype=np.int32)
        mask_t = torch.LongTensor([this_batch[i][3]
                                  for i in range(len(this_batch))]).to(device)
        index_t = nn.utils.rnn.pad_sequence(index_t)
        batch_t = nn.utils.rnn.pad_sequence(batch_t)

        return (x_static_t, index_t, batch_t, input_lengths_t, mask_t)

    print('games start count : ', batch_size - count)
    rewards = np.zeros(batch_size, dtype=np.float32)
    batch_action = np.zeros((batch_size, 3), dtype=np.float32)
    total_rewards = np.zeros(batch_size, dtype=np.float32)
    trajectory_length = np.zeros(batch_size, dtype=np.int32)
    policy_data = [None for _ in range(batch_size)]
    ac_loss, q1_loss, q2_loss, entropy_1, entropy_2 = 0, 0, 0, 0, 0
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
                this_batch.append(
                    [torch.from_numpy(state[i][1]), state[i][3][0], state[i][3][1], state[i][7]])
                index.append(i)

        with torch.no_grad():
            if len(index):
                if random_explore:
                    x1 = np.random.uniform(low=-actor.continious_range_1 + actor.offset_1,
                                           high=actor.continious_range_1 + actor.offset_1, size=len(index))
                    x2 = np.random.uniform(low=-actor.continious_range_2 + actor.offset_2,
                                           high=actor.continious_range_2 + actor.offset_2, size=len(index))
                    mask = np.array([this_batch[i][3]
                                    for i in range(len(this_batch))])
                    action = np.stack([np.random.uniform(low=-config.continious_range, high=config.continious_range,
                                      size=len(index)), mask * x1 + (1 - mask) * x2], axis=1).astype(np.float32)
                    batch_action[:, :-1][index] = action
                else:
                    action, _ = actor.sample_from_density_fn(
                        *actor(*get_data(this_batch)), return_pi=False)
                    batch_action[:, :-1][index] = action.detach().cpu().numpy()

                for i, idx in enumerate(index):
                    rewards[idx] = 0
                    policy_data[idx] = deepcopy(this_batch[i])

            step_rewards, start_state, _ = env.step(state, batch_action)

            for i in range(batch_size):
                rewards[i] += step_rewards[i]
                total_rewards[i] += step_rewards[i]

                if start_state[i] != 2 and state[i][10] != 1:
                    done = state[i][10] == 2
                    buffer.add_observation(deepcopy(policy_data[i]), deepcopy(batch_action[i][:-1]), rewards[i], [
                                           torch.from_numpy(state[i][1]).clone(), state[i][3][0].clone(), state[i][3][1].clone(), state[i][7]], done)

            count = sum(
                [1 if state[i][10] == 2 else 0 for i in range(batch_size)])

        if counter >= config.update_every and buffer.size > config.min_buffer_size and not random_explore:
            num_batches = max(counter // config.step_per_lr_update, 1)
            counter = 0
            ac_loss, q1_loss, q2_loss, entropy_1, entropy_2 = train_from_buffer(
                buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, num_steps=num_batches, train_steps=train_steps, neta=neta)
            train_steps += num_batches

    env.reset()

    for i in range(len(trajectory_length)):
        print('trajectory_length %d, reward_generated %.4f, last_reward %.4f, done %d' % (
            trajectory_length[i], total_rewards[i], rewards[i], state[i][10] == 2))

    ret = {'num_states': num_states, 'train_stats': [counter, train_steps, ac_loss, q1_loss, q2_loss,
                                                     entropy_1 + entropy_2], 'total_rewards': total_rewards, 'trajectory_length': trajectory_length}

    return 1, ret


def train_from_buffer(buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, loss_fn, qnormclip=1, num_steps=1, train_steps=0, neta=0.996, print_steps=False):
    q1.train()
    q2.train()
    actor.train()
    sum_ac_loss, sum_q1_loss, sum_q2_loss, sum_entropy_1, sum_entropy_2 = 0, 0, 0, 0, 0

    for i in range(num_steps):
        batch = buffer.generate_batch(
            config.batch_size, (i+1)*1.0/num_steps, w=neta)

        with torch.no_grad():
            action, logp = actor.sample_from_density_fn(
                *actor(*batch['next_state']))
            tval1 = q1_target.forward(*batch['next_state'], action)
            tval2 = q2_target.forward(*batch['next_state'], action)
            tmin_val = torch.min(tval1, tval2)
            v_t1 = tmin_val - (config.alpha * logp).sum(1)
            target = batch['reward'] + config.gamma * \
                (1 - batch['done']) * v_t1

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
        state_loss = ((config.alpha * logp).sum(1) - min_val).mean()
        actor_optimizer.zero_grad()
        state_loss.backward()
        actor_optimizer.step()

        logp = logp.detach()
        g_alpha_1 = -(logp[:, 0] + config.target_entropy_steering).mean()
        g_alpha_2 = -(logp[:, 1] + config.target_entropy_throttle).mean()
        config.alpha[0, 0] = np.clip(config.alpha[0, 0].item(
        ) - config.lr * g_alpha_1.item(), config.min_alpha[0], config.max_alpha[0])
        config.alpha[0, 1] = np.clip(config.alpha[0, 1].item(
        ) - config.lr * g_alpha_2.item(), config.min_alpha[1], config.max_alpha[1])

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
        differential_entropy_1 = -(logp[:, 0]).mean()
        differential_entropy_2 = -(logp[:, 1]).mean()

        ac_loss = state_loss.item()
        q1_loss = loss_1.item()
        q2_loss = loss_2.item()
        entropy_1 = differential_entropy_1.item()
        entropy_2 = differential_entropy_2.item()

        sum_ac_loss += ac_loss
        sum_q1_loss += q1_loss
        sum_q2_loss += q2_loss
        sum_entropy_1 += entropy_1
        sum_entropy_2 += entropy_2

        train_steps += 1

        writer.add_scalar('actor_loss', ac_loss, train_steps)
        writer.add_scalar('q1_loss', q1_loss, train_steps)
        writer.add_scalar('q2_loss', q2_loss, train_steps)
        writer.add_scalar(
            'entropy_1', differential_entropy_1.item(), train_steps)
        writer.add_scalar(
            'entropy_2', differential_entropy_2.item(), train_steps)

        if print_steps:
            print('step[%d/%d] : actor %.4f q1 %.4f q2 %.4f steer entropy %.4f throttle entropy %.4f' %
                  (i+1, num_steps, ac_loss, q1_loss, q2_loss, differential_entropy_1.item(), differential_entropy_2.item()))

    print('step %d  actor %.4f  q1 %.4f  q2 %.4f  entropy=(%.4f, %.4f)  alpha=(%.4f, %.4f)' % (num_steps, sum_ac_loss / num_steps, sum_q1_loss /
          num_steps, sum_q2_loss / num_steps, sum_entropy_1 / num_steps, sum_entropy_2 / num_steps, config.alpha[0, 0].item(), config.alpha[0, 1].item()))

    return sum_ac_loss / num_steps, sum_q1_loss / num_steps, sum_q2_loss / num_steps, sum_entropy_1 / num_steps, sum_entropy_2 / num_steps


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
            status, rets = learn(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer,
                                 q2_optimizer, loss_fn, num_policy_trajectory=10, counter=0, train_steps=0, neta=neta, random_explore=True)
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
            status, rets = learn(env, buffer, writer, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer,
                                 q2_optimizer, loss_fn, num_policy_trajectory=10, counter=counter, train_steps=train_steps, neta=neta)

            if status == 1:
                patience = min(failure_patience, patience + 0.1)
            elif status == 0:
                continue
            else:
                print('ERROR : unable to train trajectory')
                if patience <= 0:
                    raise Exception(
                        'unable to train trajectory with actor_list %d' % (len(env.hero_list)))
                patience -= 1
                continue

            counter, train_steps, ac_loss, q1_loss, q2_loss, entropy = rets['train_stats']
            rewards = rets['total_rewards']
            trajectory_lengths = rets['trajectory_length']

            for j in range(len(rewards)):
                num_trajectory += 1
                writer.add_scalar(
                    'relative_improvement', rewards[j] - config.random_reward, num_trajectory)
                writer.add_scalar('rewards', rewards[j], num_trajectory)
                writer.add_scalar('trajectory_length',
                                  trajectory_lengths[j], num_trajectory)

        neta = max(1 + (config.neta - 1) * ((i - current_epoch + 1)
                   * 1.5 / (epochs - current_epoch)), config.neta)
        print('\n..............\n epoch %d completed : total train steps = %d, neta = %.4f \n..............\n' % (
            i, train_steps, neta))

        if (i+1) % config.save_every == 0 and config.path_to_save != '':
            torch.save({
                'epoch': i,
                'actor_state_dict': actor.state_dict(),
                'q1_state_dict': q1.state_dict(),
                'q2_state_dict': q2.state_dict(),
                'q1_target_state_dict': q1_target.state_dict(),
                'q2_target_state_dict': q2_target.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'q1_optimizer_state_dict': q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': q2_optimizer.state_dict(),
                'actor_loss': ac_loss,
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'entropy': entropy,
                'num_trajectories': num_trajectory,
                'train_steps': train_steps,
                'neta': neta,
                'alpha': config.alpha,
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
        actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict'])
        q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        config.alpha = checkpoint['alpha']

        return checkpoint['epoch'], checkpoint['num_trajectories'], checkpoint['train_steps'], checkpoint['neta']

    ds = Dataset.Dataset(dumps, carla_map, 1024, 1024, config.grid_dir)
    env = Environment(10, ds, client, world)
    env.initialize()

    try:
        neta = 1
        num_trajectory = 0
        train_steps = 0
        current_epoch = 0

        if len(sys.argv) > 2:
            current_epoch, num_trajectory, train_steps, neta = load_model(
                sys.argv[2])
            print('starting from ...', current_epoch,
                  num_trajectory, train_steps, neta, config.alpha)

        exp = Experience_Buffer()
        exp.load_state()
        train(env, exp, logger, q1, q2, q1_target, q2_target, actor, actor_optimizer, q1_optimizer, q2_optimizer, random_explore=True,
              epochs=50, step_per_epoch=10, current_epoch=current_epoch, neta=neta, num_trajectory=num_trajectory, train_steps=train_steps)

    except:
        traceback.print_exception(*sys.exc_info())
    finally:
        print('cleaning')
        logger.close()
        env.exit()
        exp.save_state()


if __name__ == '__main__':
    main()
