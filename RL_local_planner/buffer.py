import numpy as np
from Config import config
import torch
import torch.nn as nn
import pickle
from imports import device


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