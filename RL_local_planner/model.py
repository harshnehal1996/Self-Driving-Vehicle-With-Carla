import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as N
import torch.distributions.categorical as C
from config import config


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dynamic_encoder = nn.GRU(
            config.embedding_size + 16, 128, bidirectional=True)
        self.positional_embeddings = nn.Embedding(
            config.num_rays + 3, config.embedding_size, padding_idx=0)
        self.mode_embedding = nn.Embedding(2, 8)
        self.project = nn.Linear(128, 76)

        self.mlp_1 = nn.Linear(config.num_rays + 16, 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(140, 96)
        self.mlp_4 = nn.Linear(3, 6)
        self.mlp_5 = nn.Linear(110, 64)
        self.mlp_6 = nn.Linear(64, 32)
        self.mlp_7 = nn.Linear(32, 1)

    def forward(self, x_static, index_seq, sequence, input_lengths, mask, action):
        batch_size = len(input_lengths)
        positional = self.positional_embeddings(index_seq)
        dynamic_feature = torch.cat([sequence, positional], axis=2)
        packed = nn.utils.rnn.pack_padded_sequence(
            dynamic_feature, input_lengths, batch_first=False, enforce_sorted=False)
        _, hidden = self.dynamic_encoder(packed)
        x_dynamic = torch.mean(hidden, axis=0)

        clipped_action = torch.stack(
            [torch.clip(action[:, 1], min=0), -torch.clip(action[:, 1], max=0)], axis=-1)
        x = self.mlp_1(x_static)
        x = self.leakyRelu(x)
        x = self.mlp_2(x)
        x = self.leakyRelu(x)
        x = torch.cat([x, self.leakyRelu(self.project(x_dynamic))], axis=1)
        x = self.mlp_3(x)
        x = self.leakyRelu(x)

        y = self.mlp_4(torch.cat([action[:, :1], clipped_action], axis=1))
        z = self.mode_embedding(mask)

        x = torch.cat([x, y, z], axis=1)
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
        self.dynamic_encoder = nn.GRU(
            config.embedding_size + 16, 128, bidirectional=True)
        self.positional_embeddings = nn.Embedding(
            config.num_rays + 3, config.embedding_size, padding_idx=0)
        self.mode_embedding = nn.Embedding(2, 8)
        self.project = nn.Linear(128, 76)
        self.mlp_1 = nn.Linear(config.num_rays + 16, 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(140, 120)
        self.mlp_4 = nn.Linear(120, 64)
        self.mlp_5 = nn.Linear(64, 2)
        self.mlp_6 = nn.Linear(128, 64)
        self.mlp_7 = nn.Linear(64, 2)
        self.continious_range_1 = 0.9
        self.continious_range_2 = 0.2
        self.offset_1 = 0.1
        self.offset_2 = 0.4

    def forward(self, x_static, index_seq, sequence, input_lengths, mask):
        batch_size = len(input_lengths)
        positional = self.positional_embeddings(index_seq)
        dynamic_feature = torch.cat([sequence, positional], axis=2)
        packed = nn.utils.rnn.pack_padded_sequence(
            dynamic_feature, input_lengths, batch_first=False, enforce_sorted=False)
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

        y = torch.cat([x_mid, self.mode_embedding(mask)], axis=1)
        y = self.mlp_6(y)
        y = self.leakyRelu(y)
        y_ret = self.mlp_7(y)

        mu = torch.cat([x_ret[:, :1], y_ret[:, :1]], axis=1)
        log_sigma = torch.cat([x_ret[:, 1:], y_ret[:, 1:]], axis=1)
        log_sigma = torch.clamp(log_sigma, config.sigma_min, config.sigma_max)

        return mu, log_sigma, mask

    def sample_from_density_fn(self, mu, log_sigma, mask, return_pi=True, deterministic=False):
        if deterministic:
            pi = None
            sample = mu
        else:
            std = torch.exp(log_sigma)
            ndist = N.Normal(mu, std)
            sample = ndist.rsample()
            if return_pi:
                pi = ndist.log_prob(
                    sample) - (2 * (np.log(2) - sample - F.softplus(-2 * sample)))
            else:
                pi = None

        action = torch.tanh(sample)
        throttle = action[:, 1]
        throttle = mask * (throttle * self.continious_range_1 + self.offset_1) + \
            (1 - mask) * (throttle * self.continious_range_2 + self.offset_2)
        action = torch.stack(
            [action[:, 0] * config.continious_range, throttle], axis=1)

        return action, pi
