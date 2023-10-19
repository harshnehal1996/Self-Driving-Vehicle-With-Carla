#!/usr/bin/env python
# coding: utf-8
from paths import ProjectPaths
sys.path.append(glob.glob(ProjectPaths.carla_pylibs)
                [0])  # import carla python library

from train import train
from env import Environment
from buffer import Experience_Buffer
from model import Critic, Actor
from config import config
from imports import use_cuda, device
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import carla
from copy import deepcopy
import utils.Dataset as Dataset
import traceback
import pickle
import sys
import os
import glob


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
