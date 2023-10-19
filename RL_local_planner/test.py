
import cv2
import numpy as np
import torch
import torch.nn as nn
from imports import device
from config import config


def test_network(env, q1, actor=None, max_frames=5000, filename=''):
    # first argument is the environment
    # second argument is the critic network
    # third argument is the actor network
    # fourth argument is the maximum number of frames to run
    # fifth argument is the filename to save the video

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
