from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from imports import device
from Config import config


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
