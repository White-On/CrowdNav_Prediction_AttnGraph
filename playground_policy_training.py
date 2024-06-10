from rl.networks.envs import make_env
# from rl.networks.envs import make_vec_envs
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from rl.networks.storage import RolloutStorage
from crowd_sim.envs import *
from rl.networks.model import Policy
from rl.networks.CRYOSHELL import CryoShell
from torch.distributions.normal import Normal


from rich import print
import torch
from rl import ppo
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from logger import logging_setup
import pandas as pd
import numpy as np
import gym
import torch.optim as optim
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.shape_input = np.array(envs.envs[0].observation_space.spaces['graph_features'].shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.shape_input, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = CryoShell(
            vehicle_node_size=envs.envs[0].observation_space.spaces['robot_node'].shape[0],
            n_embd=envs.envs[0].observation_space.spaces['graph_features'].shape[0],
            n_feature_graph=envs.envs[0].spatial_edge_dim,
            n_head=4,
            n_layer=2,
            n_action=2
            )

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, graph_feature):
        return self.critic(graph_feature.view(-1, self.shape_input))

    def get_action_and_value(self, graph_feature, robot_node, visibility_mask, action=None):
        action_mean = self.actor(graph_feature, visibility_mask, robot_node)
        # logging.info(f"{action_mean = }")
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(graph_feature.view(-1, self.shape_input))

def main():
    from importlib import import_module
    model_dir = 'trained_models/env_experiment'

    log_path = Path(model_dir) / 'playground_env.log'
    logging_setup(log_path)


    model_dir_temp = model_dir
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]
    # import arguments.py from saved directory
    # if not found, import from the default directory
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.arguments'
        logging.info(f'Importing arguments from {model_dir_string}.py')
        model_arguments = import_module(model_dir_string)
        get_args = getattr(model_arguments, 'get_args')
    except:
        logging.error('Failed to get get_args function from ', model_dir, '/arguments.py\
                \nImporting arguments from default directory.')
        from arguments import get_args

    algo_args = get_args()

    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
        logging.info(f'Importing Config from {model_dir_string}.py')

    except:
        logging.error('Failed to get Config function from ', model_dir, '/configs/config.py')
        from crowd_nav.configs.config import Config
    env_config = config = Config()
    
    arena_size = env_config.sim.arena_size
    arena_viz_factor = 2

    # set up visualization
    # fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-arena_size*arena_viz_factor, arena_size*arena_viz_factor)
    ax.set_ylim(-arena_size*arena_viz_factor, arena_size*arena_viz_factor)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.ion()
    plt.show()

    seed = np.random.randint(0, 1000)
    nb_enviroments = 1

    # env = make_env("CrowdSimCar-v0", seed, 1, "_", True,config=env_config, ax=ax)
    env = DummyVecEnv(
        [make_env("CrowdSimCar-v0", seed, i, "_", True,config=env_config, ax=ax, envNum=nb_enviroments) for i in range(nb_enviroments)]
    )

    # logging.info(f"{env.envs[0].observation_space = }")
    env.reset()

    num_steps = 200
    num_updates = 1
    log_file = 'env_experiment.log'
    save = False
    learning_rate = 1e-4
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    batch_size = 32
    nb_update_epochs = 4
    clip_coef = 0.2
    minibatch_size = 16
    clip_vloss = True
    norm_adv = True
    vf_coef = 0.5
    ent_coef = 0.01
    target_kl = 0.01
    max_grad_norm = 0.5

    log_results_episodes = {'episode':[], 'status':[], 'reward':[], 'steps':[]}

    speed_action_space = [env.envs[0].action_space.low[0], env.envs[0].action_space.high[0]]
    delta_action_space = [env.envs[0].action_space.low[1], env.envs[0].action_space.high[1]]

    agent = Agent(env)
    # logging.info(f"{agent.actor = }")
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    observations_graph_features = torch.zeros((num_steps, nb_enviroments, env.envs[0].observation_space.spaces['graph_features'].shape[0], env.envs[0].observation_space.spaces['graph_features'].shape[1]))
    observations_node_vehicle = torch.zeros((num_steps, nb_enviroments, env.envs[0].observation_space.spaces['robot_node'].shape[0]))
    observations_visible_masks = torch.zeros((num_steps, nb_enviroments, env.envs[0].observation_space.spaces['visible_masks'].shape[0], env.envs[0].observation_space.spaces['visible_masks'].shape[0]))
    actions = torch.zeros((num_steps, nb_enviroments, env.envs[0].action_space.shape[0]))
    logprobs = torch.zeros((num_steps, nb_enviroments))
    rewards = torch.zeros((num_steps, nb_enviroments))
    dones = torch.zeros((num_steps, nb_enviroments))
    values = torch.zeros((num_steps, nb_enviroments))

    next_obs = env.reset()
    next_obs_graph_features = next_obs['graph_features']
    next_obs_node_vehicle = next_obs['robot_node']
    next_obs_visible_masks = next_obs['visible_masks']
    next_done = torch.zeros(nb_enviroments)

    # print(f'{next_obs_graph_features.shape = }, {next_obs_node_vehicle.shape = }, {next_obs_visible_masks.shape =}')

    for update in range(num_updates):
        # we reset the environment at the beginning of each update
        env.reset()
        # we step though the environment for num_steps steps
        # if the environment is done, we reset it and keep on exploring
        for step in range(num_steps):
            # we only render the environment for the first environment
            env.envs[0].render()

            # smooth brain policy

            # angle_from_goal = [env.envs[i].robot.get_angle_from_goal() for i in range(nb_enviroments)]
            # angle_to_take = [np.clip(angle, delta_action_space[0], delta_action_space[1]) for angle in angle_from_goal]

            # distance_from_humans = [env.envs[i].compute_distance_from_human() for i in range(nb_enviroments)]
            # closest_human_distance = np.min(distance_from_humans, axis=1)

            # speed = np.clip(closest_human_distance*0.2, 0, 1)
            # action = np.vstack([speed, angle_to_take]).T
            # logging.info(f"{action = }")


            observations_graph_features[step] = torch.tensor(next_obs_graph_features.tolist())
            observations_node_vehicle[step] = torch.tensor(next_obs_node_vehicle.tolist())
            observations_visible_masks[step] = torch.tensor(next_obs_visible_masks.tolist())

            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    observations_graph_features[step], 
                    observations_node_vehicle[step], 
                    observations_visible_masks[step]
                    )
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            obs, reward, done, info = env.step(action)

            next_obs_graph_features = torch.Tensor(obs['graph_features'])
            next_obs_node_vehicle = torch.Tensor(obs['robot_node'])
            next_obs_visible_masks = torch.Tensor(obs['visible_masks'])

            rewards[step] = torch.tensor(reward)

            idx_done = np.where(done)[0]
            for idx in idx_done:
                env.envs[idx].reset()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_graph_features).reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # logging.info(f"{advantages.shape = }")
        # logging.info(f"{returns.shape = }")
        
        out_pred = obs['graph_features'][0, :, 2:]
        ack = env.envs[0].talk2Env(out_pred)

        # flatten the batch
        b_obs_graph_features = observations_graph_features.reshape((-1,) + env.envs[0].observation_space.spaces['graph_features'].shape)
        b_obs_node_vehicle = observations_node_vehicle.reshape((-1,) + env.envs[0].observation_space.spaces['robot_node'].shape)
        b_obs_visible_masks = observations_visible_masks.reshape((-1,) + env.envs[0].observation_space.spaces['visible_masks'].shape)

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,env.envs[0].action_space.shape[0]))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        
    
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(nb_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_graph_features[mb_inds], b_obs_node_vehicle[mb_inds], b_obs_visible_masks[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break
            

    plt.show()
    if save:
        pd.DataFrame(log_results_episodes).to_csv(log_file, index=False)
    
    env.close()

if __name__ == '__main__':
    main()