from logger import logging_setup, display_config
import logging

import argparse
from argparse import Namespace
import os

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from rl.networks.envs import make_vec_envs
from rl.evaluation import evaluate
from rl.networks.model import Policy

from crowd_sim import *

from config_reader import read_config
from pathlib import Path

def main():
	"""
	The main function for testing a trained model
	"""

	# the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the model directory that we are testing
	parser.add_argument('--model_dir', type=str, default='trained_models/env_experiment')
	# render the environment or not
	parser.add_argument('--visualize', default=True, action='store_true')
	# if -1, it will run 500 different cases; if >=0, it will run the specified test case repeatedly
	parser.add_argument('--test_case', type=int, default=-1)
	# model weight file you want to test
	parser.add_argument('--test_model', type=str, default='41665.pt')
	# whether to save trajectories of episodes
	parser.add_argument('--render_traj', default=False, action='store_true')
	# whether to save slide show of episodes
	parser.add_argument('--save_slides', default=False, action='store_true')
	# verbose mode
	parser.add_argument('--verbose', default=True, action='store_true')
	test_args = parser.parse_args()
	if test_args.save_slides:
		test_args.visualize = True

	# configure logging and device
	# print test result in log file
	log_file = os.path.join(test_args.model_dir,'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)
	if test_args.visualize:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
	else:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_' + test_args.test_model + '.log')

	logging_setup(log_file)

	from importlib import import_module
	model_dir_temp = test_args.model_dir
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
		logging.error('Failed to get get_args function from ', test_args.model_dir, '/arguments.py\
				\nImporting arguments from default directory.')
		from arguments import get_args

	algo_args = get_args()

	# import config class from saved directory
	# if not found, import from the default directory

	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')
		logging.info(f'Importing Config from {model_dir_string}.py')

	except:
		logging.error('Failed to get Config function from ', test_args.model_dir, '/configs/config.py')
		from crowd_nav.configs.config import Config
	env_config = config = Config()

	if test_args.verbose:
		display_config(test_args,'Test Arguments')
		display_config(algo_args,'Arguments [bold]from argument.py[/]')

		# TODO: make use of the crowdNav.config file for this part
		
		# conf = read_config(Path(test_args.model_dir) / 'crowdNav.config')
		# display_config(Namespace(**conf.algorithm_config),'Environment Config')
		# algo_args = Namespace(**conf.algorithm_config)
		# algo_args.cuda = not algo_args.no_cuda and torch.cuda.is_available()

		assert algo_args.algo in ['a2c', 'ppo', 'acktr']
		if algo_args.recurrent_policy:
			assert algo_args.algo in ['a2c', 'ppo'], \
				'Recurrent policy is not implemented for ACKTR'

	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False


	torch.set_num_threads(1)
	device = torch.device("cuda" if algo_args.cuda else "cpu")

	logging.info('Create other envs with new settings')

	# set up visualization
	if test_args.visualize:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-6.5, 6.5) # 6
		ax.set_ylim(-6.5, 6.5)
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		# ax.set_xlabel('x(m)', fontsize=16)
		# ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
	logging.info('Load model from: ' + load_path)


	# create an environment
	env_name = algo_args.env_name

	eval_dir = os.path.join(test_args.model_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	env_config.render_traj = test_args.render_traj
	env_config.save_slides = test_args.save_slides
	env_config.save_path = os.path.join(test_args.model_dir, 'social_eval', test_args.test_model[:-3])
	envs = make_vec_envs(env_name, algo_args.seed, 1,
						 algo_args.gamma, eval_dir, device, allow_early_resets=True,
						 config=env_config, ax=ax, test_case=test_args.test_case, pretext_wrapper=config.env.use_wrapper)

	if config.robot.policy not in ['orca', 'social_force']:
		# load the policy weights
		actor_critic = Policy(
			envs.observation_space.spaces,
			envs.action_space,
			base_kwargs=algo_args,
			base=config.robot.policy)
		actor_critic.load_state_dict(torch.load(load_path, map_location=device))
		actor_critic.base.nenv = 1

		# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
		nn.DataParallel(actor_critic).to(device)
	else:
		actor_critic = None

	test_size = config.env.test_size

	exit(0)
	# call the evaluation function
	evaluate(actor_critic, envs, 1, device, test_size, logging, config, algo_args, test_args.visualize)


if __name__ == '__main__':
	main()