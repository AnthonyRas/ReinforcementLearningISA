from algosRL.test.discrete import test_ppo, test_sac, test_iqn, test_rainbow
import numpy as np
from tianshou.env import SubprocVectorEnv

global PPO, SACdiscrete, IQN, Rainbow

# SubprocVectorEnv is faster for testing runs after epochs
for module in [test_ppo, test_sac, test_iqn, test_rainbow]:
    module.DummyVectorEnv = SubprocVectorEnv

# set arguments for learners
def set_args(test_module, task, step_per_epoch, epoch, seed, device):
    args = test_module.get_args()
    args.task, args.step_per_epoch, args.epoch, args.seed, args.device = \
        task, step_per_epoch, epoch, seed, device
    args.reward_threshold = np.Inf
    args.training_num = 10
    args.test_num = 20
    args.step_per_collect = 10
    args.gamma = 0.99
    return args

# call PPO algorithm
def PPO(task, step_per_epoch, epoch, seed, device):
    args = set_args(test_ppo, task, step_per_epoch, epoch, seed, device)
    args.step_per_collect = 2000
    try:
        test_ppo.test_ppo(args)
    except AssertionError:
        pass

def SACdiscrete(task, step_per_epoch, epoch, seed, device):
    args = set_args(test_sac, task, step_per_epoch, epoch, seed, device)
    try:
        test_sac.test_discrete_sac(args)
    except AssertionError:
        pass

def IQN(task, step_per_epoch, epoch, seed, device):
    args = set_args(test_iqn, task, step_per_epoch, epoch, seed, device)
    try:
        test_iqn.test_iqn(args)
    except AssertionError:
        pass

def Rainbow(task, step_per_epoch, epoch, seed, device):
    args = set_args(test_rainbow, task, step_per_epoch, epoch, seed, device)
    args.prioritized_replay = True  # part of most Rainbow implementations
    try:
        test_rainbow.test_rainbow(args)
    except AssertionError:
        pass