import gym

### so that gym.envs has the below attributes
gym.make("CartPole-v1")
gym.make("MountainCar-v0")
gym.make("LunarLander-v2")
###

# CartPole
global __init__old_CP
global __init__new_CP
global step_old_CP
global step_new_CP
# MountainCar
global __init__old_MC
global __init__new_MC
global step_old_MC
global step_new_MC
# LunarLander
global __init__old_LL
global __init__new_LL
global step_old_LL
global step_new_LL

# modify CartPole to fix an issue with episode termination requiring thousands of steps
## patch __init method
__init__old_CP = gym.envs.classic_control.CartPoleEnv.__init__
def __init__new_CP(self, render_mode = None):
    __init__old_CP(self, render_mode)
    self.step_number = 0
    self._max_episode_steps = 500
gym.envs.classic_control.CartPoleEnv.__init__ = __init__new_CP
## patch step method
step_old_CP = gym.envs.classic_control.CartPoleEnv.step
def step_new_CP(self, action : int):
    state, reward, terminated, truncated, info = step_old_CP(self, action)
    self.step_number += 1
    if self.step_number >= 500:
        terminated = True
    if terminated:
        self.step_number = 0
    return state, reward, terminated, truncated, info
gym.envs.classic_control.CartPoleEnv.step = step_new_CP

# modify MountainCarEnv to fix an issue with episode termination requiring thousands of steps
## patch __init method
__init__old_MC = gym.envs.classic_control.MountainCarEnv.__init__
def __init__new_MC(self, render_mode = None, goal_velocity = 0):
    __init__old_MC(self, render_mode)
    self.step_number = 0
    self._max_episode_steps = 500
gym.envs.classic_control.MountainCarEnv.__init__ = __init__new_MC
## patch step method
step_old_MC = gym.envs.classic_control.MountainCarEnv.step
def step_new_MC(self, action : int):
    state, reward, terminated, truncated, info = step_old_MC(self, action)
    self.step_number += 1
    if self.step_number >= 500:
        terminated = True
    if terminated:
        self.step_number = 0
    return state, reward, terminated, truncated, info
gym.envs.classic_control.MountainCarEnv.step = step_new_MC

# modify LunarLander to cap episode duration, since it is possible for the lander to fly for excessive time
## patch __init method
__init__old_LL = gym.envs.box2d.lunar_lander.LunarLander.__init__
def __init__new_LL(self, render_mode = None):
    __init__old_LL(self, render_mode)
    self.step_number = 0
    self.wind_power = 0.0
    self.turbulence_power = 0.0
    self.scale = 999 # arbitrary since setAttributes in utils.py is used
    gym.envs.box2d.lunar_lander.SCALE = self.scale
    self.initial_random = 998 # arbitrary since setAttributes in utils.py is used
    gym.envs.box2d.lunar_lander.INITIAL_RANDOM = self.initial_random
    self.engine_power_mult = 997 # arbitrary since setAttributes in utils.py is used
    gym.envs.box2d.lunar_lander.MAIN_ENGINE_POWER = 13.0 * self.engine_power_mult
    gym.envs.box2d.lunar_lander.SIDE_ENGINE_POWER = 0.6 * self.engine_power_mult
gym.envs.box2d.lunar_lander.LunarLander.__init__ = __init__new_LL
## patch step method
step_old_LL = gym.envs.box2d.lunar_lander.LunarLander.step
def step_new_LL(self, action : int):
    state, reward, terminated, truncated, info = step_old_LL(self, action)
    self.step_number += 1
    if not terminated and self.step_number >= 600:
        terminated = True
        reward = -100
    if terminated:
        self.step_number = 0
    ###
    # print(f"SCALE = {gym.envs.box2d.lunar_lander.SCALE}")
    # print(f"INITIAL_RANDOM = {gym.envs.box2d.lunar_lander.INITIAL_RANDOM}")
    # print(f"engine_power_mult = {self.engine_power_mult}")
    # print(f"MAIN_ENGINE_POWER = {gym.envs.box2d.lunar_lander.MAIN_ENGINE_POWER}")
    # print(f"SIDE_ENGINE_POWER = {gym.envs.box2d.lunar_lander.SIDE_ENGINE_POWER}")

    return state, reward, terminated, truncated, info
gym.envs.box2d.lunar_lander.LunarLander.step = step_new_LL