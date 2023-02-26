from numpy.core.fromnumeric import shape
from numpy.lib.function_base import average
import torch
from itertools import chain
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import dcor
import random
import gym
from utilities.misc import *
from computeMetadata.performance.algorithms.call_algorithms import *
import gc
import json
import time
from computeMetadata.features.getPolicyNetworkData import *
from computeMetadata.features.policyNetworkFeatures import *
from computeMetadata.features.getDistCorrFeatures import *
from computeMetadata.performance.getPerformanceData import *
from computeMetadata.features.getRLfeatures import *
from computeMetadata.instancers import *
import itertools
import sys
import os
import json
import ast
from scipy.optimize import minimize_scalar
from instanceGeneration.generatorInitial import *
import signal
import warnings
warnings.filterwarnings("ignore")
from utilities import patchGym
import tianshou
# monkey patch tianshou to give epoch test results when running test_ppo, etc.
oldNext = tianshou.trainer.base.BaseTrainer.__next__
def patchedNext(self):
    results = oldNext(self)
    with open("temp.txt", 'a') as file:
        file.write(str(results[1]['test_reward']) + ', ')
    with open("temp_best_reward.txt", 'a') as file:
        file.write(str(results[1]['best_reward']) + ', ')
    return results
tianshou.trainer.base.BaseTrainer.__next__ = patchedNext


# disable prints from tianshou to reduce clutter
setattr(tianshou.trainer.OnpolicyTrainer, "verbose", property(lambda self,\
             x = False: x, lambda self, v: None))
setattr(tianshou.trainer.OnpolicyTrainer, "show_progress", property(lambda self,\
             x = False: x, lambda self, v: None))
setattr(tianshou.trainer.OffpolicyTrainer, "verbose", property(lambda self,\
             x = False: x, lambda self, v: None))
setattr(tianshou.trainer.OffpolicyTrainer, "show_progress", property(lambda self,\
             x = False: x, lambda self, v: None))

# details the parameters selected for feature computations
class featureParameters:
    def __init__(self, numRandomSteps = 3000, numPrototypes = 100, N = 100, M = 20, knearest = 5):
        self.numRandomSteps = numRandomSteps
        self.numPrototypes = numPrototypes
        self.N = N
        self.M = M
        self.knearest = knearest

# details the parameters selected for algorithm runs
class algoParameters:
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu',
     learners = {"algo_PPO" : PPO, "algo_SACdiscrete" : SACdiscrete,
      "algo_IQN" : IQN, "algo_Rainbow" : Rainbow}, 
      numLearnerSamples = 3, step_per_epoch = 8000, epoch = 4):
        self.learners = learners
        self.numLearnerSamples = numLearnerSamples
        self.step_per_epoch = step_per_epoch
        self.epoch = epoch
        self.device = device  # device to run RL algs on, 'cuda' or 'cpu'

# the instancer object will be doing almost everything
instancerObj = instancer(featureParameters(), algoParameters())

# generate a 'grid' of instances, varying environment parameters
Instances = generateInitialInstances(instancerObj,
 envNames = ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"], ks = [4, 8, 8])


# get instances for which features have been computed so far
if os.path.isfile("instances_features_completed.txt"):
    with open("instances_features_completed.txt", "r") as file:
        text_string = file.read()
    instances_features_completed = ast.literal_eval(text_string)
else:
    instances_features_completed = []
# get instances for which performance metrics have been computed so far
if os.path.isfile("instances_performance_completed.txt"):
    with open("instances_performance_completed.txt", "r") as file:
        text_string = file.read()
    instances_performance_completed = ast.literal_eval(text_string)
else:
    instances_performance_completed = []

# shuffle instances to make it less likely that a failure point is later on rather than early
random.seed(0)
random.shuffle(Instances)
# for each instance in the list 'Instances', compute features then performance metrics
for i in tqdm(range(len(Instances))):
    instanceName = Instances[i]
    # compute features for instance
    if instanceName not in instances_features_completed:
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        features = instancerObj.computeFeatures(instanceName, nJobs = 16)
        instances_features_completed.append(instanceName)
        with open("instances_features_completed.txt", "w") as file:
            file.write(str(instances_features_completed))
    if instanceName not in instances_performance_completed:
        # compute performance metrics for instance
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        performances = instancerObj.computePerformanceMetrics(instanceName, nJobs = 16)
        instances_performance_completed.append(instanceName)
        with open("instances_performance_completed.txt", "w") as file:
            file.write(str(instances_performance_completed))