import gym
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import average
import torch
from itertools import chain
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import dcor
from computeMetadata.performance.algorithms.call_algorithms import *
import gc
import json
import time
from computeMetadata.features.getPolicyNetworkData import *
from computeMetadata.features.policyNetworkFeatures import *
from computeMetadata.features.getDistCorrFeatures import *
from computeMetadata.performance.getPerformanceData import *
from computeMetadata.features.getRLfeatures import *
import itertools
import sys
import os
from utilities.misc import *
import signal
import time

class instancer:
    def __init__(self, fParams, pParams, doRLfeatures = True):
        self.fParams = fParams  # of type 'featureParameters'
        self.pParams = pParams  # of type 'algoParameters'
        self.doRLfeatures = doRLfeatures
    def readInstanceName(self, instanceName : str) -> tuple[str, dict]:
        specification = instanceName.split("#")
        envName = specification[0]
        envParamVector = list(map(float, specification[1:]))
        updateAttrs = dict({})
        with open("envAttributes.json") as attrDataJSON:
            attrData = (json.load(attrDataJSON))
        attrNames = list(attrData[envName].keys())
        for i in range(len(attrNames)):
            updateAttrs[attrNames[i]] = envParamVector[i]
        return envName, updateAttrs
    def writeInstanceName(self, envName : str, envParamVector : list[float]) -> str:
        return envName + "#" + "#".join([str(val) for val in envParamVector])
    def computeFeatures(self, instanceName : str, nJobs : int) -> pd.DataFrame:
        envName, updateAttrs = self.readInstanceName(instanceName)
        maxStepsDict = dict({"CartPole-v1" : 500, "MountainCar-v0" : 500, "LunarLander-v2" : 1000})
        setattr(gym.Env, '_max_episode_steps', property(lambda self,\
             x = maxStepsDict[instanceName.split("#")[0]]: x, lambda self, v: None))
        setAttributes(updateAttrs)
        feature_computation_times = {"DCOR" : [], "PolicyNetwork" : [], "RLlandmark" : []}
        start_time = time.time()
        start_cpu_time = time.process_time()
        dcorFeatures, avRandomScore, prototypes = computeDCORfeatures(envName, self.fParams.numRandomSteps, self.fParams.numPrototypes)
        end_time = time.time()
        end_cpu_time = time.process_time()
        feature_computation_times["DCOR"].append(end_time - start_time)
        feature_computation_times["DCOR"].append(end_cpu_time - start_cpu_time)
        feature_computation_times["DCOR"] = str(feature_computation_times["DCOR"])
        start_time = time.time()
        start_cpu_time = time.process_time()
        policyFeatures = computePolicyNetworkFeatures(envName, self.fParams.N, self.fParams.M, prototypes, self.fParams.knearest, instanceName, nJobs)
        end_time = time.time()
        end_cpu_time = time.process_time()
        feature_computation_times["PolicyNetwork"].append(end_time - start_time)
        feature_computation_times["PolicyNetwork"].append(end_cpu_time - start_cpu_time)
        feature_computation_times["PolicyNetwork"] = str(feature_computation_times["PolicyNetwork"])
        start_time = time.time()
        start_cpu_time = time.process_time()
        if self.doRLfeatures:
            rltable = computeRLtable(envName, updateAttrs, nJobs, self.fParams.knearest)
            RLfeatures = convertRLtableToDictionary(rltable)
            allFeatures = {**policyFeatures, **dcorFeatures, **RLfeatures}
        else:
            allFeatures = {**policyFeatures, **dcorFeatures}
        end_time = time.time()
        end_cpu_time = time.process_time()
        feature_computation_times["RLlandmark"].append(end_time - start_time)
        feature_computation_times["RLlandmark"].append(end_cpu_time - start_cpu_time)
        feature_computation_times["RLlandmark"] = str(feature_computation_times["RLlandmark"])
        resetAttributes(updateAttrs)
        setattr(gym.Env, '_max_episode_steps', None)
        timeData = pd.DataFrame(feature_computation_times, index = [0])
        timeData.insert(0, 'instances', instanceName)
        writeIfExists("timeDataFeatures.csv", timeData)
        allFeatures = pd.DataFrame(allFeatures, index=[0])
        allFeatures.insert(0, 'instances', instanceName)
        nameOfFile = "newMetadata.csv"
        do_not_interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if os.path.isfile(nameOfFile):
            allFeatures.to_csv(nameOfFile, mode = "a", header = False, index = False)
        else:
            allFeatures.to_csv(nameOfFile, index = False)
        signal.signal(signal.SIGINT, do_not_interrupt)
        return allFeatures
    def computePerformanceMetrics(self, instanceName : str, nJobs : int) -> None:
        envName, updateAttrs = self.readInstanceName(instanceName)
        maxStepsDict = dict({"CartPole-v1" : 500, "MountainCar-v0" : 500, "LunarLander-v2" : 1000})
        setattr(gym.Env, '_max_episode_steps', property(lambda self,\
             x = maxStepsDict[instanceName.split("#")[0]]: x, lambda self, v: None))
        setAttributes(updateAttrs)
        _, _, _, _, _, _, _, _, _, avRandomScore = randomEpisodesParallel(envName, nJobs)
        learnerPerformances, allEpochData, timeData = computePerformanceData(envName, self.pParams.learners,
         self.pParams.numLearnerSamples, self.pParams.step_per_epoch, self.pParams.epoch, self.pParams.device)
        normalisedPerformances = normalisePerformanceData(envName, list(self.pParams.learners.keys()),
         learnerPerformances, avRandomScore)
        resetAttributes(updateAttrs)
        do_not_interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
        setattr(gym.Env, '_max_episode_steps', None)
        normPerfDF = pd.DataFrame(normalisedPerformances, index=[0])
        normPerfDF.insert(0, 'instances', instanceName)
        rawPerfDF = pd.DataFrame(learnerPerformances, index=[0])
        rawPerfDF.insert(0, 'instances', instanceName)
        rawPerfDF.insert(1, 'randomScore', avRandomScore)
        allEpochData = pd.DataFrame(allEpochData, index = [0])
        allEpochData.insert(0, 'instances', instanceName)
        allEpochData.insert(1, 'randomScore', avRandomScore)
        timeData = pd.DataFrame(timeData, index = [0])
        timeData.insert(0, 'instances', instanceName)
        writeIfExists("newNormPerfDF.csv", normPerfDF)
        writeIfExists("newRawPerfDF.csv", rawPerfDF)
        writeIfExists("allEpochData.csv", allEpochData)
        writeIfExists("timeDataPerf.csv", timeData)
        signal.signal(signal.SIGINT, do_not_interrupt)