import numpy as np
import gym
import pandas as pd
from computeMetadata.performance.algorithms.call_algorithms import *
import time

# learners = {"PPO" : PPO, "SACdiscrete" : SACdiscrete, "IQN" : IQN, "Rainbow" : Rainbow}
def computePerformanceData(envName, learners, numLearnerSamples, step_per_epoch, epoch, device):
    learnerPerformances = dict()
    allEpochData = dict()
    timeData = dict()
    for l in list(learners.keys()):
        allEpochData[l] = [[] for _ in range(numLearnerSamples)]
        # timeData[l] = dict({"Wall Time" : [], "CPU Time" : []})
        timeData[l] = [[], []]  # index 0 is wall time, 1 is cpu time
        scores = []
        seed = 0
        for i in range(numLearnerSamples):
            open('temp.txt', 'w').close()
            open('temp_best_reward.txt', 'w').close()
            start_wall_time = time.time()
            start_cpu_time = time.process_time()
            learners[l](envName, step_per_epoch, epoch, seed, device)
            end_wall_time = time.time()
            end_cpu_time = time.process_time()
            seed += 1
            timeData[l][0].append(end_wall_time - start_wall_time)
            timeData[l][1].append(end_cpu_time - start_cpu_time)
            with open('temp.txt', 'r') as file:
                epochsData = file.read()
            epochsData = epochsData[:-2] if epochsData[-2:] == ", " else epochsData
            epochsData = list(map(float, epochsData.split(", ")))
            allEpochData[l][i] = epochsData
            with open('temp_best_reward.txt', 'r') as file:
                best_reward = file.read()
                best_reward = best_reward[:-2] if best_reward[-2:] == ", " else best_reward
                best_reward = list(map(float, best_reward.split(", ")))[-1]
                scores.append(best_reward)
        averageScore = np.mean(scores)
        learnerPerformances[l] = averageScore
        allEpochData[l] = str(allEpochData[l])
        timeData[l] = str(timeData[l])
    return learnerPerformances, allEpochData, timeData

def normalisePerformanceData(envName, learnerNames, learnerPerformances, avRandomScore):
    scores = np.array(list(learnerPerformances.values()))
    maxScore = np.max(scores)
    if maxScore - avRandomScore <= 0:
        scores = 0*(scores - avRandomScore)
    else:
        scores = (scores - avRandomScore)/(np.max(scores) - avRandomScore)
    # Truncate scores from below by 0. Normalised score is then
    #  a relative measure of how much learning has been achieved over random policy.  
    return dict(zip(learnerPerformances.keys(), list((scores >= 0) * scores)))