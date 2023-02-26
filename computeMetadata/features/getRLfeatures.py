import gym
from matplotlib.pyplot import stem
import numpy as np
from multiprocessing import Pool
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import cdist
import math
from itertools import chain
import pandas as pd
import json
from tqdm import tqdm
import os
from utilities.misc import *

# do episodes with random policy, for worker in parallel process
# differs from random policy function in getDistCorrFeatures (e.g. format of output)
def randomEpisodesProcess(args):
    envName = args[0]
    stepsLB = args[1]
    seed = args[2]
    prevStateList, actionList, rewardList, newStateList, returnList, episodeStartPointList = [], [], [], [], [], []
    env = gym.make(envName)
    env.action_space.seed(seed)
    while len(rewardList) < stepsLB:
        state, _ = env.reset(seed = seed)
        seed += 1
        episodeEnded = False
        episodeStartPoint = len(rewardList)
        episodeStartPointList.append(episodeStartPoint)
        while not episodeEnded:
            action = env.action_space.sample()
            prevState = state
            state, reward, episodeEnded, _, _ = env.step(action)
            prevStateList.append(list(prevState))
            actionList.append(action)
            rewardList.append(reward)
            newStateList.append(state)
        episodeEndPoint = len(rewardList)
        returns = np.cumsum(rewardList[episodeStartPoint:episodeEndPoint][::-1])[::-1]
        returnList.extend(returns)
    stateMins = list(np.min(prevStateList, axis = 0))
    stateMaxs = list(np.max(prevStateList, axis = 0))
    episodeReturns = list(np.array(returnList)[episodeStartPointList])
    env.close()
    return [prevStateList, actionList, rewardList, newStateList,
     returnList, episodeStartPointList, stateMins, stateMaxs, episodeReturns]

def randomEpisodesParallel(envName, nJobs):
    stepsLBi = 1000
    with Pool(nJobs) as p:
        output = p.map(randomEpisodesProcess, [(envName, stepsLBi, 1000*(1+i)) for i in range(16)])
    prevStateList = [i for lst in [x[0] for x in output] for i in lst]
    actionList = [i for lst in [x[1] for x in output] for i in lst]
    rewardList = [i for lst in [x[2] for x in output] for i in lst]
    newStateList = [i for lst in [x[3] for x in output] for i in lst]
    returnList = [i for lst in [x[4] for x in output] for i in lst]
    episodeStartPointList = [i for lst in [x[5] for x in output] for i in lst]
    stateMins = np.min([x[6] for x in output], axis = 0)
    stateMaxs = np.max([x[7] for x in output], axis = 0)
    episodeReturnsList = [i for lst in [x[8] for x in output] for i in lst]
    env = gym.make(envName)
    env.action_space.seed(0)
    A = env.action_space.n
    avRandomScore = np.mean(episodeReturnsList)
    env.close()
    return prevStateList, actionList, rewardList, newStateList,\
         returnList, episodeStartPointList, stateMins,\
             stateMaxs, A, avRandomScore

def generatePrototypes(stateMins, stateMaxs, numPrototypes = 100):
    space = Space(list(zip(stateMins, stateMaxs)))
    lhs = Lhs(lhs_type = "classic", criterion = None)
    prototypes = np.array(lhs.generate(space.dimensions, numPrototypes))
    return prototypes

def q(protos, action, theta):
    return np.sum(theta[protos, action])

def runEpisodeQ(args):
    theta = args[0]
    envName = args[1]
    prototypes = args[2]
    knearest = args[3]
    alpha = args[4]
    gamma = args[5]
    explorationMode = args[6]
    epsilon = args[7]
    learn = args[8]
    seed = args[9]
    np.random.seed(seed)
    data = dict.fromkeys(['state', 'action', 'protos'])
    env = gym.make(envName)
    env.action_space.seed(seed)
    data['state'], _ = env.reset(seed = seed)
    D = len(data['state'])
    distances = cdist(prototypes, np.array(data['state']).reshape(1, D))
    data['protos'] = np.argpartition(distances.flatten(), knearest)[:knearest]
    Q = np.sum(theta[data['protos'], :], axis = 0)
    if explorationMode == "epsilonGreedy" and np.random.rand() < epsilon:
        data['action'] = env.action_space.sample()
    else:
        data['action'] = np.argmax(Q)
    episodeReward = 0
    episodeEnded = False
    while not episodeEnded:
        prevData = {'prevState': data['state'], 'prevAction': data['action'], 'prevProtos': data['protos']}
        data['state'], reward, episodeEnded, _, _ = env.step(prevData['prevAction'])
        distances = cdist(prototypes, np.array(data['state']).reshape(1, D))
        data['protos'] = np.argpartition(distances.flatten(), knearest)[:knearest]
        Q = np.sum(theta[data['protos'], :], axis = 0)
        if explorationMode == "epsilonGreedy" and np.random.rand() < epsilon:
            data['action'] = env.action_space.sample()
        else:
            data['action'] = np.argmax(Q)
        if learn:
            if not episodeEnded:
                update = alpha * (reward + gamma * q(data['protos'], data['action'], theta) - q(prevData['prevProtos'], prevData['prevAction'], theta))
            else:
                update = alpha * (reward - q(prevData['prevProtos'], prevData['prevAction'], theta))
            theta[prevData['prevProtos'], prevData['prevAction']] += update # NOTE: assume featureVector is binary 1 or 0, so multiplication is implicit
        episodeReward += reward
    env.close()
    del env
    return episodeReward

def trainAgent(args):
    envName = args[0]
    prototypes = args[1]
    knearest = args[2]
    lowEstimate = args[3]
    highEstimate = args[4]
    A = args[5]
    alpha = args[6]
    gamma = args[7]
    M = args[8]
    explorationMode = args[9]
    epsilon = args[10]
    if explorationMode == "optimistic":
        theta = np.zeros((prototypes.shape[0], A)) + highEstimate
    elif explorationMode == "epsilonGreedy":
        theta = np.zeros((prototypes.shape[0], A)) + lowEstimate
    else:
        assert(False)
    Rtestmax = -np.Inf
    for i in range(M):
        seed = 10*(i+1)
        R = runEpisodeQ((theta, envName, prototypes, knearest, alpha, gamma, explorationMode, epsilon, True, seed))
        alpha = alpha * 0.99
        if (i + 1) % math.ceil(M/4) == 0:
            testReturns = []
            for j in range(10):
                seed = 1000*(i+1)*(j+1)
                testReturns.append(runEpisodeQ((theta, envName, prototypes, knearest, alpha, gamma, explorationMode, epsilon, False, seed)))
            Rtest = sum(testReturns)/len(testReturns)
            if Rtest > Rtestmax:
                Rtestmax = Rtest
    return dict({"Rtestmax" : Rtestmax, "gamma" : gamma, "numPrototypes" : prototypes.shape[0], "explorationMode" : explorationMode})

def computeRLtable(envName, updateAttrs, nJobs, knearest = 5):
    with open("envAttributes.json") as attrDataJSON:
        attrData = (json.load(attrDataJSON))[envName]
    for attkey in updateAttrs.keys():
        attrData[attkey] = {'val' : updateAttrs[attkey]}
    envParamsString = "#" + "#".join([str(val['val']) for val in list(attrData.values())])
    setAttributes(updateAttrs)
    prevStateList, actionList, rewardList, newStateList, returnList, episodeStartPointList, stateMins, stateMaxs, A, avRandomScore = randomEpisodesParallel(envName, nJobs)
    prototypes10 = generatePrototypes(stateMins, stateMaxs, 10)
    prototypes50 =  generatePrototypes(stateMins, stateMaxs, 50)
    prototypes100 =  generatePrototypes(stateMins, stateMaxs, 100)
    prototypesList = [prototypes10, prototypes50, prototypes100]
    lowEstimate = min(returnList)/knearest
    highEstimate = np.abs((50 + max(returnList))/knearest)
    nRepeats = 1

    with Pool(nJobs) as p:
        results = p.map(trainAgent, [(envName, prototypes, knearest, lowEstimate, highEstimate, A, 0.1, gamma, 100, explorationMode, 0.1) for gamma in [0.9, 0.95, 1] for prototypes in prototypesList for explorationMode in ["optimistic", "epsilonGreedy"] for k in range(nRepeats)])
    df = pd.DataFrame(results)

    scores = df['Rtestmax']
    if np.max(scores) - avRandomScore <= 0:
        scores = 0*(scores - avRandomScore)
    else:
        scores = (scores - avRandomScore)/(np.max(scores) - avRandomScore) 
    scores = list((scores >= 0) * scores)
    df['normalisedScore'] = scores
    df['instances'] = [envName + envParamsString for i in range(df.shape[0])]
    df['avRandomScore'] = [avRandomScore for i in range(df.shape[0])]
    df = df[['instances', 'Rtestmax', 'avRandomScore', 'normalisedScore', 'gamma', 'numPrototypes', 'explorationMode']]
    resetAttributes(updateAttrs)
    writeIfExists("RLfeatures.csv", df)
    return df

def convertRLtableToDictionary(rltable):
    paramNames = ["gamma", "numPrototypes", "explorationMode"]
    featureNameTemplates = ['feature_GX', 'feature_PX', 'feature_EX']
    nvals = [3, 3, 2]
    RLfeatures = dict({})
    for i in range(len(paramNames)):
        values = list(rltable.groupby([paramNames[i]]).median()['normalisedScore'])
        maxVal = max(values)
        if maxVal != 0:
            values = list(map(lambda x: x/maxVal, values))
        featureNames = [featureNameTemplates[i] + str(j) for j in range(1, len(values)+1)]
        for j in range(len(featureNames)):
            RLfeatures[featureNames[j]] = values[j]
    return RLfeatures