import numpy as np
import pandas as pd
import dcor
import gym
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import cdist

# random episodes for distance correlation computations
# no parallelisation here to make distance correlation
# computations simpler; does not take much time anyway
def randomEpisodes(envName, maxSteps):
    env = gym.make(envName)
    env.action_space.seed(0)
    prevStateList = []
    actionList = []
    rewardList = []
    newStateList = []
    returnList = []
    episodeStartPointList = []
    seed = 0
    while len(rewardList) < maxSteps:
        state, _ = env.reset(seed = seed)
        seed += 1
        episodeEnded = False
        episodeStartPoint = len(rewardList)
        episodeStartPointList.append(episodeStartPoint)
        while not episodeEnded:
            action = env.action_space.sample()
            prevState = state
            state, reward, episodeEnded, _, _ = env.step(action)
            # env.render()
            prevStateList.append(list(prevState))
            actionList.append(action)
            rewardList.append(reward)
            newStateList.append(state)
            # env.render()
        episodeEndPoint = len(rewardList)
        returns = np.cumsum(rewardList[episodeStartPoint:episodeEndPoint][::-1])[::-1]
        returnList.extend(returns)
    return np.array(prevStateList), np.array(actionList), np.array(rewardList), np.array(newStateList), np.array(returnList), episodeStartPointList

def computeDCORfeatures(envName, numSteps, numPrototypes):
    prevStateArray, actionArray, rewardArray, newStateArray, returnArray, episodeStartPointList = randomEpisodes(envName, numSteps)
    prevStateArray = prevStateArray.astype('float')
    actionArray = actionArray.astype('float')
    rewardArray = rewardArray.astype('float')
    newStateArray = newStateArray.astype('float')
    returnArray = returnArray.astype('float')
    env = gym.make(envName)
    env.action_space.seed(0)
    D = env.observation_space.shape[0] # dimensionality of observation space
    A = env.action_space.n # dimensionality of action space
    env.close()
    sqrDcorSASp = dcor.u_distance_correlation_sqr(np.concatenate((prevStateArray, np.vstack(actionArray)), axis = 1), newStateArray)
    sqrDcorSRtn = dcor.u_distance_correlation_sqr(prevStateArray, returnArray)
    sqrDcorSSp = dcor.u_distance_correlation_sqr(prevStateArray, newStateArray)
    sqrDcorSARtn = dcor.u_distance_correlation_sqr(np.concatenate((prevStateArray, np.vstack(actionArray)), axis = 1), returnArray)
    sqrDcorSASpRtn = dcor.u_distance_correlation_sqr(np.concatenate((prevStateArray, np.vstack(actionArray)), axis = 1),
     np.concatenate((newStateArray, np.vstack(returnArray)), axis = 1))
    sqrDcorASp = dcor.u_distance_correlation_sqr(actionArray, newStateArray)
    sqrDcorARtn = dcor.u_distance_correlation_sqr(actionArray, returnArray)
    sqrDcorSpRtn = dcor.u_distance_correlation_sqr(newStateArray, returnArray)
    sqrDcorASpRtn = dcor.u_distance_correlation_sqr(actionArray, np.concatenate((newStateArray, np.vstack(returnArray)), axis = 1))
    sqrDcorSSpRtn = dcor.u_distance_correlation_sqr(prevStateArray, np.concatenate((newStateArray, np.vstack(returnArray)), axis = 1))
    avRandomScore = np.mean(returnArray[episodeStartPointList])
    features = {"feature_sqrDcorSASp" : sqrDcorSASp, "feature_sqrDcorSRtn" : sqrDcorSRtn,
     "feature_sqrDcorSSp" : sqrDcorSSp, "feature_sqrDcorSARtn" : sqrDcorSARtn,
     "feature_sqrDcorSASpRtn" : sqrDcorSASpRtn, "feature_sqrDcorASp" : sqrDcorASp,
     "feature_sqrDcorARtn" : sqrDcorARtn, "feature_sqrDcorSpRtn" : sqrDcorSpRtn,
     "feature_sqrDcorASpRtn" : sqrDcorASpRtn, "feature_sqrDcorSSpRtn" : sqrDcorSSpRtn}
    stateMins = np.min(prevStateArray, axis = 0)
    stateMaxs = np.max(prevStateArray, axis = 0)
    stateMaxs[stateMins == stateMaxs] += 1
    space = Space(list(zip(stateMins, stateMaxs)))
    lhs = Lhs(lhs_type = "classic", criterion = None)
    prototypes = np.array(lhs.generate(space.dimensions, numPrototypes))
    return features, avRandomScore, prototypes