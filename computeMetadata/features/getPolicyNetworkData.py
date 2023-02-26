import gym
import numpy as np
from multiprocessing import Pool
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import cdist

def getPi(theta, state, prototypes, knearest):
    D = len(state)
    distances = cdist(prototypes, np.array(state).reshape(1, D))
    activePrototypes = np.argpartition(distances.flatten(), knearest)[:knearest]
    eta = np.sum(theta[activePrototypes, :], axis = 0)
    pi = np.exp(eta - max(eta))/sum(np.exp(eta - max(eta)))
    return pi

def runEpisode(args):
    seed = args[4]
    np.random.seed(seed)
    theta = args[0]
    envName = args[1]
    prototypes = args[2]
    knearest = args[3]
    env = gym.make(envName)
    env.action_space.seed(seed)
    state, _ = env.reset(seed = seed)
    episodeReward = 0
    episodeEnded = False
    while not episodeEnded:
        pi = getPi(theta, state, prototypes, knearest)
        action = np.random.choice(len(pi), p = pi)
        state, reward, episodeEnded, _, _ = env.step(action)
        episodeReward += reward
    env.close()
    del env
    return episodeReward

# -args-
# [0]: M
# [1]: envName 
# [2]: theta with shape D x A where D = dim(obs space), A = dim(action space)
# [3]: prototypes for Kanerva coding with shape numPrototypes x D
# [4]: the number of prototypes active at a time
def policyNetworkProcess(args):
    M = args[0]
    envName = args[1]
    theta = args[2]
    prototypes = args[3]
    knearest = args[4]
    episodeRewards = []
    for i in range(M):
        episodeRewards.append(runEpisode((theta, envName, prototypes, knearest, i)))
    return [theta.flatten(), episodeRewards]

def runAllEpisodes(N, M, envName, prototypes, knearest, nJobs):
    np.random.seed(0)
    env = gym.make(envName)
    env.action_space.seed(0)
    numPrototypes = np.shape(prototypes)[0]
    A = env.action_space.n # dimensionality of action space
    env.close()
    space = Space([(-1., 1.) for i in range(numPrototypes * A)])
    lhs = Lhs(lhs_type = "classic", criterion = None)
    thetas = np.array(lhs.generate(space.dimensions, N))
    with Pool(nJobs) as p:
        results = p.map(policyNetworkProcess, [(M, envName, thetas[i].reshape(numPrototypes, A), prototypes, knearest) for i in range(N)])
    paramData = []
    episodeRewardData = []
    for result in results:
        paramData.append(result[0])
        episodeRewardData.append(result[1])
    return paramData, episodeRewardData