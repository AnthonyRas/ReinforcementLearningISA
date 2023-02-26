import math
import numpy as np
from itertools import chain
import pandas as pd
from computeMetadata.features.getPolicyNetworkData import *
import os
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis

def euclidDist(paramVectorA, paramVectorB):
    return np.linalg.norm(paramVectorB - paramVectorA)

def computePOIC(N, M, paramData, episodeRewardData, eta):
    rmax = max(sum(episodeRewardData, [])) # compute sample maximum of episode rewards, across all episodes run
    pI = [1/M*sum(math.exp((r - rmax - 1e-9)/eta) for r in episodeRewardData[i]) for i in range(N)] # - 1e-9 to avoid log(0) errors
    p = 1/N*sum(pI)
    oe = -p*math.log(p) - (1-p)*math.log(1-p) # optimality entropy
    mcoe = 1/N*sum(-pI[i]*math.log(pI[i]) - (1 - pI[i])*math.log(1 - pI[i]) for i in range(N)) # mean conditional optimality entropy
    return oe - mcoe

def computePIC(N, M, paramData, episodeRewardData, B):
    rmin = min(sum(episodeRewardData, []))
    rmax = max(sum(episodeRewardData, []))
    bins = np.linspace(rmin, rmax, num = B+1)
    p = (np.histogram(episodeRewardData, bins)[0] + 1e-9)/(N*M + (1e-9)*B) # posterior mean estimates of bin probabilities, given Dirichlet prior
    pI = [list((np.histogram(episodeRewardData[i], bins)[0] + 1e-9)/(M + (1e-9)*B)) for i in range(N)]
    re = -sum(p[b]*math.log(p[b]) for b in range(B)) # return entropy
    mcre = -1/N*sum(sum(pI[i][b]*math.log(pI[i][b]) for b in range(B)) for i in range(N))
    return re - mcre

def computeFitnessLandscape(N, paramData, episodeRewardData):
    allEuclidDistances = []
    for i in range(len(paramData)):
        iEuclidDistances = []
        for j in range(len(paramData)):
                iEuclidDistances.append(euclidDist(paramData[i], paramData[j]))
        allEuclidDistances.append(iEuclidDistances)
    X = np.array(list(chain(*allEuclidDistances)))
    X = X[np.nonzero(X)]
    allEuclidDistances = np.array(allEuclidDistances)
    neighbourArray = (allEuclidDistances <= np.percentile(X[np.nonzero(X)], 15)) # defines epsilon for neighbourhood
    neighboursOf = []
    for i in range(N):
        iNeighboursOf = []
        for j in range(N):
            if i != j and neighbourArray[i, j]:
                iNeighboursOf.append(j)
        neighboursOf.append(iNeighboursOf)
    numNeighbours = np.sum(neighbourArray, axis = 1) - 1
    fitnesses = np.mean(episodeRewardData, axis = 1)
    return(fitnesses, neighboursOf, numNeighbours)

def computeNeutrality(fitnesses, neighboursOf, numNeighbours, epsilonF):
    return 1/np.count_nonzero(numNeighbours)*sum(1/numNeighbours[i]*sum(abs(fitnesses[j] - fitnesses[i]) <= epsilonF for j in neighboursOf[i]) for i in np.nonzero(numNeighbours)[0])

def computeLocalOptimaProportion(fitnesses, neighboursOf, numNeighbours, isStrict, isMaxima):
    if isMaxima:
        if isStrict:
            return 1/np.count_nonzero(numNeighbours)*sum((1 == 1/numNeighbours[i]*sum(fitnesses[i] > fitnesses[j] for j in neighboursOf[i])) for i in np.nonzero(numNeighbours)[0])
        else:
            return 1/np.count_nonzero(numNeighbours)*sum((1 == 1/numNeighbours[i]*sum(fitnesses[i] >= fitnesses[j] for j in neighboursOf[i])) for i in np.nonzero(numNeighbours)[0])
    else:
        if isStrict:
            return 1/np.count_nonzero(numNeighbours)*sum((1 == 1/numNeighbours[i]*sum(fitnesses[i] < fitnesses[j] for j in neighboursOf[i])) for i in np.nonzero(numNeighbours)[0])
        else:
            return 1/np.count_nonzero(numNeighbours)*sum((1 == 1/numNeighbours[i]*sum(fitnesses[i] <= fitnesses[j] for j in neighboursOf[i])) for i in np.nonzero(numNeighbours)[0])

def computePolicyNetworkFeatures(envName, N, M, prototypes, knearest, instanceName, nJobs):
    paramData, episodeRewardData = runAllEpisodes(N, M, envName, prototypes, knearest, nJobs)
    B = 2*M
    PIC = computePIC(N, M, paramData, episodeRewardData, B)
    f = lambda eta: -computePOIC(N, M, paramData, episodeRewardData, eta)
    POIC = -minimize_scalar(f, bounds = (1, 1e+6), method = 'bounded').fun
    RwD = pd.DataFrame(episodeRewardData)
    y = RwD.mean(axis = 1)
    fitnesses, neighboursOf, numNeighbours = computeFitnessLandscape(N, paramData, episodeRewardData)
    neutrality = computeNeutrality(fitnesses, neighboursOf, numNeighbours, 0)
    features = {"feature_PIC" : PIC, "feature_POIC" : POIC,
     "feature_neutrality" : neutrality,
     "feature_lostrictmax" : computeLocalOptimaProportion(fitnesses, neighboursOf, numNeighbours, isStrict = True, isMaxima = True),
     "feature_lomax" : computeLocalOptimaProportion(fitnesses, neighboursOf, numNeighbours, isStrict = False, isMaxima = True),
     "feature_lostrictmin" : computeLocalOptimaProportion(fitnesses, neighboursOf, numNeighbours, isStrict = True, isMaxima = False),
     "feature_lomin" : computeLocalOptimaProportion(fitnesses, neighboursOf, numNeighbours, isStrict = False, isMaxima = False),
     "feature_yskew" : skew(y), "feature_ykurtosis" : kurtosis(y)}
    y = pd.DataFrame(y).transpose()
    y.insert(0, 'instances', instanceName)
    nameOfFile = "allYs.csv"
    if os.path.isfile(nameOfFile):
        y.to_csv(nameOfFile, mode = "a", header = False, index = False)
    else:
        y.to_csv(nameOfFile, index = False)
    return features