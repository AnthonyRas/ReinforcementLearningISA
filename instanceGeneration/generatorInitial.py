import json
import numpy as np
import itertools
from computeMetadata.instancers import instancer

# Takes a list of environment names ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"]
# and a list of integers [4, 8, 8] determining how finely grained the 'grid' is
# and outputs a list of instance names of the form
# environment_name#parameter1_value#parameter2_value#...
def generateInitialInstances(instancerObj : instancer, envNames : list[str],
 ks : list[int]) -> list[str]:
    instanceNames = []
    for nmI in range(len(envNames)):
        # get default instance
        with open("envAttributes.json") as attrDataJSON:
                attrData = (json.load(attrDataJSON))[envNames[nmI]]
        defaultVals = dict({})
        for key in attrData.keys():
            defaultVals[key] = attrData[key]['val']
        defaultVals = np.array(list(defaultVals.values()))
        defaultInstance = instancerObj.writeInstanceName(envNames[nmI], defaultVals)
        instanceNames.append(defaultInstance)
        # generate grid
        gridVals = dict({})
        for key in attrData.keys():
            gridVals[key] = np.linspace(attrData[key]['lo'], attrData[key]['hi'], ks[nmI])    # possible values for each environment parameter, NOTE: assuming that each parameter is a float!!
        grid = np.array(list(itertools.product(*gridVals.values())))
        for i in range(grid.shape[0]):
            instanceNames.append(instancerObj.writeInstanceName(envNames[nmI], grid[i, :]))
    return instanceNames