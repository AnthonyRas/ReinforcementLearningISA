import pandas as pd
import numpy as np
import json
from computeMetadata.performance.getPerformanceData import normalisePerformanceData

def best_returns(n_epochs):
    allEpochData = pd.read_csv("allEpochData.csv")
    # for each algorithm and for each trial of a given algorithm, 
    # take maximum average test return observed in first 'n_epochs' epochs
    # then average this value across the trials of given algorithm
    # do this for each algorithm
    allEpochData.iloc[:, 2:] = allEpochData.iloc[:, 2:].applymap(\
        lambda epochs_string: [max(x[0:n_epochs]) for x in json.loads(epochs_string)]).applymap(\
        lambda epochs_max: np.mean(epochs_max))
    return allEpochData

def normalise_dataframe_row(df, i):
    avRandomScore = df.loc[i, "randomScore"]
    learnerPerformances = dict(df.iloc[i, 2:])
    return {"instances" : df.loc[i, "instances"], 
    **normalisePerformanceData(None, None, learnerPerformances, avRandomScore)}

# NOTE: assumes epoch = 4 in 'algoParameters' object when computing data
for n_epochs in range(1, 4+1):
    df = best_returns(n_epochs)
    df = pd.DataFrame(list(map(lambda i: normalise_dataframe_row(df, i), range(len(df)))))
    df.to_csv("normalisedEpochData_" + str(n_epochs) + ".csv", index = False)


# remove all instances for which no algorithm did better than random
# after all epochs (possible infeasibility)
learner_names = ["algo_PPO", "algo_SACdiscrete", "algo_IQN", "algo_Rainbow"]
performance_data_name = "normalisedEpochData_" + "4" + ".csv"
performance_data = pd.read_csv(performance_data_name)
instance_set = np.where(performance_data.loc[:, learner_names].sum(axis = 1) > 0.0)[0]

for n_epochs in range(1, 4+1):
    performance_data_name = "normalisedEpochData_" + str(n_epochs) + ".csv"
    performance_data = pd.read_csv(performance_data_name)
    feature_data = pd.read_csv("newMetadata.csv")
    # see the document for an explanation of this
    feature_data.drop(['feature_sqrDcorSRtn', 'feature_sqrDcorSSp',
     'feature_sqrDcorSARtn', 'feature_sqrDcorSASpRtn', 'feature_sqrDcorASp',
      'feature_sqrDcorARtn', 'feature_sqrDcorSpRtn', 'feature_sqrDcorASpRtn',
       'feature_sqrDcorSSpRtn'], axis = 1, inplace = True)
    all_metadata = pd.merge(feature_data, performance_data, on = "instances", how = "inner")
    all_metadata = all_metadata.iloc[instance_set]
    # get source of each instance
    all_metadata["source"] = all_metadata["instances"].map(lambda x: x.split("#")[0].split("-")[0])
    all_metadata.to_csv("./metadata/" + str(n_epochs) + "/metadata.csv", index = False)