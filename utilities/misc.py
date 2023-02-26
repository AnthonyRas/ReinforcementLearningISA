import os
import gym
import pandas as pd

# if a CSV file exists then append a row, otherwise generate a CSV file
def writeIfExists(nameOfFile : str, dataframe : pd.DataFrame) -> None:
        if os.path.isfile(nameOfFile):
            dataframe.to_csv(nameOfFile, mode = "a", header = False, index = False)
        else:
            dataframe.to_csv(nameOfFile, index = False)

# set environment parameters
# 'property' is used here so that value of attribute is not overwritten
# when inheriting classes are initialised, so that dependent attributes
# (e.g. self.total_mass = self.masspole + self.masscart) also change
def setAttributes(updateAttrs : dict) -> None:
    for attr in updateAttrs.keys():
        setattr(gym.Env, attr, property(lambda self,\
             x = updateAttrs[attr]: x, lambda self, v: None))

# resets environment parameters
def resetAttributes(updateAttrs : dict) -> None:
    for attr in updateAttrs.keys():
        setattr(gym.Env, attr, None)