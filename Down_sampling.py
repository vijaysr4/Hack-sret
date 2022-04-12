import pandas as pd
import numpy as np
from sklearn.utils import resample

df = pd.read_csv("D:/Hack_sert/water_potability.csv")

df_P = df[df.iloc[:,-1]==1]
df_NP = df[df.iloc[:,-1]==0]

print(df_P)
print(df_NP)

df_NP = resample(df_NP, replace=False, n_samples=1278)
print(df_P)
print(df_NP)

df_down_sampled = pd.concat([df_NP, df_P])
print(df_down_sampled)
df_down_sampled.to_csv('D:/Hack_sert/Downsampled_data.csv')