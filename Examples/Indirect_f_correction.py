# Change the threshold from 1e-4 to 0.1
# Correct the former simulation data
import pandas as pd
import numpy as np 
import re
from tqdm import tqdm
data_Community_augmentation='simulation_data/Richness_augmentation_p0.1_2018_04_16-13-52.csv'
data_name=data_Community_augmentation
df = pd.read_csv(data_name)
print (set(df.keys()))

data_indirect=df[(df['Indirect failure']==1)]
length=(len(data_indirect.index))
for i in tqdm(range(length)):
    index=data_indirect.index[i]
    array1=list(data_indirect[data_indirect.index==index]['Survie species abuncance'])[0]
    array1=np.asfarray(re.findall(r"[-+]?\d*\.\d+|\d+", array1))
    if (index-1) in df.index:
        array2=list(df[df.index==index-1]['Survie species abuncance'])[0]
        array2=np.asfarray(re.findall(r"[-+]?\d*\.\d+|\d+", array2))
        if np.allclose(np.sum(array1), np.sum(array2), rtol=1e-01, atol=1e-03) and df.iloc[index]['step']==df.iloc[index-1]['step']+1:
            df.loc[index,['Indirect failure']]=0
            df.loc[index,['Rejection failure']]=1
file_name='simulation_data/Richness_augmentation_p0.1_2018_04_16-13-52_correction.csv'          
df.to_csv(file_name, index=False, encoding='utf-8')   