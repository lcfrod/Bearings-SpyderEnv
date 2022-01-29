# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:12:31 2021
Updated on  2022-01-29

@author: LuRodrigues
"""
import numpy  as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

print('* * Start of the Process * *')


def compute_chunck_feat_values(chunck):
    """Append multiple values to a key in a dictionary of Max() values """
    # Takes the Max of each column
    max_key      = chunck['ts'].max()
    max_acc_Horz = chunck['acc_horz'].max() 
    max_acc_Vert = chunck['acc_vert'].max() 
    if max_key not in dict_ft:
        dict_ft[max_key] = list()
        
    dict_ft[max_key].extend([max_acc_Horz, max_acc_Vert])
    
    chunck.drop(chunck.index, inplace=True)
    return dict_ft

chunksize = 2560
nchunks = 0
file_name = 'C:/FEMTO/Dataset/Bearing2_1.csv'

dict_ft   = dict()
for chunk in pd.read_csv(file_name ,
                         chunksize = chunksize, 
                         iterator = True) :
    dict_td = compute_chunck_feat_values(chunk) 
    nchunks = nchunks + 1

# Convert the Dictionary into a Dataframe and rename the Columns
e_df = pd.DataFrame(dict_td ) 
e_df = e_df.T
e_df.rename(columns={0: 'max_acc_Horz', 1: 'max_acc_Vert', 2: 'rms_Vert'}, inplace=True)
e_df = e_df.rename_axis('ts').reset_index()

# Save Dataframe to file
e_df.to_csv('C:/Temp/Feat_Bearing2_1.csv', index=False ,encoding='utf-8')

# Smoothing  
ix = np.arange(len(e_df['ts']))

alpha = 0.2
Hraw_signal =  pd.Series(e_df['max_acc_Horz'], ix).astype(np.float64)
Vraw_signal =  pd.Series(e_df['max_acc_Vert'], ix).astype(np.float64)

fitH  = SimpleExpSmoothing(Hraw_signal).fit(smoothing_level= alpha, optimized=False)
fitV  = SimpleExpSmoothing(Vraw_signal).fit(smoothing_level= alpha, optimized=False)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1) # row 1, col 2 index 1

plt.title("Max Horizontal Acceleration")
plt.plot(ix, Hraw_signal, label='Raw',  color = 'blue')
plt.plot(ix, fitH.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')

plt.subplot(1, 2, 2) # index 2
plt.title("Max Vertical Acceleration")
plt.plot(ix, Vraw_signal, label='Raw' ,   color = 'blue')
plt.plot(ix, fitV.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')
plt.show()




print(e_df)
print('* * End of the Process * *')