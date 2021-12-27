# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:12:31 2021

@author: LuRodrigues
"""

# Generate Time Domain Features
# FEMTO_Time_Domain_Features
import numpy  as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

def compute_chunck_feat_values(chunck):
    """Append multiple values to a key in a dictionary of Max() values """
    # Takes the Max of each column
    max_key      = chunck['ts'].max()
    max_acc_Horz = chunck['acc_horz'].max().astype(np.float64)
    max_acc_Vert = chunck['acc_vert'].max().astype(np.float64)
    
    rms_Horz =   np.sqrt(((chunck['acc_horz'] ) ** 2).mean())
    rms_Vert =   np.sqrt(((chunck['acc_vert'] ) ** 2).mean())
    
    kurtosis_Horz =   kurtosis(chunck['acc_horz']) ** 0.25
    kurtosis_Vert =   kurtosis(chunck['acc_vert']) ** 0.25
    
    
 
    if max_key not in dict_ft:
        dict_ft[max_key] = list()
        
    dict_ft[max_key].extend([max_acc_Horz, max_acc_Vert , rms_Horz, rms_Vert, kurtosis_Horz, kurtosis_Vert])
    
    chunck.drop(chunck.index, inplace=True)
    return dict_ft


chunksize = 2560
file_name = 'C:/Temp/Bearing2_1.csv'

dict_ft   = dict()
for chunk in pd.read_csv(file_name ,
                         chunksize = chunksize, 
                         iterator = True) :
    dict_td = compute_chunck_feat_values(chunk) 
    
# Convert the Dictionary into a Dataframe and rename the Columns
e_df = pd.DataFrame(dict_td ) 
e_df = e_df.T
e_df.rename(columns={0: 'max_acc_Horz', 1: 'max_acc_Vert', 2: 'rms_Vert', 3: 'rms_Horz' , 
                     4: 'kurtosis_Horz', 5: 'kurtosis_Vert'}, inplace=True)
e_df = e_df.rename_axis('ts').reset_index()

e_df.to_csv('C:/Temp/Feature2_1.csv', index=False ,encoding='utf-8')


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
    

Hraw_signal = pd.Series(e_df['max_acc_Horz'], ix)
Vraw_signal = pd.Series(e_df['max_acc_Vert'], ix)

alpha = 0.2
Hfit  = SimpleExpSmoothing(Hraw_signal).fit(smoothing_level= alpha, optimized=False)
Vfit  = SimpleExpSmoothing(Vraw_signal).fit(smoothing_level= alpha, optimized=False)

plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1) # row 1, col 2 index 1

plt.title("Smoothed Max Acceleration")
plt.plot(ix, Vfit.fittedvalues, label='Vertical Acc'  , color = 'blue')
plt.plot(ix, Hfit.fittedvalues, label='Horizontal Acc', color = 'red' )
plt.legend(loc='best')
 
plt.show()
    
ix = np.arange(len(e_df['ts']))
alpha = 0.2
Hraw_signal = pd.Series(e_df['rms_Horz'], ix).astype(np.float64)
Vraw_signal = pd.Series(e_df['rms_Vert'], ix)

fitH  = SimpleExpSmoothing(Hraw_signal).fit(smoothing_level= alpha, optimized=False)
fitV  = SimpleExpSmoothing(Vraw_signal).fit(smoothing_level= alpha, optimized=False)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1) # row 1, col 2 index 1

plt.title("RMS Horizontal Acceleration")
plt.plot(ix, Hraw_signal, label='Raw',  color = 'blue')
plt.plot(ix, fitH.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')

plt.subplot(1, 2, 2) # index 2
plt.title("RMS Vertical Acceleration")
plt.plot(ix, Vraw_signal, label='Raw' ,   color = 'blue')
plt.plot(ix, fitV.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')
plt.show()

ix = np.arange(len(e_df['ts']))
alpha = 0.2
Hraw_signal = pd.Series(e_df['kurtosis_Horz'], ix) 
Vraw_signal = pd.Series(e_df['kurtosis_Vert'], ix)  

fitH  = SimpleExpSmoothing(Hraw_signal).fit(smoothing_level= alpha, optimized=False)
fitV  = SimpleExpSmoothing(Vraw_signal).fit(smoothing_level= alpha, optimized=False)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1) # row 1, col 2 index 1

plt.title("Kurtosis Horizontal Acceleration")
plt.plot(ix, Hraw_signal, label='Raw',  color = 'blue')
plt.plot(ix, fitH.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')

plt.subplot(1, 2, 2) # index 2
plt.title("Kurtosis Vertical Acceleration")
plt.plot(ix, Vraw_signal, label='Raw' ,   color = 'blue')
plt.plot(ix, fitV.fittedvalues, label='Smoothed', color = 'red')
plt.legend(loc='best')
plt.show()

Hraw_signal = pd.Series(e_df['kurtosis_Horz'], ix).astype(np.float64)
Vraw_signal = pd.Series(e_df['kurtosis_Vert'], ix).astype(np.float64)

alpha = 0.2
Hfit  = SimpleExpSmoothing(Hraw_signal).fit(smoothing_level= alpha, optimized=False)
Vfit  = SimpleExpSmoothing(Vraw_signal).fit(smoothing_level= alpha, optimized=False)

plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1) # row 1, col 2 index 1

plt.title("Smoothed Kurtosis ^ (1/4)")
plt.plot(ix, Vfit.fittedvalues, label='Vertical Acc'  , color = 'blue')
plt.plot(ix, Hfit.fittedvalues, label='Horizontal Acc', color = 'red' )
plt.legend(loc='best')
 
plt.show()

print('* * End of Process * *')