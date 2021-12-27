# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:29:31 2021

@author: Luiz Rodrigues
"""
import matplotlib.pyplot as plt
import pandas as pd
import math
from spectrum import aryule, Periodogram, arma2psd
import numpy as np
from scipy import linalg, signal
from scipy.stats import kurtosis
from scipy.signal import lfilter,hann
from aryule_mat import aryule_mat
import concurrent.futures
import time
import datetime

def AR_signal_separation(p, x, maxP):
    wrkKurt = 0
    e        = np.zeros(len(x))    # Discrete Signal

    if (p % 50) == 0:
        print('p = ', p)
    a, var, reflec = aryule_mat(x, p + 1)         # Using  aryule Matlab 
    #   a, var, reflec = aryule(x, p + 1)         # Using  aryule Python 
    x_toep = np.insert(x, 0, 0)                   # oeplitz Matrix
    padding = np.zeros(p, x_toep.dtype)
    first_col = np.r_[x_toep, padding]
    first_row = np.r_[x_toep[0], padding]
    X_tmp = linalg.toeplitz(first_col, first_row)  # Create Toeplitz Matrix
    X = X_tmp[:-(p + 1), :]                        # Delete the last(n_row+1) lines to customize Toeplitz Matrix
    xp = np.dot(-X, a[1:])
    e = np.subtract(x, xp)
    wrkKurt = kurtosis(e[p+1:], fisher=False)
    return (e, wrkKurt )


def main():
  print ("* * Start FEMTO Processing * * ")
  qty_recs = 2560
  rawData = 'C:/Bearings/Datasets/PHM2012/Bearing3_3.csv'
#  rawData = 'C:/Bearings/Datasets/CWRU/bearing270.csv'
  compute_time = -10.0
  compute_recs = 0
  
  
  # Parameters of PHM2012 Data

  sampRate = 12e3                                         # Sampling Rate (Hz)
  rpm      = 1796                                         # Shaft rotating speed
  bearFail = ['BPFO',  'BPFI',   'FTF',  'BSF' ]
  bearFreq = [3.053 ,  4.947 ,  0.382 ,  1.994 ]          #  BPFO, BPFI, FTF, BSF
  bearFreq[:] = [k * (rpm/60) for k in bearFreq]

  #maxP = math.ceil(sampRate/ np.max(bearFreq))        # Calc max order of AR
  maxP = 82
 
  windLeng = ([2**4, 2**5, 2**6 ])                     # Window Lenght of STFT
  tempKurt = np.zeros(maxP+1)

  # Butterworth digital filter Parameters
  freqRange = [5.3e3,  5.9e3]                          # Band-pass filter parms
  filter_order = 2
 
 
  for df in pd.read_csv(rawData ,
 #                      header=None,
                        usecols=[2] ,      # acc_vert
                        chunksize = qty_recs, 
                        iterator = True) :
     compute_time = compute_time + 10
     compute_recs = compute_recs + len(df)
     x = df.to_numpy()                            # Set the Column name
     N = len(x)
 
     # 1- Plot the Raw Signal  Kurtosis
     rawKurt = kurtosis(x[0:], fisher=False)
     fig, (ax1) = plt.subplots(1,1)
     fig.set_size_inches(9, 3)
     ax1.set_ylabel('Amplitude' ) 
     ax1.set_xlabel('Time (s)')
     fig.suptitle('Raw Signal - Kurtosis = '+ '{:.5f}'.format(float(rawKurt)) )
     t = np.linspace(0, 1, len(x))
     ax1.plot(t, x) 
     
     
     # ===========   Discrete signal separation  using Auto Regressive Model  ====
     #
     tempKurt = np.zeros(maxP+1)    # work array of Kurtosis
     e        = np.zeros(len(x))    # Discrete Signal
     for p in range(1, maxP+1):
         e,tempKurt[p] = AR_Signal_Separation(p,x, maxP)

    

    
     break
     
     
     
    
 
  
  print(str(datetime.timedelta(seconds=compute_time )))
  print(compute_recs)
  
  
  
if __name__== "__main__":
  main()
