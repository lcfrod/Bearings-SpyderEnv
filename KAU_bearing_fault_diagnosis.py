# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:50:16 2021

@author: Luiz Rodrigues
"""
# ---------------------------------------
# KAU_Bearing_Fault_Diagnosis_01
# Korea Aerospace University(KAU) data
# Author : Luiz Rodrigues
# Date   : 01-Nov-2021
# ---------------------------------------
import matplotlib.pyplot
from pandas import read_csv
import math
from spectrum import aryule, Periodogram, arma2psd
import numpy as np
from scipy import linalg, signal
from scipy.stats import kurtosis
from scipy.signal import lfilter,hann
from aryule_mat import aryule_mat
import matplotlib.pyplot as plt
import concurrent.futures
import time

print('Processing Bearing signal')

def AR_Signal_Separation(p):
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
    tempKurt[p] = kurtosis(e[p+1:], fisher=False)
    return (e, tempKurt )

# Parameters of Korea Aerospace University(KAU) Data
rawData = 'C:/Bearings/Datasets/KAU/bearing1.csv'    # Raw data to Load
sampRate = 51.2e3                                    # Sampling Rate (Hz) 
rpm      = 1200                                      # Shaft rotating speed
bearFail = ['BPFO',  'BPFI' ,   'FTF',  'BSF' ]
bearFreq = [4.4423,  6.5577 ,  0.4038, 5.0079 ]      #  BPFO, BPFI, FTF, BSF 
bearFreq[:] = [k * (rpm/60) for k in bearFreq]

#maxP = math.ceil(sampRate/ np.max(bearFreq))        # Calc max order of AR
maxP = 390
 
windLeng = ([2**4, 2**5, 2**6, 2**7, 2**8 ])         # Window Lenght of STFT  
tempKurt = np.zeros(maxP+1)

# Butterworth digital filter Parameters
freqRange = [1.8e4,  2.3e4]                          # Band-pass filter parms
filter_order = 2

#  Read the Bearing data signal
df = read_csv(rawData, header=0)                    # Read data file
x, y, z = np.split(df, [int(.2*len(df)), int(.5*len(df))])
x = df['vib'].to_numpy()                            # Set the Column name
N = len(x)

# 1- Plot the Raw Signal  Kurtosis
rawKurt = kurtosis(x[0:], fisher=False)
fig, (ax1) = plt.subplots(1,1)
fig.set_size_inches(9, 3)
ax1.set_ylabel('Amplitude' ) 
ax1.set_xlabel('Time (s)')
fig.suptitle('Raw Signal - Kurtosis = ' +  '{0:.7g}'.format(rawKurt))
t = np.linspace(0, 1, len(x))
ax1.plot(t , x) 

# ===========   Discrete signal separation  using Auto Regressive Model  ====
tempKurt = np.zeros(maxP+1)    # work array of Kurtosis
e        = np.zeros(len(x))    # Discrete Signal
for p in range(1, maxP+1):
    e,tempKurt = AR_Signal_Separation(p)

print(p)

# Selects the AR Model with Maximum Kurtosis
optP = (np.where(tempKurt == np.amax(tempKurt)))    # Selects the Max Kurtosis
optP = int(optP[0])                      # Converts tuple to int
optA, var, reflec = aryule_mat(x, optP)  # Convert optP Tuple to int
optA[0] = 0
xp = lfilter((-1 * optA), 1, x)
e = np.subtract(x[optP:], xp[optP:])     #  Residual signal = Raw - Max Kurtosis

# 2 - Plot Maximum Kurtosis
fig, (ax2) = plt.subplots(1,1)
fig.set_size_inches(8, 4)
ax2.set_xlabel('AR Model Order(p)')
ax2.set_ylabel('Kurtosis')
fig.suptitle('Maximum Kurtosis Point = ' +  '{0:.7g}'.format(tempKurt[optP])   )
t = np.linspace(0, maxP-1 , maxP)
ax2.plot(optP,tempKurt[optP],  marker="*", markersize=10, 
         markeredgecolor="red", markerfacecolor="black")
ax2.plot(t , tempKurt[1:]) 

# 3 - Plot Residual Signal for Maximum Kurtosis
fig, (ax3) = plt.subplots(1,1)
fig.set_size_inches(8, 3)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
fig.suptitle('Residual Signal for Max Kurtosis '  +  '{0:.7g}'.format(tempKurt[optP]))
t = np.linspace(0, 2, len(e))
ax3.plot(t , e)  

# ================  Demodulation Band Selection (STFT & Spectral Kurtosis ) ==
Ne = len(e)
numFreq = int(max(windLeng) + 1)
specKurt = np.zeros((len(windLeng), numFreq))
lgd = []
for i in range(0, len(windLeng)):
    windFunc = signal.hann(windLeng[i])  # Short Time Fourier Transform
    numOverlap = np.fix( windLeng[i] / 2)
    numWind = int(np.fix((Ne-numOverlap)/(windLeng[i] - numOverlap)))
    n = list(range(0, windLeng[i]+1 ))
    STFT = np.zeros((numWind , numFreq))
    for t in range(1,numWind + 1):
        ini  = int(n[0])    # Start of the Window
        fin  = int(n[-1])   # End of the Window
        stft = np.fft.fft( np.multiply(e[ini:fin], windFunc), 2*(numFreq -1 ))
        stft = np.abs(stft[0:numFreq])/windLeng[i]/np.sqrt(np.mean(windFunc**2))*2
        STFT[t-1,:] = stft.T     # Transposta
        n = n + (windLeng[i] - numOverlap)

    # Spectral Kurtosis
    for j in range(0,numFreq):
        specKurt[i,j] = (np.mean(np.abs(STFT[:,j])**4 ) / (np.mean(np.abs(STFT[:,j]) ** 2)) **2) - 2

    lgd.append( ['Window size : '+ str(int(windLeng[i]))] )
    
# 4- Plot Spectral kurtosis of STFT Windows
freq = ( np.arange(0,numFreq,1) / (2 * (numFreq -1)  ) * sampRate)
fig,(ax4) = plt.subplots(1,1)
fig.set_size_inches(8, 4)
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Spectral kurtosis')
for i in range(0, len(windLeng)):
    ax4.plot(freq , specKurt[i], label = ', '.join(lgd[i]) ) 
ax4.legend();

# Apply Butterworth digital filter on Residual Signal 
#freqRange = [1.8**4 , 2.3**4]    # Band-pass filter best parameters
#filter_order = 2

b, a = signal.butter(filter_order , [freqRange[0]/(sampRate/2), freqRange[1]/(sampRate/2)], btype='bandpass')
X = lfilter(b, a, e)  # Band-passed signal
kurtoX =  kurtosis(X[0:], fisher=False)

# 5- Plot Vibration Signal after  band-pass filtering
fig, (ax5) = plt.subplots(1,1)
fig.set_size_inches(8, 3)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('SK Filtered Signal')
fig.suptitle('Vibration Signal after Band-pass Filtering. Kurtosis = ''{0:.7g}'.format(kurtoX))  
t = np.linspace(0, 2, len(e))
ax5.plot(t , X) 

# ===========  Envelope Analysis =================
aX = signal.hilbert(X)          # hilbert(x) - returns an analytic signal of x
envel = abs(aX)
envel = envel - np.mean(envel)   # envelope signal
fftEnvel =  abs(np.fft.fft(envel)) / Ne *2
fftEnvel = fftEnvel[0:math.ceil(N/2)]

# 6 ---------   Plot Bearing Failure Frequencies  -----------------
freq = ( np.arange(0,Ne-1,1) / Ne * sampRate)
freq = freq[0:math.ceil(N/2)]  
scale_factor = 10**2
fig, (ax6) = plt.subplots(1,1)
ax6.set_xlim(0,max(bearFreq) * 2)
y_data_scaled = [y * scale_factor for y in fftEnvel]
ax6.set_ylabel("Amplitude [g] x 10^2")
ax6.set_xlabel("Frequency [Hz]")
ax6.stem(freq,y_data_scaled, 'b', markerfmt='bo', basefmt=" ", label="Signal")
xx,yy = np.meshgrid(bearFreq, plt.ylim(), sparse=False,)
ax6.plot(xx,yy, label=bearFail,  marker='o', ls='--' )
ax6.legend()

print('* * * End of Process * * *')