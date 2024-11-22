import numpy as np
from perbas import *
from numpy.fft import *

#  STFT.M   H.G.Feichtinger,   5-23/25-1990 , Aug. 92  
# 
#  determines the STFT of a vector  over a lattice 
#  with lattice constants  a (time) and b (frequ)
#
#  STFT_f(ak,bm) is the output
#
#  USAGE: ST =  STFT(signal,window,a,b)
# 
#  See also:  ISTFT, TFFILT, PERBAS 
#  
# modificato: FM 18/11/2006 per funzionare con i programmi di GD
#             FM 03/12/2012 tradotto in Python

def stft(x,w,a=1,b=1):
  n = len(x)
  res = np.zeros((n/a,n/b)) 
  w1 = np.concatenate((w, np.zeros(n-len(w)))) 
  ww1 = np.concatenate((w1, w1))  
  for jj in range(n/a): 
    y  = x * ww1[ (n - jj*a) : (2*n - jj*a) ]
    y1 = perbas(y,b)
    v = fft(y1)
    res[(jj+n/a/2) % (n/a)][:] = v 
  #endfor
  stf = res.T 
  return stf
