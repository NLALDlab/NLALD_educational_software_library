import numpy as np
from math import *

def hamming(N):
  w = 0.54 - 0.46 * np.cos( 2*np.pi*np.array(np.arange(0,N))/(N-1));
  return w
