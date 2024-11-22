import numpy as np

# PERBAS
# Function perx = perbas(x,k)
# Description : PERBAS(x,k) gives the periodic version (with k periods)
#               from a vector x. 
# Example  :    perbas([1 2 3 4 5 6 7 8 9],3)=[1+4+7,2+5+8,3+6+9] 
# Input    :    x: vector
#               k: integer, which divides the length of x
# Output   :    perx=a vector of length n/k
#
# Converse :    periodz (not yet done) 
# 
# Usage    :  y = perbas(x,k);
# 
# See also :  STFT (important use!)  

# Author(s) : H.G. FEICHTINGER,  05/1990, revised W.Reuter, 1999 
# modificato: FM 03/12/2012 tradotto in Python
# Literatur :
#
# Copyright : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
#             http://nuhag.mat.univie.ac.at/
#             Permission is granted to modify and re-distribute this
#             code in any manner as long as this notice is preserved.
#             All standard disclaimers apply.
#
# Externals :
# Subfunctions :  none

# TODO:  make it work for rows OR columns, output same "kind"  HGFei 

def perbas(x,k):
  l = len(x) 
  if l%k != 0:
    print('ERROR: k does not divide the length of l')
  #endif
  m = int(l/k)
  u = np.reshape(x.T,(k,m))
  perx = sum(u)         
  if k == 1:  
    perx = x
  #endif  
  return np.squeeze(perx)