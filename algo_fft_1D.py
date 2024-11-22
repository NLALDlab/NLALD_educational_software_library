import numpy as np
from math import *

def fft_1D(u):
  # Implementazione dell'algoritmo della Fast Fourier Transform, a
  # decimazione nel tempo e con la disposizione della sequenza di ingresso a
  # disposizione invertita dei bit, quindi sfrutta il calcolo "sul posto".
  N = len(u)
  # e' necessario dichiarare che "u" e' un vettore di numeri complessi, anche se il vettore di dati e' di numeri reali, perche' alla fine conterra' i coeff. della DFT.
  u = np.array(u, dtype=complex)  
  #
  # riordinamento dei dati a disposizione invertita dei bit:
  j = int(0)
  for n in range(0,N):
    if N < 32:
      print('n=', np.binary_repr(n,int(log(N,2))), '   j=', np.binary_repr(j,int(log(N,2))))
    #endif
    if j > n:
      temp = u[j]
      u[j] = u[n]
      u[n] = temp
    #endif
    # calcolo il prossimo "j"
    k = int(N/2)
    while k <= j and k>0:
      j = j - k
      k = k / 2
    #endwhile
    j = int(j + k)
  #endfor
  #
  # calcolo "sul posto" della DFT  
  # NB: qui "i" e "j" sono usate come variabili, non sono l'unita' immaginaria !
  for s in range(1,int(log(N,2))+1):  # "s" indica lo stadio corrente
    c = 1.+0.j  # W_N^0
    w = cos(pi/2**(s-1))-1.j*sin(pi/2**(s-1))  # W_N^{2^{log2(N)-s}}
    for k in range(0,2**(s-1)): # "k" indica il numero di farfalle interlacciate
      for i in range(k, N, 2**s): # "i" e' l'indice del nodo alto della k-esima farfalla di ogni gruppo.
        # calcolo della farfalla
        ip = i + 2**(s-1)
        #print 'i=', i, '  ip=', ip, '  c=', c
        temp = u[ip] * c
        u[ip] = u[i] - temp
        u[i] = u[i] + temp
      #endfor
      c = c * w
    #endfor
  #endfor
  return u
