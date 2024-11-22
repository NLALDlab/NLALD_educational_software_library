import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt


def freqresp(h,stile,tieni=True):
  #
  # data la risposta di un sistema DLTI al campione unitario "h", questa funzione calcola la risposta in frequenza "H" del sistema;
  # se "nfig" e' specificato, crea il grafico di modulo e fase di "H".
  # se "stile" e' specificato, disegna le curve con quello stile (vedi "plot()").
  # se "tieni==True" , il grafico viene sovrapposto alla figura esistente.
  N = len(h)
  H = fft(h)
  plt.figure(1)
  if stile is None:
    stile = 'b-'
  #endif
  fn = np.arange(0.,1.+1./N,2./N)
  ifn = int(N/2)+1
  modulo = 20.*np.log10( abs(H[0:ifn]) )
  fase = np.angle(H[0:ifn])*360./(2.*np.pi)
  plt.figure(1); plt.plot(fn, modulo, stile);
  plt.figure(2); plt.plot(fn, fase, stile);
  return H,fn