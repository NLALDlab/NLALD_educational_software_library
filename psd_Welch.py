import numpy as np
from numpy.fft import *
from hamming import *

def psd_Welch(si,Tc=1.0,nsseq=1):
    #
    # metodo di Welch:
    #
    N = int(len(si)/nsseq)
    #print N
    #print "mean(si) = ",np.mean(si)
    # periodogramma modificato:
    s = np.reshape(si,(nsseq,N)).T
    #print s
    w = hamming(N); # finestra di Hamming
    w_ms = np.sum(w*w)/N;  # valore quadratico medio della sequenza di finestratura dei dati
    #print "mean(w_ms) = ",np.mean(w_ms)
    #w = np.asmatrix(w).T; # finestra di Hamming
    tmp = np.repeat(np.atleast_2d(w).T, nsseq, axis=1)
    #print tmp.shape
    s = s * tmp 
    #print "mean(s) = ",np.mean(s)
    #print s
    #print "nsseq=",nsseq, "  N=",N, "w_ms=",w_ms
    S = np.zeros((N,nsseq))
    for i in range(nsseq):
      S[:,i] = abs(fft(s[:,i].T).T)**2
    #endfor
    #print S
    I = np.sum(S,axis=1) / ( nsseq * N * w_ms)
    f = np.arange(0., 1./Tc, 1./(N*Tc))
    return I,f