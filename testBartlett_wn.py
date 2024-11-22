import numpy as np
from psd_Welch import *

def testBartlett_wn(s):
    #
    # test di Bartlett (o del periodogramma cumulato): serve per verificare se una sequenza e' rumore bianco
    #
    # Una sequenza x(k) generata da rumore bianco presenta una sequenza di autocorrelazione in cui tutti i
    # campioni per shift > 0 sono nulli. Quindi, osservare valori della sequenza di auto-correlazione
    # significativamente (in senso statistico) non-nulli corrisponde alla presenza di componenti
    # non-random nella sequenza x(k). Per individuare componenti periodiche mischiate al rumore e' 
    # vantaggioso considerare il "periodogramma cumulato":
    #
    N = len(s)
    #
    [Pss, f] = psd_Welch(s)
    est_mean_s = np.mean(s)
    est_var_s = np.sum(np.power(s - est_mean_s,2))/(N-1)
    CPss = np.zeros(N)
    for i in range(N):
      CPss[i] = sum(Pss[0:i+1]) / (N * est_var_s)
    #endfor
    return CPss, f