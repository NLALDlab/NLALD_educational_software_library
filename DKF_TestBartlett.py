import numpy as np
from scipy import fft

def TestBartlett(Signal):
    """
    Test of Bartlett (or cumulative periodogram): used to check if a sequence is white noise.
    """
    N = len(Signal)

    # Compute PSD with Welch's method
    T = 1
    SeqNum = 1

    NSeq = N // SeqNum
    Signal = Signal.reshape((NSeq, SeqNum))
    Win = 0.54 - 0.46*np.cos(2*np.pi*np.arange(NSeq)/(NSeq-1))  # Hamming window
    WinNorm = np.linalg.norm(Win)**2 / NSeq  # RMS value of the window sequence
    Signal *= Win[:, np.newaxis]
    PSD = np.sum(np.abs(fft(Signal, axis=0))**2, axis=1) / (SeqNum * NSeq * WinNorm)
    Freq = np.arange(0, 1/(2*T), 1/(NSeq*T))

    # Compute normalized cumulative PSD
    EstMean = np.mean(Signal)
    EstVar = np.sum((Signal - EstMean)**2) / (N - 1)

    CumulativePeriodogram = np.cumsum(PSD[1:len(Freq)]) / (N * EstVar)
    CumulativePeriodogram = np.concatenate(([0], CumulativePeriodogram / CumulativePeriodogram[-1]))

    return CumulativePeriodogram, Freq
