import numpy as np

def upsample(s,N):
  ls = len(s)
  #print "ls = ", ls
  v = np.vstack((s, np.zeros((N-1, ls))))
  x = np.squeeze( np.reshape(v.T,(1, N*ls)) )
  return x
