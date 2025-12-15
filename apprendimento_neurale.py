import numpy as np
import matplotlib.pyplot as plt
from math import *

def sigmoid_logistic(x):
  f = 2./(1.0+np.exp(-2.*x)) - 1.0
  return f

def sigmoid_logistic_derivative(x):
  Df = 2.0*np.exp(-2.*x) / ((1.0+np.exp(-2.*x))**2)
  return Df

def apprendimento_neurale(p,input,target,toll,graphics=False,maxiter=500000,eta=0.005,decim_ris_ratio=1000,n_layers=1):
  # questa funzione implementa l'algoritmo di backpropagation per reti neurali
  # FF di due layers con funzioni di attivazione di tipo sigmoide
  n,N = input.shape
  m,N = target.shape
  W1 = 0.001 * np.random.rand(p,n+1)
  W2 = 0.001 * np.random.rand(m,p+1)
  iter = 0 
  grad_sqnorm_hist = np.zeros(ceil(maxiter/decim_ris_ratio)+1)
  err_pred = np.inf + np.zeros((m,N))
  err_pred_hist = np.zeros(ceil(maxiter/decim_ris_ratio)+1)
  target_pred = np.zeros(target.shape)
  target_pred_hist = np.zeros((N,ceil(maxiter/decim_ris_ratio)+1))
  delta2 = np.zeros(m)
  deltaW2 = np.zeros(W2.shape)
  delta1 = np.zeros(p)
  deltaW1 = np.zeros(W1.shape)
  E = np.inf
  while E>toll and iter<maxiter:
    iter = iter + 1
    if False:
        if iter%1000 == 0:  print("iter = ",iter," , E = ",E)
    #endif
    #print "iter = ", iter
    for k in range(N):
      if n==1:
        v1input = np.atleast_2d([1., input[0,k]]).T
      else:
        v1input = np.atleast_2d(np.concatenate((np.array([1.]), np.squeeze(input[:,k].T)))).T
      #endif
      s1 = W1 @ v1input
      if n_layers==1:
          if m == 1:
              target_pred[:,[k]] = np.sum(sigmoid_logistic(s1))
          elif m == p:
              target_pred[:,[k]] = np.sum(sigmoid_logistic(s1))
          else:
              print("ERROR: output dimension mismatch!")
          #endif
          for i in range(m):
            err_pred[i,k] = target[i,k] - target_pred[i,k]
          #endfor
          backprop_err = err_pred[:,k]
          if m == 1: backprop_err = np.repeat(backprop_err,p)
          #
          for i in range(p):
            delta1[i] = backprop_err[i] * sigmoid_logistic_derivative(s1[i])
          #endfor
      else:
          z = sigmoid_logistic(s1)
          #print("z = ",z)
          v1z = np.atleast_2d(np.concatenate((np.array([1.]), np.squeeze(np.array(z.T))))).T
          #print("p = ",p)
          #print("W2 = ",W2)
          #print("v1z = ",v1z)
          s2 = W2 @ v1z
          #print("s2 = ",s2)
          target_pred[:,[k]] = sigmoid_logistic(s2)
          for i in range(m):
            err_pred[i,k] = target[i,k] - target_pred[i,k]
            delta2[i] = err_pred[i,k] * sigmoid_logistic_derivative(s2[i])
          #endfor
          grad_2 = - 2. * np.atleast_2d(delta2).T @ v1z.T
          #print("grad_2.shape = ",grad_2.shape)
          deltaW2 = - eta * grad_2
          W2 = W2 + deltaW2
          #
          backprop_err = W2[:,0:p].T @ delta2
          #
          for i in range(p):
            delta1[i] = backprop_err[i] * sigmoid_logistic_derivative(s1[i])
          #endfor
      #endif
      grad_1 = - 2. * np.atleast_2d(delta1).T @ v1input.T 
      #print("grad_1.shape = ",grad_1.shape)
      deltaW1 = - eta * grad_1
      #
      W1 = W1 + deltaW1
      #print 'iter = ', iter, '   input = ', input[:,k], '   err_pred = ', err_pred[:,k]
      if n_layers==1:
        tmpgrad = np.reshape(grad_1,p*(n+1))
      else:
        tmpgrad = np.hstack((np.reshape(grad_1,p*(n+1)),np.reshape(grad_2,m*(p+1))))
      #endif
      grad_sqnorm_hist[int(iter/decim_ris_ratio)] = tmpgrad @ tmpgrad.T
    #endfor
    if iter%decim_ris_ratio == 0: 
        target_pred_hist[:,int(iter/decim_ris_ratio)] = target_pred
        if graphics:
            plt.figure(1); plt.clf(); plt.title("iter = "+str(iter))
            plt.figure(1); plt.plot(np.squeeze(target),'b.-'); 
            plt.plot(np.squeeze(target_pred),'r.'); 
            plt.show()
        #endif
    #endif
    E = np.sum(np.sum(np.power(err_pred,2))) / N
    #if iter % 1000 == 0:
    #  print  "iter = ", iter, "   E = ", E
    #endif
    if iter%decim_ris_ratio == 0: err_pred_hist[int(iter/decim_ris_ratio)] = E
  #endwhile
  err_pred_hist = err_pred_hist[0:int(iter/decim_ris_ratio)+1]
  target_pred_hist = target_pred_hist[:,0:int(iter/decim_ris_ratio)+1]

  return W1,W2,err_pred_hist,target_pred_hist,grad_sqnorm_hist,iter
