import numpy as np
import numpy.matlib as npmat
from scipy.sparse import *
from scipy.sparse.linalg import *

def f0(t): # forzante di tipo "flash" applicata al nodo 0
  t = t * 5e-04
  W = 6.6296e+006
  sig = 0.010
  Q = (W*t) * np.exp(-np.sqrt(t)/sig)
  return Q

def g0(n, Dx):
  # supponiamo che ci sia temperatura uniforme pari a 20 gradi
  T0 = 20.0 * np.ones(n+1)
  return T0

def create_fd_matrix(n,Dx):
  A = lil_matrix((n+1,n+1))
  for i in range(1,n+1): # nodi interni: discretizzazione del laplaciano
    A[i,i] = -2./Dx**2
    if i > 0:      A[i,i-1] = 1./Dx**2   #endif
    if i < n:      A[i,i+1] = 1./Dx**2   #endif
  #endfor
  # devo imporre le CB: quindi aggiungo le equazioni corrispondenti
  A[0,0] = -1./Dx; A[0,1] = 1./Dx
  A[n,n] = -1./Dx; A[n,n-1] = 1./Dx
  return A

def calore_1d_fd(Lungh,n,Dx,N,eulero):
  # Approssimazione del seguente problema:
  # Dato (t,x) in [t0, tf] x [0,L], con L variabile, sia h il passo temporale
  # di discretizzazione di [t0, tf] e Dx il passo di [0,L].
  # equazione del calore: T=T(t,x) temperatura in [t0, tf] x [0,L]
  # dT/dt - c (d^2)T/d (x^2)=0
  # dT/dx(t,0)=f0(t), forzante(flash) nota
  # dT/dx (t,L)=0, condizioni di Neumnann omogenee
  # T(t0,x)=g0(x), condizioni iniziali note
  # problema discretizzato nello spazio: Ti(t)=T(t,xi), xi=i*Dx;
  # T'(t)=c A T(t)+b(t), T(t)=[T0(t),T1(t),...,Tn(t)]', b(t)=[f0(t), 0,...,0]' ; (ODE)
  # T(t0)=g0 <=> [T0(t0),T1(t0),...,Tn-1(t0)]' = [g0(x0),g0(x1),...,g0(xn-1),g0(xn)]';
  c = 1.
  tfin = 10.
  tin = 0.
  dt = (tfin-tin)/N
  A = lil_matrix((n+1,n+1))
  for i in range(1,n+1): # nodi interni: discretizzazione del laplaciano
    A[i,i] = -2./Dx**2
    if i > 0:      A[i,i-1] = 1./Dx**2   #endif
    if i < n:      A[i,i+1] = 1./Dx**2   #endif
  #endfor
  # devo imporre le CB: quindi aggiungo le equazioni corrispondenti
  A[0,0] = -1./Dx; A[0,1] = 1./Dx
  A[n,n] = -1./Dx; A[n,n-1] = 1./Dx
  # otteniamo quindi l'equazione discretizzata nello spazio:
  # T'(t)=c A T(t)+b(t), T(t)=[T0(t),T1(t),...,Tn(t)]', b(t)=[f0(t), 0,...,0]' ; (ODE)
  # T(t0)=g0 <=> [T0(t0),T1(t0),...,Tn-1(t0)]' = [g0(x0),g0(x1),...,g0(xn-1),g0(xn)]';
  # Osservazione: b(t) puo' essere pensato come l'ingresso del modello (u=b(t))
  # quindi otteniamo il seguente sistema state-space:
  # T'(t)=c A T(t)+b(t);
  # T_vero_x0(t)=T0(t)=C T(t), C=[1,0,...,0]; <=> serie temporale delle temperature raccolte in x0
  # Quindi l'ordine nx del modello state space e' n+1.
  # per risoluzione dell'(ODE) uso un theta metodo
  T_old = np.atleast_2d(g0(n,Dx)).T; # CI (per k=0)
  T_x0 = np.zeros(N+1)
  T_hist = np.zeros((n+1,N+1))
  T_hist[:,[0]] = T_old
  T_x0[0] = T_old[0]
  M1 = np.eye(n+1)-dt*c*A
  print(M1[0:4,0:4])
  if eulero == 0:  # esplicito
    for k in range(N):
      b = np.zeros((n+1,1))
      b[0] = f0(k*dt+tin)
      T = dt * (c*A*T_old+b)+T_old  # T(k+1)=(I+hcA)T(k)+h*b
      T_old = T.copy()
      T_x0[k+1] = T_old[0,0]
    #endfor
  elif eulero == 1:  # implicito
    for k in range(N):
      b = np.zeros((n+1,1))
      b[0] = f0(k*dt+tin)
      T = np.linalg.solve(M1, dt*b+T_old)  # T(k+1)=(I-dt*c*A)^{-1}T(k)+(I-dt*c*A)^{-1}*dt*b
      T_hist[:,[k+1]] = T 
      T_old = T.copy()
      T_x0[k+1] = T_old[0,0]
    #endfor
  #endif
  return T_x0,M1,dt,T_hist