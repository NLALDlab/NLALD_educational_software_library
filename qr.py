import numpy as np
from math import *

# Calcolo della fattorizzazione QR con il metodo di Gram-Schmidt Classico :
def CGS(A):
    # INPUT:  A - np.array 2D
    m = A.shape[0] 
    n = A.shape[1]  
    Q = np.zeros([m, n])
    R = np.zeros([n, n])
    for j in range(0,n):  # itero sulle colonne
      Q[:,j] = A[:,j];
      for i in range(0,j):
        R[i,j] = Q[:,i].T @ A[:,j];  # G.S. Classico
        Q[:,j] = Q[:,j] - R[i,j]*Q[:,i];
        #print('Q[:,', j, '] = ', Q[:,j])
      #endfor
      R[j,j] = np.linalg.norm(Q[:,j]);             # norma 2
      if  R[j,j]==0:  break;  #endif       # significa che c'e' dipendenza lineare.
      Q[:,j] = Q[:,j] / R[j,j];            # Q ha colonne ortogonali e di norma 2 unitaria, e dunque e' ortonormale
    #endfor                                
    return Q,R
    
    
# Calcolo della fattorizzazione QR con il metodo di Gram-Schmidt Modificato :
def MGS(A,verbose=False):
    # INPUT:  A - np.matrix
    m = A.shape[0] 
    n = A.shape[1]  
    Q = np.zeros([m, n])
    R = np.zeros([n, n])
    for j in range(0,n):  # itero sulle colonne
      Q[:,j] = A[:,j];
      for i in range(0,j):
        R[i,j] = Q[:,i].T @ Q[:,j];  # G.S. Modificato
        Q[:,j] = Q[:,j] - R[i,j]*Q[:,i];
        #print('Q[:,', j, '] = ', Q[:,j])
      #endfor
      R[j,j] = np.linalg.norm(Q[:,j]);             # norma 2
      if verbose: print("R[",j,",",j,"] = ",R[j,j])
      #if R[j,j] < 1.e-15: break; # NB: ONLY FOR RANK-REVEALING!
      if R[j,j]==0:  break;               # significa che c'e' dipendenza lineare.
      Q[:,j] = Q[:,j] / R[j,j];            # Q ha colonne ortogonali e di norma 2 unitaria, e dunque e' ortonormale
    #endfor                                
    return Q,R


# Calcolo della fattorizzazione QR con le trasformazioni di Householder
# L'algoritmo calcola R sul posto occupato da A, ed i vettori di Householder v_k che definiscono
# univocamente le matrici Q_k
def Householder(A,verbose=False):
    # INPUT:  A - np.array 2D
    m = A.shape[0] 
    n = A.shape[1]  
    V = np.zeros([m, n])
    for k in range(0,n):  # itero sulle colonne
        x = np.atleast_2d(A[k:m,k]).T
        #print("||x||_2 = ",np.linalg.norm(x,2))
        #print("||A[k:m,k]||_2 = ",np.linalg.norm(A[k:m,k],2))
        e_1 = np.zeros((m-k,1)); e_1[0] = 1.0;
        #print("np.sign(float(x[0])) = ",np.sign(float(x[0])))
        segno = 1 if float(x[0]) >= 0 else -1
        v_k = segno*np.linalg.norm(x,2)*e_1 + x;     # v_k e' stato cambiato di segno per maggior efficienza del calcolo
        #print("qr.Householder: v_k.shape = ",v_k.shape)
        #print("||v_k||_2 = ",np.linalg.norm(v_k,2))
        v_k = v_k / np.linalg.norm(v_k,2);
        A[k:m,k:n] = A[k:m,k:n] - 2*v_k@v_k.T@A[k:m,k:n];
        if verbose: print('A = ',A)
        V[k:m,k] = np.squeeze(v_k)
    #endfor
    return V,A
    
    
def Givens_coeff(xp,xq):
    if xq==0.:
      c=1.; s=0.;
    else:
      if abs(xq)>abs(xp):
        r=xp/xq; s=-1./sqrt(1.+r**2); c=-s*r;
      else:
        r=xq/xp; c=1./sqrt(1.+r**2); s=-c*r;
      #endif
    #endif
    return c,s

    
