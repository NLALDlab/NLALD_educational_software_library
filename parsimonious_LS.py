import numpy as np
import matplotlib.pyplot as plt

def parsimonious_LS(vk,b,maxgp,stderr=0.0):
  # "A": matrix of the linear model
  n = len(vk)
  Ai = np.vander(vk)
  n2res = np.zeros(maxgp+1)
  res_liminf = np.zeros(maxgp+1)
  estvar_errpred = np.zeros(maxgp+1)
  estvar_parest = np.zeros((maxgp+1,maxgp+1))
  for gradopol in range(maxgp+1):
    A = Ai[:,range(n-(gradopol+1),n)]; #print("A = \n",A)
    #print "A.shape: ",A.shape
    Qh,Rh = np.linalg.qr(A.copy())
    x = np.linalg.solve( Rh , Qh.T@b )
    if False: #gradopol > 1:
        print(np.hstack((A[:,[0]]*x[0],A[:,[1]]*x[1],A[:,[2]]*x[2])))
    #endif
    n2res[gradopol] = np.linalg.norm(A@x - b, 2)
    mmn = A.shape[0] - A.shape[1]
    if mmn>0:  # overdetermined system
      estvar_errpred[gradopol] = n2res[gradopol]**2 / mmn
    #endif
    covd = np.diag(estvar_errpred[gradopol]*(np.linalg.inv(A.T@A)))
    estvar_parest[gradopol,0:gradopol+1] = covd
    plt.figure(10+gradopol); plt.plot(vk,b,'bo'); plt.plot(vk,A@x,'mo'); 
    plt.title('degree = '+str(gradopol)); plt.show()
    B = np.concatenate((A, b), axis=1)
    aug_par = np.vstack((x, np.array([-1])))
    n2_aug_par = np.linalg.norm(aug_par)
    res = B @ aug_par
    n2_res = np.linalg.norm(res)
    #print("zero check: ",n2_res - n2res[gradopol])
    #print "B.shape: ",B.shape
    U,S,V = np.linalg.svd(B); V = V.T  # NB: uncomment gets the usual SVD: U@S@V.T
    res_liminf[gradopol] = S[-1] * n2_aug_par
    UA,SA,VA = np.linalg.svd(A); VA = VA.T  # NB: uncomment for the usual SVD: U@S@V.T
    TLS_aug_par = V[:,-1]/( -V[-1,-1] )
    n2_TLS_aug_par = np.linalg.norm(TLS_aug_par)
    TLS_res = B @ TLS_aug_par
    n2_TLS_res = np.linalg.norm(TLS_res)
    print("estimated parameters: ", x.T)
    print("variance of parameters estimation error: ", covd)
    print("------------")
    print("singular values of A: ",SA)
    print("singular values of [A b]: S = ",S)
    print("aug par norm = ",n2_aug_par)
    print("S[-1] * aug par norm = ",res_liminf[gradopol])
    print("res norm = ",n2_res)
    print("------------")
    print("estimated TLS_aug_par: ", TLS_aug_par)    
    print("TLS aug par norm = ",n2_TLS_aug_par)
    print("S[-1] * TLS aug par norm = ",S[-1] * n2_TLS_aug_par)
    print("TLS res norm = ",n2_TLS_res)
    plt.figure(10+gradopol); plt.plot(vk,b,'bo'); plt.plot(vk,A@np.atleast_2d(TLS_aug_par[0:-1]).T,'go'); plt.title("TLS");
    plt.show()
  #endfor
  plt.figure(40); plt.plot(n2res); plt.title('2-norm of the residual')
  plt.figure(50); plt.plot(estvar_errpred,'b-'); plt.plot(np.array([0., maxgp]),np.array([stderr**2, stderr**2]),'r-'); plt.title('variance of the measurement error (red) and its estimate (blue)');
  plt.figure(60); plt.plot(n2res,'r-'); plt.plot(res_liminf,'b-'); plt.title('2-norm of the residual (red) and its lim-inf (blue)');

           
           