import numpy as np

def simula_DLTI(b,a,u):
    na = np.size(a);
    nb = np.size(b);
    #print "nb = ", nb
    if na==1:  a=np.array([a, 0.]); na += 1; #endif  # serve per non inserire "if" nel ciclo "for"
    if nb==1:  b=np.array([b, 0]); nb += 1; #endif  # serve per non inserire "if" nel ciclo "for"
    u_past = np.zeros(nb-1);
    y_past = np.zeros(na-1);
    N = np.size(u);
    y = np.zeros(N);
    #print "b = ", b
    for n in range(0,N):
        vb = np.dot(b,np.concatenate((np.array([u[n]]), np.array(u_past))))
        #print "vb = ", vb
        y[n] = 1./a[0] * np.sum(np.array([ np.dot(-a[1:na],y_past),  vb ]));
        u_past[1:nb-1] = u_past[0:nb-2];  u_past[0] = u[n];
        y_past[1:na-1] = y_past[0:na-2];  y_past[0] = y[n];
    #endfor
    return y


def simula_DLTI_StateSpace_continuo(A,B,C,D,u,x0,Ts):
    N = u.shape[u.ndim-1]
    y = np.zeros([C.shape[0],N])
    x = np.atleast_2d(x0.copy()).T
    X_hist = np.zeros([x.shape[0],N+1])
    X_hist[:,0] = np.squeeze( x0.copy() )
    Ad = np.eye(A.shape[0]) - Ts*A
    for i in range(0,N):
        x = np.linalg.solve(Ad, (x + Ts*B@np.atleast_2d(u[:,i]).T) )
        X_hist[:,i+1] = np.squeeze(x)
        y[:,i] = C@x  +  D@u[:,i];
    #endfor
    return y, X_hist, Ad


def simula_DLTI_StateSpace_discreto(A,B,C,D,u,x0):
    N = u.shape[u.ndim-1]
    y = np.zeros([C.shape[0],N])
    x = x0.copy()
    X_hist = np.zeros([x.shape[0],N+1])
    X_hist[:,0] = np.squeeze( x0.copy() )
    for i in range(0,N):
        y[:,i] = C@x  +  D@u[:,i];
        x = A@x + B@np.atleast_2d(u[:,i])
        X_hist[:,i+1] = np.squeeze(x)
    #endfor
    return y, X_hist





