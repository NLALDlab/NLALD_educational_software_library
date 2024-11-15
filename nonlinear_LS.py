import matplotlib.pyplot as plt
from math import *
import numpy as np


calc_FN = lambda r: (np.linalg.norm(r,2) / np.sqrt(len(r)))**2

def plot_FN(y_meas,y,theta_lims,N1,N2,E=None,theta=None):
    dp1 = (theta_lims[1]-theta_lims[0])/(N1-1)
    dp2 = (theta_lims[3]-theta_lims[2])/(N2-1)
    V = np.zeros((N1,N2))
    for i1 in range(N1):
        for i2 in range(N2):
            p1 = theta_lims[0] + i1*dp1
            p2 = theta_lims[2] + i2*dp2
            V[i1,i2] = calc_FN(y(np.array([p1,p2])) - y_meas)
        #endfor
    #endfor
    nlevs = max([50,int(ceil(N1/2))])
    if dp2>0.0:    
        x2 = np.arange(theta_lims[2],theta_lims[3]+dp2/2,dp2)
        print("x2.shape = ",x2.shape)
    #endif
    if dp1>0.0:    
        x1 = np.arange(theta_lims[0],theta_lims[1]+dp1/2,dp1)
        print("x1.shape = ",x1.shape)
    #endif
    print("V.shape = ",V.shape)
    if dp1>0.0 and dp2>0.0:
        plt.figure(100); plt.axis('equal'); plt.contourf(x2,x1,np.array(V),nlevs,cmap='rainbow'); 
        if E is not None:
            plt.plot(E[:,1],E[:,0],'y-o')
        #endif
        if theta is not None:
            plt.plot(theta[1],theta[0],'rd')
        #endif
    elif dp1==0.0:
        plt.figure(101); plt.plot(x2,V[0,:].T,'b-'); plt.axis([theta_lims[2],theta_lims[3],0.0,np.max(V[0,:])])       
        if E is not None:
            jp=0; tmpv = calc_FN(y(np.array([E[jp,0],E[jp,1]])) - y_meas)
            plt.plot(E[jp,1],tmpv,'bo')
            jp=1; tmpv = calc_FN(y(np.array([E[jp,0],E[jp,1]])) - y_meas)
            plt.plot(E[jp,1],tmpv,'co')
            jp=2; tmpv = calc_FN(y(np.array([E[jp,0],E[jp,1]])) - y_meas)
            plt.plot(E[jp,1],tmpv,'go')
            for jp in range(3,E.shape[0]-1):
                tmpv = calc_FN(y(np.array([E[jp,0],E[jp,1]])) - y_meas)
                plt.plot(E[jp,1],tmpv,'yo')
            #endfor
            jp=E.shape[0]-1; tmpv = calc_FN(y(np.array([E[jp,0],E[jp,1]])) - y_meas)
            plt.plot(E[jp,1],tmpv,'ro')
        #endif
        if theta is not None:
            tmpv = calc_FN(y(np.array([theta[0],theta[1]])) - y_meas)
            plt.plot(theta[1],tmpv,'rd')
        #endif
        plt.ylim((-0.5,3.0)); plt.xlim((-3.0,3.0))
    elif dp2==0.0:
        plt.figure(102); plt.plot(x1,V[:,0])        
    #endif
    plt.show()
    

def GN_LM(y_meas,y,theta_est,h_theta=None,stop_iter_thrs_on_applied_coeff_update=1.e-16,overfit_thrs=1.e-5,metodo_stima='GN',analisi=False,LM_semplificato=False,\
          opzione_solve='LS',max_iter=20,est_stderrmis=0.0,TRradius_max=2.5,TRradius_init_divisor=2.0,TRradius_approx_thrs=1.e-3,TRshrink_thrs=0.25,TRexpand_thrs=0.75,update_accept_thrs=0.002,SD_step=0.1,momentum_coef=0.0,lam_Tich=1.0,silence=False):
    # y_meas: np.array 1-dim di N valori misurati che il modello stimato deve approssimare
    # y(theta_est): modello parametrico (funzione che restituisce un array 1-dim di N componenti)
    # theta_est: np.array 1-dim di n_theta valori contenente le stime iniziali dei parametri
    # h_theta: np.array 1-dim di n_theta valori contenente le perturbazioni da applicare ai parametri per il calcolo di psi
    # stop_iter_thrs_on_applied_coeff_update: soglia del criterio di arresto basato sul controllo dell'incremento
    # overfit_thrs: soglia sulla funzione costo, al di sotto della quale si termina il procedimento di stima.
    # metodo_stima: 'GN'=Gauss-Newton (line-search), 'LM'=Levenberg-Marquardt (trust-region)
    # analisi = True, False
    # LM_semplificato: True, False
    # opzione_solve: 'LS', 'TLS', 'TSVD1'
    # max_iter: numero massimo di iterazioni di Newton
    # est_stderrmis: stima della deviazione standard dell'errore di misura.
    ### parametri per il metodo Trust-Region-Newton (Levenberg-Marquardt):
    # TRradius_max: 
    # TRradius_init_divisor: 
    # TRradius_approx_thrs: 
    # TRshrink_thrs: 
    # TRexpand_thrs: 
    # update_accept_thrs: 
    #
    n_theta = len(theta_est)
    N = len(y_meas)
    if 1: #h_theta.any()==None:
        h_theta = 0.2 * np.ones(n_theta)
    #endif
    TRradius = TRradius_max / TRradius_init_divisor
    #print("TODO: STAMPO I PARAMETRI DI CHIAMATA DELLA FUNZIONE ...")
    print("*******************************************************")
    print("metodo_stima = ",metodo_stima,)
    if metodo_stima=='LM' and LM_semplificato:
        print(" semplificato")
    else:
        print("\n")
    #endif
    print("opzione_solve = ",opzione_solve)
    print("-------------------------------------------------------")
    applied_coeff_update = float(np.inf) # incrementi applicati ai parametri dall'iterazione corrente (inizialmente "inf" per rendere sicuramente vera la condizione di esecuzione del ciclo while)
    iteraz = 0 # contatore dell'iterazione
    E = np.zeros((max_iter+1,n_theta))
    FN_hist = np.zeros(max_iter+1)
    E[0,:] = theta_est
    print("start:  theta_est = ", theta_est, "FN = ", calc_FN(y(theta_est) - y_meas))
    plt.figure(iteraz); plt.plot(y_meas,'b-'); plt.plot(y(theta_est),'r-'); plt.title('iteraz = '+str(iteraz))
    psi_r = np.zeros((N,n_theta))
    previous_direction = np.zeros(n_theta)
    FN_iter = float(np.inf)  # funzione costo (inizialmente ad infinito)
    #
    while np.max(np.abs(applied_coeff_update)) > stop_iter_thrs_on_applied_coeff_update and iteraz < max_iter and FN_iter > overfit_thrs:  
        iteraz += 1
        r = y(theta_est) - y_meas
        #print("r.shape = ",r.shape)
        FN = calc_FN(r)
        #print("FN = ", FN)
        ### Calcolo la matrice di sensitivita' PSI
        for itheta in range(0,n_theta):
            #print("itheta = ", itheta)
            store_coeff = theta_est[itheta]
            theta_est[itheta] -= h_theta[itheta]/2.0
            y1 = y(theta_est)
            theta_est[itheta] = store_coeff
            #
            store_coeff = theta_est[itheta]
            theta_est[itheta] += h_theta[itheta]/2.0
            y2 = y(theta_est)
            theta_est[itheta] = store_coeff
            #
            psi_r[:,itheta] = (y2 - y1) / h_theta[itheta]
        #endfor
        if analisi:
            if 0: print("psi_r = ", psi_r)
            print("psi_r.shape = ", psi_r.shape)
            S = np.linalg.svd(psi_r,compute_uv=False)
            print("valori singolari di psi_r = ", S)
        #endif
        ### Calcolo l'aggiornamento con il metodo (damped) Gauss-Newton:
        if metodo_stima=='GN' or metodo_stima=='LM':
            if opzione_solve == 'LS':
                if False:
                    pass
                else:
                    [Q,R] = np.linalg.qr( psi_r )
                    #print("Q = ",Q)
                    #print("R = ",R)
                    #print("r.shape = ",r.shape)
                    delta_theta_GN = np.squeeze( np.linalg.solve(R, -(Q.T @ np.atleast_2d(r).T)) )
                #endif
            elif opzione_solve == 'TLS':
                M = np.hstack((psi_r, -np.atleast_2d(r).T))
                U, S, V = np.linalg.svd(M,full_matrices=0, compute_uv=1); V = V.T
                delta_theta_GN = - V[:,2] / V[2,2]
                delta_theta_GN = delta_theta_GN[0:2]
            elif opzione_solve == 'TSVD1':
                U, S, V = np.linalg.svd(psi_r,full_matrices=0, compute_uv=1); V = V.T
                delta_theta_GN = np.dot( np.dot( V[:,0:1], np.dot( np.linalg.inv(np.diag(S)[0:1,0:1]), U[:,0:1].T ) ), -r )
                delta_theta_GN = np.squeeze(delta_theta_GN)
            elif opzione_solve == 'Tich':
                tmpM = np.vstack((psi_r,np.sqrt(lam_Tich)*np.eye(psi_r.shape[1])))
                _,R = np.linalg.qr(tmpM)
                tmpx = np.squeeze( np.linalg.solve(R.T, - psi_r.T @ np.atleast_2d(r).T) )
                delta_theta_GN = np.squeeze( np.linalg.solve(R, tmpx) )
            #endif
            if metodo_stima=='GN':
                temp_FN = FN
                mu = 1.0
                decrate_FN = 0.0 #1.e-3  # tasso di riduzione di FN che rende accettabile l'aggioramento delle stime
                                    # NB: provare con "decrate_FN = 1.e-2": talvolta non converge !
                while temp_FN >= FN*(1-decrate_FN) and mu > 0.0:
                    delta_theta_temp = mu * delta_theta_GN
                    y_temp = y(theta_est + delta_theta_temp)
                    temp_FN = calc_FN(y_temp - y_meas)
                    if not silence: print("mu = ", mu, "temp_FN = ", temp_FN)
                    if mu > 1e-8:
                        mu /= 2.0
                    else:
                        mu = 0.0
                    #endif
                #endwhile
                if mu > 0.0:
                    delta_theta_GN = delta_theta_temp
                else:
                    delta_theta_GN = np.zeros(n_theta)
                #endif
            #endif
            if analisi:
                print("delta_theta_GN = ", delta_theta_GN)
                if est_stderrmis > 0.0:
                    print("covarianza err stima dei parametri = ", est_stderrmis * (psi_r.T @ psi_r))
                #endif
            #endif
        #endif
        if metodo_stima=='LM':
            ### Calcolo l'aggiornamento con il metodo Trust-Region-Newton (Levenberg-Marquardt):
            r = np.atleast_2d(r).T
            applicable_Parametri_delta_TRN = np.zeros(n_theta)
            delta_theta_TRN = delta_theta_GN.copy()
            if not silence: print("np.linalg.norm(delta_theta_GN) - TRradius = ", np.linalg.norm(delta_theta_GN)-TRradius)
            lam = 0.01  # o 1.0e3 scelta iniziale di lambda ("lam")
            tmpit2 = 0 
            while np.linalg.norm(applicable_Parametri_delta_TRN) == 0.0:
                tmpit2 += 1
                # valuto se la soluzione G-N sta nella trust region e calcolo la soluzione TRN:
                if (np.linalg.norm(delta_theta_TRN) > TRradius) or (np.linalg.norm(delta_theta_GN)==0.0):
                    if not silence: print("aggiornamento TRN !")
                    #lam = 1.0e3  # scelta iniziale di lambda ("lam")
                    trust_ok = False
                    MLM = np.zeros([N+psi_r.shape[1], psi_r.shape[1]])  # matrice del sistema sovradeterminato di Levenberg-Marquardt
                    MLM[0:N,:] = psi_r
                    tmpit = 0
                    while trust_ok == False:  # ricerca del valore di lambda in modo che "np.linalg.norm(delta_theta_TRN) == TRradius"
                        tmpit += 1
                        MLM[N:,:] = np.sqrt(np.array([lam])) * np.eye(psi_r.shape[1])
                        # soluzione del sistema con la fattorizzazione QR:
                        Qlambda, Rlambda = np.linalg.qr(MLM)  # NB: questa fattorizzazione si presta all'aggiornamento ricorsivo.
                        if not silence: print("cond(MLM) = ",np.linalg.cond(MLM.T@MLM,2),"  ,  cond(psi_r) = ",np.linalg.cond(psi_r.T@psi_r,2))
                        g = 2./N * psi_r.T @ r
                        ytmp = np.linalg.solve(Rlambda.T, -g)
                        delta_theta_TRN = np.squeeze( np.linalg.solve(Rlambda, ytmp) )
                        if not silence: print("lambda = ", lam, "   np.linalg.norm(delta_theta_TRN) - TRradius = ", np.linalg.norm(delta_theta_TRN)-TRradius,"   TRradius_approx_thrs = ",TRradius_approx_thrs)
                        if np.linalg.norm(delta_theta_TRN) - TRradius < TRradius_approx_thrs:
                            trust_ok = True
                        else:
                            if LM_semplificato:
                                if np.linalg.norm(delta_theta_TRN) < TRradius:
                                    lam = lam * 0.5
                                elif np.linalg.norm(delta_theta_TRN) > TRradius:
                                    lam = lam * 2.5
                                #endif
                            else: # root finding:
                                q_l = np.linalg.solve(Rlambda.T, delta_theta_TRN)
                                lam = lam + (np.linalg.norm(delta_theta_TRN) / np.linalg.norm(q_l))**2 * ((np.linalg.norm(delta_theta_TRN)-TRradius)/TRradius)
                                if lam < 0.0:
                                    lam = 0.0
                                #endif
                            #endif
                        #endif
                        if tmpit >= 10:
                            trust_ok = True  # gli algoritmi pratici non chiedono il lambda ottimale, ma si accontentano d 2 o 3 iterazioni.
                        #endif
                    #endwhile
                    if not silence: print("delta_theta_TRN = ",delta_theta_TRN,"    theta_est  = ",theta_est) 
                    theta_upd_temp = theta_est + delta_theta_TRN
                    y_temp = y(theta_upd_temp)
                    temp_FN_TRN = calc_FN(y_temp - y_meas)
                    # calcolo "rho" [Nocedal-Wright, p.67 e p.262]:
                    vtmp = r + 0.5 * psi_r @ np.atleast_2d(delta_theta_TRN).T  
                    if not silence: print("delta_theta_TRN = ",delta_theta_TRN)
                    predicted_reduction = - delta_theta_TRN @ psi_r.T @ vtmp 
                    if not silence: print("predicted_reduction = ",predicted_reduction)
                    if not silence: print("FN = ",FN," , temp_FN_TRN = ",temp_FN_TRN)
                    rho_iter = (FN - temp_FN_TRN) / predicted_reduction
                    if not silence: print("rho_iter = ",rho_iter)
                    y_temp = y(theta_est + delta_theta_GN)
                    temp_FN = calc_FN(y_temp - y_meas)
                    if not silence: print("temp_FN = ",temp_FN,"   temp_FN_TRN = ",temp_FN_TRN)
                    if temp_FN < temp_FN_TRN: # puo' accadere solo se la soluzione GN e' fuori dalla trust-region
                        TRradius *= (1 + (temp_FN_TRN - temp_FN)/temp_FN_TRN)
                    else:
                        # scelgo il raggio della trust-region, "TRradius", come in [Nocedal-Wright, Algorithm 4.1, p.68]:
                        norm2_delta_theta_TRN = np.linalg.norm(delta_theta_TRN)
                        if rho_iter < TRshrink_thrs:
                            TRradius *= 0.5  # = norm2_delta_theta_TRN / 4.
                        else:
                            if (rho_iter > TRexpand_thrs) and (abs(norm2_delta_theta_TRN - TRradius) < TRradius_approx_thrs):
                                TRradius = min([2*TRradius, TRradius_max])
                            else:
                                # TRradius rimane invariato
                                pass
                            #endif
                        #endif
                        if rho_iter > update_accept_thrs:
                            applicable_Parametri_delta_TRN = delta_theta_TRN.copy()
                        else:
                            delta_theta_TRN = np.zeros((n_theta,1))
                            tempFN_TRN = FN
                        #endif
                    #endif
                    if not silence: print("rho_iter = ", rho_iter, "   TRradius = ", TRradius)
                else:  # la soluzione G-N sta dentro alla trust region
                    if not silence: print("aggiornamento GN !")
                    applicable_Parametri_delta_TRN = np.atleast_2d(delta_theta_GN.copy()).T
                    temp_FN_TRN = FN
                #endif
                if tmpit2 >= 50:
                    print("non converge !!!")
                    break  # TODO cosa fare se non converge ?
                #endif
            #endwhile
        #endif
        #
        if metodo_stima == 'GN':
            applied_coeff_update = delta_theta_GN
        elif metodo_stima == 'LM':
            applied_coeff_update = np.array(applicable_Parametri_delta_TRN).T[0]
            if not silence: print("LM: applied_coeff_update = ",applied_coeff_update)
        elif metodo_stima == 'SD':
            [Q,R] = np.linalg.qr( psi_r )
            #print("Q = ",Q)
            #print("R = ",R)
            actual_SD_direction = np.squeeze( (Q.T @ np.atleast_2d(r).T) )
            direction = actual_SD_direction + momentum_coef * previous_direction
            applied_coeff_update = SD_step * direction
            previous_direction = direction.copy()
            if not silence: print("delta_theta_SD = ", applied_coeff_update)
        else:
            print("errore: metodo di stima inesistente!")
        #endif
        theta_est += applied_coeff_update
        E[iteraz,:] = theta_est
        r = y(theta_est) - y_meas
        FN = calc_FN(r)
        FN_hist[iteraz] = FN
        if (not silence) or iteraz%100==0: print("fine di iterazione ", iteraz, ":  theta_est = ", theta_est, "   FN = ", FN,\
                                                "\n-----------------------------------------------------------")
        if not silence: plt.figure(iteraz); plt.plot(y_meas,'b-'); plt.plot(y(theta_est),'r-'); plt.title('iteraz = '+str(iteraz)); #axis([0., 1., -0.5, 0.5])
    #endwhile
    return E[0:iteraz+1,:],FN_hist
