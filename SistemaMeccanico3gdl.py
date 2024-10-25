import numpy as np
from  sistemi_DLTI import *

class SistemaMeccanico3gdl:
    # costruttore:
    def __init__(self, M1 = 27.0,  		  # [Kg]
                       M2 = 120.0, 		  # [Kg]
                       M3 = 1.1246e4,  	  # [Kg]
                       K1 = 18.0e7, 	  # [N/m]
                       K2 = 6.0e7, 		  # [N/m]
                       K3 = 1.2e8, 	      # [N/m]
                       C1 = 5.0e4, 		  # [N*s/m]
                       C2 = 4.6e4, 		  # [N*s/m]
                       C3 = 2.4e5, 	      # [N*s/m]
                       deltaX1 = 0.005,   # [m]
                       deltaX2 = 0.005,   # [m]
                       deltaX3 = 0.05):   # [m]
        self.M1 = M1
        self.M2 = M2
        self.M3 = M3
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.deltaX1 = deltaX1
        self.deltaX2 = deltaX2
        self.deltaX3 = deltaX3

    def build_model(self,M1_stimato=None,M2_stimato=None,M3_stimato=None,K1_stimato=None,K2_stimato=None,K3_stimato=None,C1_stimato=None,C2_stimato=None,C3_stimato=None,deltaX1_imposto=None,deltaX2_imposto=None,deltaX3_imposto=None):
        if M1_stimato is not None:
          self.M1 = M1_stimato;
        #endif
        if M2_stimato is not None:
          self.M2 = M2_stimato;
        #endif
        if M3_stimato is not None:
          self.M3 = M3_stimato;
        #endif
        if K1_stimato is not None:
          self.K1 = K1_stimato;
        #endif
        if K2_stimato is not None:
          self.K2 = K2_stimato;
        #endif
        if K3_stimato is not None:
          self.K3 = K3_stimato;
        #endif
        if C1_stimato is not None:
          self.C1 = C1_stimato;
        #endif
        if C2_stimato is not None:
          self.C2 = C2_stimato;
        #endif
        if C3_stimato is not None:
          self.C3 = C3_stimato;
        #endif
        if deltaX1_imposto is not None:
          self.deltaX1 = deltaX1_imposto;
        #endif
        if deltaX2_imposto is not None:
          self.deltaX2 = deltaX2_imposto;
        #endif
        if deltaX3_imposto is not None:
          self.deltaX3 = deltaX3_imposto;
        #endif
        A = np.array([[-self.C1/self.M1,     self.C1/self.M1,         0.,       -self.K1/self.M1,      self.K1/self.M1,          0.       ], \
                      [self.C1/self.M2,  -(self.C1+self.C2)/self.M2,    self.C2/self.M2,       self.K1/self.M2,   -(self.K1+self.K2)/self.M2,     self.K2/self.M2       ], \
                      [   0.,      self.C2/self.M3,     -(self.C2+self.C3)/self.M3,     0.,       self.K2/self.M3,     -(self.K2+self.K3)/self.M3   ], \
                      [   1.,        0.,          0.,          0.,         0.,           0.       ], \
                      [   0.,        1.,          0.,          0.,         0.,           0.       ], \
                      [   0.,        0.,          1.,          0.,         0.,           0.       ]]);
        config_attuatori = 1;
        if config_attuatori == 1:
          Bf = [1/self.M1, 0, 0];
        elif config_attuatori == 2:
          Bf = [0, 1/self.M2, 0];
        elif config_attuatori == 3:
          Bf = [0, 0, 1/self.M3];
        #endif
        # Caso con 1 ingresso esterno:
        B = np.array([[Bf[0],   self.K1*self.deltaX1/self.M1                ], \
                      [Bf[1],   (self.K2*self.deltaX2-self.K1*self.deltaX1)/self.M2   ], \
                      [Bf[2],   (self.K3*self.deltaX3-self.K2*self.deltaX2)/self.M3   ], \
                      [0.,                 0.                ], \
                      [0.,                 0.                ], \
                      [0.,                 0.                ]])
        # scelgo la configurazione dei sensori
        config_sensori = 4;
        if config_sensori == 1:
          C = np.array([[0., 0., 0., 1., 0., 0.],[0., 0., 0., 0., 1., 0.],[0., 0., 0., 0., 0., 1.]]); D = np.array([[0., 0.],[0., 0.],[0., 0.]])
        elif config_sensori == 2:
          C = np.array([[0., 0., 0., 1., 0., 0.][0., 0., 0., 0., 1., 0.]]);  D = np.array([[0., 0.][0., 0.]])
        elif config_sensori == 3:
          C = np.array([[0., 0., 0., 1., 0., -1.][0., 0., 0., 0., 1., -1.]]);  D = np.array([[0., 0.][0., 0.]])
        elif config_sensori == 4:
          C = np.array([[0., 0., 0., 1., 0., 0.]]);  D = np.array([[0., 0.]])
        elif config_sensori == 5:
          C = np.array([[0., 0., 0., 0., 1., 0.]]);  D = np.array([[0., 0.]])
        elif config_sensori == 6:
          C = np.array([[0., 0., 0., 0., 0., 1.]]);  D = np.array([[0., 0.]])
        elif config_sensori == 7:
          C = np.array([[1., 0., 0., 0., 0., 0.]]);  D = np.array([[0., 0.]])
        #endif
        return A, B, C, D


    def simula_sistema(self, A, B, C, D, load, Ts):
        N = load.shape[load.ndim-1]
        # stato iniziale
        x0 = np.array([0., 0., 0., self.deltaX1+self.deltaX2+self.deltaX3, self.deltaX2+self.deltaX3, self.deltaX3])
        # formo il segnale di ingresso 
        u = np.zeros([2,N])
        u[0,:] = load
        u[1,:] = np.ones(N)
        u = np.array(u)
        #
        y,X_hist,Ad = simula_DLTI_StateSpace_continuo(A,B,C,D,u,x0,Ts)
        response = y - C@x0
        return response, X_hist

    def write(self):
        print("M1 = ", self.M1)
        print("M2 = ", self.M2)
        print("M3 = ", self.M3)
        print("K1 = ", self.K1)
        print("K2 = ", self.K2)
        print("K3 = ", self.K3)
        print("C1 = ", self.C1)
        print("C2 = ", self.C2)
        print("C3 = ", self.C3)
        print('coordinate di partenza: y1=%.2f, y2=%.2f, y3=%.2f' % (self.deltaX3+self.deltaX2+self.deltaX1, self.deltaX3+self.deltaX2, self.deltaX3))
    
