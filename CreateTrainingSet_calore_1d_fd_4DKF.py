import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.io as sio
import pickle
from calore_1d_fd import *

WorkingDirectory = './'

TBegin = 0
TEnd = 1.0
dt = 0.02
N = np.int64(np.floor((TEnd-TBegin)/dt))
tEuler = np.arange(TBegin,TEnd+dt,dt)

n = 50
Dx = 0.1
Lungh = n * Dx
yBegin = np.atleast_2d(g0(n,Dx)).T;

yEuler = yBegin.copy()
ySolver = np.zeros((len(tEuler), len(yBegin)))   # array for solution
ySolver[0, :] = np.squeeze(yEuler)

#Model
nx = n+1
MassMat = np.eye(nx)
LinMat = create_fd_matrix(n,Dx).toarray()

#Observation matrix
C = np.eye(LinMat.shape[1])

#Stochastic matrices
Q = 5e-0*np.eye(np.shape(C)[1])
R = 5e-0*np.eye(np.shape(C)[0])
P0 = 1e-7*np.eye(np.shape(C)[1])

ActivateModelNoise = 1
ActivateMeasNoise = 1
ActivateFirstStateNoise = 1
ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(np.shape(C)[1],N+1))
MeasNoise = np.sqrt(R)@np.random.normal(0,1,(np.shape(C)[0],N+1))
FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(np.shape(C)[1],1))

#Implicit Euler + discrete model noise
M1 = np.eye(n+1) - dt*np.linalg.inv(MassMat)@LinMat;
AInit = np.linalg.inv(M1);
for k in range(N):
    b = np.zeros((n+1,1))
    b[0] = f0(k*dt+TBegin)
    yEuler = np.linalg.solve(M1, yEuler + dt*b)  # T(k+1)=(I-dt*c*A)^{-1}T(k)+(I-dt*c*A)^{-1}*dt*b
    #print("yEuler.shape = ",yEuler.shape)
    yEuler = yEuler + ActivateModelNoise*ModelNoise[:,[k]];
    ySolver[k+1, :] = np.squeeze(yEuler)
#endfor

plt.figure(1)
plt.plot(tEuler, ySolver)
plt.show()

#Set up training set
Meas = C@ySolver.T + ActivateMeasNoise*MeasNoise
plt.figure(2)
plt.plot(tEuler, Meas.T)
plt.title('Noisy measurements')
plt.show()

#Create dataset with solver data
TrainingInstances = 1
TrainingSet = [None] * (6)

for i in range(6):
    TrainingSet[i] = [None] * (TrainingInstances)
#endfor
for i in range(TrainingInstances):
    TrainingSet[0][i] = np.zeros((1,np.shape(ySolver)[0])) #Not used
    TrainingSet[1][i] = Meas #This should depend on i
    TrainingSet[2][i] = ySolver[0:1,:].T + ActivateFirstStateNoise*FirstStateNoise #This should depend on i
    TrainingSet[3][i] = ySolver.T #This should depend on i
    TrainingSet[4][i] = ySolver[N:N+1,:].T #This should depend on i
    TrainingSet[5][i] = 1
#endfor
print(TrainingSet[5][0])
#Save data
Experiment = '0'

sio.savemat(WorkingDirectory+'Experiment.mat', {'Experiment': Experiment})
sio.savemat(WorkingDirectory+'LayersExp'+Experiment+'.mat', {'Layers': N})
sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': C})

with open(WorkingDirectory+'LatestTrainingSetExp'+Experiment+'.mat', 'wb') as handle:
            pickle.dump(TrainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

Model = {}
Model['QInit'] = Q
Model['RInit'] = R
Model['invRInit'] = np.linalg.inv(R)
Model['PInit'] = P0
Model['AInit'] = AInit
Model['M'] = MassMat
Model['K'] = LinMat
Model['D'] = np.tile(np.zeros((nx,1)),(1,N)); #np.tile(KnownOffset,(1,N))
Model['SamplingTimes'] = dt*np.ones((N,1))
with open(WorkingDirectory+'ModelExp'+Experiment+'.mat', 'wb') as handle:
            pickle.dump(Model, handle, protocol=pickle.HIGHEST_PROTOCOL)

HiddenDynDim = 1
sio.savemat(WorkingDirectory+'HiddenDynDimExp'+Experiment+'.mat', {'HiddenDynDim': HiddenDynDim})

APAQInv = [None]*N
RInv = [None]*N
P = Model['PInit']
A = Model['AInit']
Q = Model['QInit']
InvR = Model['invRInit']

for Layer in range(N):
    #print("Layer = ",Layer)
    RInv[Layer] = InvR
    APAQInv[Layer] = np.linalg.inv(A@P@A.T + Q)
    InvP = APAQInv[Layer] + C.T@InvR@C
    P = np.linalg.inv(InvP)    

sio.savemat(WorkingDirectory+'PredictorWeightMatsExp'+Experiment+'.mat', {'PredictorWeightMats': APAQInv})
sio.savemat(WorkingDirectory+'MeasurementWeightMatsExp'+Experiment+'.mat', {'MeasurementWeightMats': RInv})


