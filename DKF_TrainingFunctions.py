import numpy as np
import scipy.io as sio
from scipy.optimize import fsolve
import pickle
import copy
from DKF_TestBartlett import *


def InitializeWeights(NetParameters):
    """
    Initializes the net's weights with Gaussian noise of mean NetParameters.InitializationMean 
    and sigma NetParameters.InitializationSigma.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    Initialization = NetParameters['Initialization']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']

    C = NetParameters['C']
    Model = NetParameters['Model']
    
    NetWeights = [None] * (Layers + 2)

    if Initialization == 'Deterministic':
        # Deterministic initialization
        P = Model['PInit']
        A = Model['AInit']
        Q = 0. #Model['QInit']
        InvR = Model['invRInit']
        InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
        KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)
        for Layer in range(Layers):
            NetWeights[Layer] = np.copy(KFGain)
        #endfor
        NetWeights[Layers] = C
    elif Initialization == 'DeterministicComplete':
        # DeterministcComplete initialization
        P = Model['PInit']
        A = Model['AInit']
        Q = Model['QInit']
        InvR = Model['invRInit']
        for Layer in range(Layers):
            InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
            P = np.linalg.inv(InvP)
            KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)
            NetWeights[Layer] = np.copy(KFGain)
        #endfor
        NetWeights[Layers] = C
    elif Initialization == 'Random':
        #Random initialization
        Mean = NetParameters['InitializationMean']
        Sigma = NetParameters['InitializationSigma']
        for Layer in range(Layers):
            NetWeights[Layer] = np.random.normal(Mean, Sigma, (ObservationDimension,ObservationDimension))
        #endfor
        NetWeights[Layers] = C
    #endif
    return NetWeights


def ConstructSGMatrices(WinLen):
    """
    Construct matrices used during SG filtering.
    """
    HalfWinLen = (WinLen - 1) // 2
    Int = np.arange(-HalfWinLen, HalfWinLen + 1)

    # Degree = 3; Fixed for now
    StencilA0 = np.flip((3 / (4 * WinLen * (WinLen**2 - 4))) * (3 * WinLen**2 - 7 - 20 * Int**2))
    StencilA1 = np.flip((1 / (WinLen * (WinLen**2 - 1) * (3 * WinLen**4 - 39 * WinLen**2 + 108))) * 
                        (75 * (3 * WinLen**4 - 18 * WinLen**2 + 31) * Int - 
                         420 * (3 * WinLen**2 - 7) * Int**3))
    return StencilA0, StencilA1


def InitializeGradsAndMoments(NetWeights, NetParameters):
    """
    Initializes the gradients for the net's parameters to zero.
    """
    Layers = NetParameters['Layers']
    # Setup gradients
    Grads = [None] * (Layers + 2)
    for Layer in range(Layers):
        Grads[Layer] = np.zeros_like(NetWeights[Layer]).astype('float64')
    #endfor
    Grads[Layers] = np.zeros_like(NetWeights[Layers]).astype('float64')
    #
    Moment1 = copy.deepcopy(Grads)
    Moment2 = copy.deepcopy(Grads)
    return Grads, Moment1, Moment2


def ComputeWeightMats(NetParameters):
    """
    Computes the weight matrices used for residue scaling.

    Parameters:
        NetParameters (dict): Dictionary containing network parameters.

    Returns:
        MeasurementWeightMats (list of numpy.ndarray): Measurement weight matrices for each layer.
        PredictorWeightMats (list of numpy.ndarray): Predictor weight matrices for each layer.
        MeasurementWeightMatsSym (list of numpy.ndarray, optional): Symmetric measurement weight matrices.
        PredictorWeightMatsSym (list of numpy.ndarray, optional): Symmetric predictor weight matrices.
    """
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']
    WeightMats = NetParameters['WeightMats']

    MeasurementWeightMats = [None] * Layers
    PredictorWeightMats = [None] * Layers

    # Cleaner propagation for test phase
    MeasurementWeightMatsSym = [None] * Layers
    PredictorWeightMatsSym = [None] * Layers

    # Compute weight matrices
    if WeightMats == 'Identity':
        for Layer in range(Layers):
            MeasurementWeightMats[Layer] = np.eye(ObservationDimension)
            PredictorWeightMats[Layer] = np.eye(StateDimension)

            MeasurementWeightMatsSym[Layer] = np.eye(ObservationDimension)
            PredictorWeightMatsSym[Layer] = np.eye(StateDimension)
        #endfor
    elif WeightMats == 'Input':
        Experiment = NetParameters['Experiment']
        MeasurementWeightMats = sio.loadmat(f'MeasurementWeightMatsExp{Experiment}.mat',squeeze_me = True)['MeasurementWeightMats']
        PredictorWeightMats = sio.loadmat(f'PredictorWeightMatsExp{Experiment}.mat',squeeze_me = True)['PredictorWeightMats']
        for Layer in range(Layers):
            MeasurementWeightMatsSym[Layer] = 0.5 * (MeasurementWeightMats[Layer] + MeasurementWeightMats[Layer].T)
            PredictorWeightMatsSym[Layer] = 0.5 * (PredictorWeightMats[Layer] + PredictorWeightMats[Layer].T)
    #endif
    return MeasurementWeightMats, PredictorWeightMats, MeasurementWeightMatsSym, PredictorWeightMatsSym


def PropagateInput(Inputs, Measurements, FirstState, Dynamic, F, NetWeights, NetParameters, decode):
    """
    Propagates the Inputs vector (u) and Measurements vector (y) through the network. 
    They are lists of size (1, NetParameters['Layers']).
    F is the 'VARMION' function block. The output is the States vector (x),
    a list of size (1, NetParameters['Layers'] + 1). States[0] is given as an input. 
    Additional outputs MeasurementMinusCFs, GainMeasurementMinusCFs, MeasurementMinusCStates, 
    and FStateDynInputs are saved for later efficiency during backpropagation.
    """

    Layers = NetParameters['Layers']
    C = NetParameters['C']
    nx = C.shape[1]
    # Setup output
    States = np.tile(None,(nx,Layers + 1))
    MeasurementMinusCStates = np.tile(None,(nx,Layers))
    GainMeasurementMinusCFs = np.tile(None,(nx,Layers))

    MeasurementMinusCFs = np.tile(None,(nx,Layers))
    FStateDynInputs = np.tile(None,(nx,Layers))

    # Initialize the first state
    States[:,[0]] = FirstState

    # Propagate through layers
    for Layer in range(Layers):
        Indx = Layer
        # Compute FStateDynInput using the provided function F
        FStateDynInput = F(States[:,Layer], None, Inputs[:,Layer:Layer+1], None, Layer, NetParameters)
        # Calculate MeasurementMinusCF and GainMeasurementMinusCF
        #MeasurementMinusCF = Measurements[:,Layer+1:Layer+2] - C @ FStateDynInput
        MeasurementMinusCF = Measurements[:,Layer+1:Layer+2] - NetWeights[-2] @ FStateDynInput
        GainMeasurementMinusCF = NetWeights[Indx]@MeasurementMinusCF
        # Save outputs
        States[:,[Layer+1]] = FStateDynInput + GainMeasurementMinusCF
        #MeasurementMinusCStates[:,Layer] = Measurements[:,Layer+1:Layer+2] - C@States[Layer+1]
        MeasurementMinusCStates[:,[Layer]] = Measurements[:,Layer+1:Layer+2] - NetWeights[-2]@States[:,[Layer+1]]
        GainMeasurementMinusCFs[:,[Layer]] = GainMeasurementMinusCF
        MeasurementMinusCFs[:,[Layer]] = MeasurementMinusCF
        FStateDynInputs[:,[Layer]] = FStateDynInput
    #endfor
    
    # parsimony:
    tmpI = np.where(np.diag(NetWeights[-2]) <= 0.0)[0]
    x_mask = np.ones(nx); x_mask[tmpI] = 0
    for Layer in range(Layers+1):    
        States[:,[Layer]] = decode(States[:,[Layer]],x_mask)
    #endfor

    return States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, FStateDynInputs


def DynamicEquation(z, x, p, u, s, M, K, d, Ts, NetParameters):
    """
    Encodes the (implicit) equation for the dynamics to be solved forward in time.
    """
    # Variables
    z = np.atleast_2d(z).T
    x = np.atleast_2d(x).T
    #print("x.shape = ",x.shape)
    F = M @ (z - x) - Ts * (K @ z + d)
    #print("type(F) = ",type(F))
    #print("F.shape = ",F.shape)
    #print("type(F[0,0]) = ",type(F[0,0]))
    F = F[:,0]
    return F.astype(float)


def F(x, p, u, s, Layer, NetParameters):
    """
    Computes the state prediction xp.
    """
    # Variables
    Model = NetParameters['Model']
    d = Model['D'][:,[Layer]]
    Ts = Model['SamplingTimes'][Layer]
    x = np.atleast_2d(x)
    if x.shape[0]<x.shape[1]: x = x.T
    
    # Equation for the dynamics
    def DynEq(z):
        return DynamicEquation(z, x, p, u, s, Model['M'], Model['K'], d, Ts, NetParameters)

    # Solve for the state prediction
    #print("type(x[0]) = ",type(x[0]))
    #print("x.shape = ",x.shape)
    #print("d.shape = ",d.shape)
    if 0: # nolinear model
        options = {'xtol': 1.e-6, 'maxfev': 1000000}
        xp = np.atleast_2d( fsolve(DynEq, x.astype(float), xtol=options['xtol'], maxfev=options['maxfev']) ).T
    elif 1: # linear model with M=IdentityMatrix
        xp = Model['AInit']@x + Ts*Model['AInit']@d
    else:
        xp = Model['AInit']@x + Ts*Model['AInit']@np.linalg.inv(Model['M'])@d
    #endif
    return xp


def ConstructTensorizedGains(NetWeights, NetParameters):
    """
    Inserts the Kalman gains into a 3-D tensor.
    """
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']

    TensorizedGains = np.zeros((StateDimension, ObservationDimension, Layers))
    # Assemble tensor
    for Layer in range(Layers):
        TensorizedGains[:,:,Layer] = NetWeights[Layer]
    #endfor
    return TensorizedGains


def ConstructLaplacianMatrices(N, SamplingTimes):
    """
    Constructs discrete Laplacian matrices for loss function term.
    """
    # Variables
    TimeStep = SamplingTimes[0]  # Assuming constant sampling
    InvTSSq = (1 / TimeStep)**2
    # Assemble the matrices
    L = np.vstack([
        np.concatenate( (np.array([2, -5, 4, -1]), np.zeros(N-4)) ), 
        np.hstack([
            np.atleast_2d(np.concatenate(([1], np.zeros(N-3)))).T, 
            np.diag(-2 * np.ones(N-2)) + np.diag(np.ones(N-3), 1) + np.diag(np.ones(N-3), -1),
            np.atleast_2d(np.concatenate( (np.zeros(N-3), [1]) )).T
        ]),
        np.concatenate( (np.zeros(N-4), np.array([-1, 4, -5, 2])) )
    ])
    # Uncomment if you want to scale L by InvTSSq
    # L = InvTSSq * L
    LtL = L.T @ L
    return L, LtL


def ComputePeriodogramResidue(MeasurementMinusCStates, MeasurementMinusCFs):
    """
    Computes the periodogram residues for both correctors and predictors at every layer.
    """
    Layers = MeasurementMinusCStates.shape[1]
    ObservationDimension = MeasurementMinusCStates.shape[0]

    CorrectorResidues = np.zeros((ObservationDimension, Layers))
    PredictorResidues = np.zeros((ObservationDimension, Layers))

    for Layer in range(Layers):
        CorrectorResidues[:,Layer:Layer+1] = MeasurementMinusCStates[:,[Layer]]
        PredictorResidues[:,Layer:Layer+1] = MeasurementMinusCFs[:,[Layer]]
    #endfor
    CorrectorPeriodogramResidues = np.zeros(ObservationDimension)
    PredictorPeriodogramResidues = np.zeros(ObservationDimension)

    for ObservedState in range(ObservationDimension):
        PredictorPeriodogram = TestBartlett(PredictorResidues[ObservedState,:])[0]
        PredictorPeriodogramResidues[ObservedState] = np.linalg.norm(PredictorPeriodogram - np.linspace(0, 1, len(PredictorPeriodogram)))
        #
        CorrectorPeriodogram = TestBartlett(CorrectorResidues[ObservedState,:])[0]
        CorrectorPeriodogramResidues[ObservedState] = np.linalg.norm(CorrectorPeriodogram - np.linspace(0, 1, len(CorrectorPeriodogram)))
    #endfor
    PeriodogramResidues = np.concatenate([CorrectorPeriodogramResidues, PredictorPeriodogramResidues])
    return PeriodogramResidues


def StateJacobian(F, x, p, u, s, Fxpu, Layer, N, NetParameters):
    """
    Computes the Jacobian matrix for F with respect to the x variables at the point (x, p, u).
    Fxpu = F(x, p, u) is given as an input for efficiency since it was already computed.
    
    Parameters:
        F (function): Function to compute the Jacobian of.
        x (numpy.ndarray): State vector.
        p (numpy.ndarray): Parameter vector.
        u (numpy.ndarray): Input vector.
        s (numpy.ndarray): Sparse matrix (or other needed matrices).
        Fxpu (numpy.ndarray): Function value at (x, p, u).
        Layer (int): Current layer index.
        N (int): Dimension of the state vector.
        NetParameters (dict): Dictionary containing network parameters.
    
    Returns:
        StateJac (numpy.ndarray): Jacobian matrix of F with respect to x.
    """
    FiniteDifferences = NetParameters['FiniteDifferences']
    h = NetParameters['FiniteDifferencesSkip']
    StateJac = np.zeros((N, N))
    # Cycle over columns of Jacobian
    for ColInd in range(N):
        # Increment in ColInd-th cardinal direction
        Increment = np.zeros((N,1))
        Increment[ColInd] = h
        if FiniteDifferences == 'Forward':
            StateJac[:,ColInd:ColInd+1] = (F(x + Increment, p, u, s, Layer, NetParameters) - Fxpu) / h
        elif FiniteDifferences == 'Backward':
            StateJac[:,ColInd:ColInd+1] = (Fxpu - F(x - Increment, p, u, s, Layer, NetParameters)) / h
        elif FiniteDifferences == 'Central':
            StateJac[:,ColInd:ColInd+1] = (F(x + Increment, p, u, s, Layer, NetParameters) - F(x - Increment, p, u, s, Layer, NetParameters)) / (2 * h)
        #endif
    #endfor
    return StateJac


def ComputeJacobians(F, States, Dyn, Inputs, SparseMat, Dynamic, FStateDynInputs, NetParameters):
    """
    Computes the Jacobians of F at the different layers of the net. StateJacobians & DynJacobians are lists of size (1, NetParameters['Layers']) where
    StateJacobians[0] = [] since it is not used during backpropagation.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    Jacobians = NetParameters['Jacobians']
    N = NetParameters['StateDimension']

    # Setup output
    StateJacobians = [None] * Layers
    DynJacobians = [None] * Layers

    if Jacobians == 'Approximated':
        # Approximate Jacobians with finite differences
        for Layer in range(1, Layers):
            #print("Layer = ",Layer)
            StateJacobians[Layer] = StateJacobian(F, States[:,[Layer]], Dyn, Inputs[:,Layer:Layer+1], SparseMat, FStateDynInputs[:,[Layer]], Layer, N, NetParameters)            
        #endfor
    elif Jacobians == 'Algebraic':
        # Set Jacobians to their exact algebraic representation, when possible
        for Layer in range(1, Layers):
            # Uncomment and define StateJacobianAlgebraic function when available
            # StateJacobians[Layer] = StateJacobianAlgebraic(F, States[Layer], Dyn, Inputs[Layer], SparseMat, FStateDynInputs[Layer], Layer, N, NetParameters)
                pass
        #endfor
    #endif
    return StateJacobians, DynJacobians


def BackPropagateOutput(StateTrue, Dynamic, States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, FStateDynInputs, TensorizedGains, MeasurementWeightMatsSym, PredictorWeightMatsSym, Grads, StateJacobians, DynJacobians, NetWeights, NetParameters):
    """
    Computes the gradients of the loss function with respect to the parameters.
    The loss function is:
    
    (Penalty0/2)*||States[Layers+1] - StateTrue||^2 + 
    sum_{Layer=1,...,Layers}(Penalty1/2)*( MeasurementMinusCStates[Layer].T )*MeasurementWeightMats[Layer]*( MeasurementMinusCStates[CurrentLayer] ) + 
    sum_{Layer=1,...,Layers}(Penalty2/2)*( GainMeasurementMinusCFs[Layer].T )*PredictorWeightMats[Layer]*( GainMeasurementMinusCFs[CurrentLayer] ) +
    (Penalty3/2)*||L*TensorizedGains||^2 + 
    Penalty4*||NetWeights[Layers+1]||_1
    """

    Layers = NetParameters['Layers']
    #C = NetParameters['C']
    LtL = NetParameters['LtL']
    StateDimension = NetParameters['StateDimension']
    BackPropagation = NetParameters['BackPropagation']
    Penalty0 = NetParameters['Penalty0']
    Penalty1 = NetParameters['Penalty1']
    Penalty2 = NetParameters['Penalty2']
    Penalty3 = NetParameters['Penalty3']
    Penalty4 = NetParameters['Penalty4']

    GradsStateEps = [None]
    GradsStateF = [None] * Layers
    GradsStateG = [None] * Layers

    # Loop backward over the layers
    for CurrentLayer in range(Layers - 1, -1, -1):
        Indx = CurrentLayer
        # Common matrix components
        #CommonMat = -NetWeights[Indx] @ C
        CommonMat = -NetWeights[Indx] @ NetWeights[-2]
        if CurrentLayer > 0:
            CommonMatState = CommonMat @ StateJacobians[CurrentLayer]
        #endif
        CommonMat = np.eye(StateDimension) + CommonMat
        # Gradient update matrix at current layer
        if CurrentLayer > 0:
            UpdateMat = (CommonMat @ StateJacobians[CurrentLayer]).T
        #endif
        # Gradient of H with respect to NetWeights[Indx]
        tmpM = Penalty3 * np.tensordot(TensorizedGains, LtL[CurrentLayer,:], axes=([2], [0]))
        Grads[Indx] += tmpM

        # Gradient of S with respect to NetWeights[Indx]
        Grads[-2] += Penalty4 * np.sign(NetWeights[-2])

        if BackPropagation == 'Complete':
            if CurrentLayer == Layers - 1:
                # Gradient of Eps with respect to state at last layer
                GradsStateEps[0] = Penalty0 * (States[:,[-1]] - StateTrue)
            #endif
            # Gradient of Eps with respect to NetWeights[Indx]
            tmpM = (GradsStateEps[0] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
            Grads[Indx] += tmpM
            # Gradient of Eps with respect to NetWeights[-2]
            tmpM = (-(NetWeights[Indx].T @ GradsStateEps[0]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
            Grads[-2] += 0 #tmpM

            # Gradient of F^CurrentLayer
            tmpM = (-Penalty1[CurrentLayer] * (NetWeights[-2].T @ MeasurementWeightMatsSym[CurrentLayer] @ MeasurementMinusCStates[:,[CurrentLayer]])).astype('float64')
            GradsStateF[CurrentLayer] = tmpM
            tmpM = (GradsStateF[CurrentLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
            Grads[Indx] += tmpM
            tmpM = (-(NetWeights[Indx].T @ GradsStateF[CurrentLayer]) @ FStateDynInputs[:,[CurrentLayer]].T - Penalty1[CurrentLayer] * MeasurementWeightMatsSym[CurrentLayer] @ MeasurementMinusCStates[:,[CurrentLayer]] @ States[:,[CurrentLayer+1]].T).astype('float64')
            Grads[-2] += tmpM

            # Gradient of G^CurrentLayer
            #print("PredictorWeightMatsSym[CurrentLayer].shape = ",PredictorWeightMatsSym[CurrentLayer].shape)
            tmpM = (Penalty2[CurrentLayer] * PredictorWeightMatsSym[CurrentLayer] @ GainMeasurementMinusCFs[:,[CurrentLayer]]).astype('float64')
            GradsStateG[CurrentLayer] = tmpM
            tmpM = (GradsStateG[CurrentLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
            Grads[Indx] += tmpM
            tmpM = (-(NetWeights[Indx].T @ GradsStateG[CurrentLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
            Grads[-2] += 0 #tmpM

            if CurrentLayer > 0:
                # Update gradients
                GradsStateEps[0] = UpdateMat @ GradsStateEps[0]
                GradsStateF[CurrentLayer] = UpdateMat @ GradsStateF[CurrentLayer]
                GradsStateG[CurrentLayer] = CommonMatState.T @ GradsStateG[CurrentLayer]
            #endif
            # Loop over past layers for gradient accumulation
            for PastLayer in range(CurrentLayer + 1, Layers):
                tmpM = (GradsStateF[PastLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
                Grads[Indx] += tmpM
                tmpM = (-(NetWeights[Indx].T @ GradsStateF[PastLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
                Grads[-2] += tmpM
                tmpM = (GradsStateG[PastLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
                Grads[Indx] += tmpM
                tmpM = (-(NetWeights[Indx].T @ GradsStateG[PastLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
                Grads[-2] += 0 #tmpM

                if CurrentLayer > 0:
                    GradsStateF[PastLayer] = UpdateMat @ GradsStateF[PastLayer]
                    GradsStateG[PastLayer] = UpdateMat @ GradsStateG[PastLayer]
            #endfor

        elif BackPropagation == 'Truncated':
            if CurrentLayer == Layers - 1:
                GradsStateEps[0] = Penalty0 * (States[:,[-1]] - StateTrue)
                tmpM = (GradsStateEps[0] @ MeasurementMinusCFs[:,[-1]].T).astype('float64')
                Grads[Indx] += tmpM
            #endif
            tmpM = (-(NetWeights[Indx].T @ GradsStateEps[0]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
            Grads[-2] += 0 #tmpM
            
            tmpM = (-Penalty1[CurrentLayer] * (NetWeights[-2].T @ MeasurementWeightMatsSym[CurrentLayer] @ MeasurementMinusCStates[:,[CurrentLayer]])).astype('float64')
            GradsStateF[CurrentLayer] = tmpM
            tmpM = (GradsStateF[CurrentLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
            Grads[Indx] += tmpM
            tmpM = (-(NetWeights[Indx].T @ GradsStateF[CurrentLayer]) @ FStateDynInputs[:,[CurrentLayer]].T - Penalty1[CurrentLayer] * MeasurementWeightMatsSym[CurrentLayer] @ MeasurementMinusCStates[:,[CurrentLayer]] @ States[:,[CurrentLayer+1]].T).astype('float64')
            Grads[-2] += tmpM

            tmpM = (Penalty2[CurrentLayer] * PredictorWeightMatsSym[CurrentLayer] @ GainMeasurementMinusCFs[:,[CurrentLayer]]).astype('float64')
            GradsStateG[CurrentLayer] = tmpM
            tmpM = (GradsStateG[CurrentLayer] @ MeasurementMinusCFs[:,[CurrentLayer]].T).astype('float64')
            Grads[Indx] += tmpM
            tmpM = (-(NetWeights[Indx].T @ GradsStateG[CurrentLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
            Grads[-2] += 0 #tmpM

            if CurrentLayer > 0:
                # Update gradients with respect to the states
                GradsStateEps[0] = UpdateMat @ GradsStateEps[0]
                GradsStateF[CurrentLayer] = UpdateMat @ GradsStateF[CurrentLayer]
                GradsStateG[CurrentLayer] = CommonMatState.T @ GradsStateG[CurrentLayer]
            #endif
            # Loop over past layers and update gradients
            for PastLayer in range(CurrentLayer + 1, Layers):
                tmpM = (-(NetWeights[Indx].T @ GradsStateF[PastLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
                Grads[-2] += tmpM
                tmpM = (-(NetWeights[Indx].T @ GradsStateG[PastLayer]) @ FStateDynInputs[:,[CurrentLayer]].T).astype('float64')
                Grads[-2] += 0 #tmpM

                if CurrentLayer > 0:
                    GradsStateF[PastLayer] = UpdateMat @ GradsStateF[PastLayer]
                    GradsStateG[PastLayer] = UpdateMat @ GradsStateG[PastLayer]
                #endif
            #endfor
        #endif
    return Grads


def UpdateWeights(NetWeights, Grads, Moment1, Moment2, Dynamic, Iterate, GainMask, NetParameters):
    """
    Updates the network's weights using the specified optimizer.

    Parameters:
        NetWeights (list of numpy.ndarray): List containing the network weights.
        Grads (list of numpy.ndarray): List containing the gradients for each weight matrix.
        Moment1 (list of numpy.ndarray): List containing the first moment estimates for Adam optimizer.
        Moment2 (list of numpy.ndarray): List containing the second moment estimates for Adam optimizer.
        Dynamic (int): Index for dynamic parameters.
        Iterate (int): Current iteration number.
        GainMask (numpy.ndarray): Mask for gain updates.
        NetParameters (dict): Dictionary containing network parameters including optimizer settings.

    Returns:
        NetWeights (list of numpy.ndarray): Updated network weights.
        Moment1 (list of numpy.ndarray): Updated first moment estimates.
        Moment2 (list of numpy.ndarray): Updated second moment estimates.
    """
    Layers = NetParameters['Layers']
    ProjectDynamics = NetParameters['ProjectDynamics']
    ActivateCLearning = NetParameters['ActivateCLearning']
    GainLearningRate = NetParameters['GainLearningRate']
    CLearningRate = NetParameters['CLearningRate']
    Optimizer = NetParameters['Optimizer']
    Epsilon = NetParameters['AdamEpsilon']
    if Optimizer == 'SGD':
        # No modification needed for SGD, use Grads as-is.
        pass
    elif Optimizer == 'Adam':
        Beta1 = NetParameters['BetaMoment1']
        Beta2 = NetParameters['BetaMoment2']        
        for Layer in range(Layers):
            # Kalman Gains
            Moment1[Layer] = Beta1*Moment1[Layer] + (1 - Beta1)*Grads[Layer]
            Moment2[Layer] = Beta2*Moment2[Layer]+ (1 - Beta2)*(Grads[Layer] ** 2)

            Moment1Hat = Moment1[Layer] / (1 - Beta1 ** Iterate)
            Moment2Hat = Moment2[Layer] / (1 - Beta2 ** Iterate)

            Grads[Layer] = Moment1Hat / (np.sqrt(Moment2Hat) + Epsilon)
        #endfor
        # C matrix
        Moment1[-2] = Beta1*Moment1[-2] + (1 - Beta1)*Grads[-2]
        Moment2[-2] = Beta2*Moment2[-2]+ (1 - Beta2)*(Grads[-2] ** 2)

        Moment1Hat = Moment1[-2] / (1 - Beta1 ** Iterate)
        Moment2Hat = Moment2[-2] / (1 - Beta2 ** Iterate)

        Grads[-2] = Moment1Hat / (np.sqrt(Moment2Hat) + Epsilon)
    #endif
    # Update weights
    for Layer in range(Layers + 1):
        if Layer < Layers:
            # Kalman Gains
            NetWeights[Layer] -= GainLearningRate * GainMask * Grads[Layer]
        #endif
    #endfor
    if ActivateCLearning == 'Yes':
        # C matrix
        tmpdw = CLearningRate * np.diag(Grads[-2])
        for iw in range(NetWeights[-2].shape[0]):
            NetWeights[-2][iw,iw] -= tmpdw[iw]
            # thresholding:
            if 1 and NetWeights[-2][iw,iw] < 0.5: NetWeights[-2][iw,iw] = 0.0
        #endfor
        if 0: # normalization:
            tmpmax = np.max(np.diag(NetWeights[-2]))
            for iw in range(NetWeights[-2].shape[0]):
                NetWeights[-2][iw,iw] = NetWeights[-2][iw,iw] / tmpmax
            #endfor
        #endif
    #endif
    return NetWeights, Moment1, Moment2


def savePickle(filename,data):
    with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
def loadPickle(filename):
    with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    return data


