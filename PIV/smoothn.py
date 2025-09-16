import numpy as np
from scipy.ndimage import distance_transform_bf
from scipy.fftpack import dctn, idctn
from scipy.optimize import minimize_scalar
from dctnmat import dctn
from idctnmat import idctn


def smoothn(y, W=None, s=None, MaxIter=100, TolZ=1e-3):
    """
    Help function -Robust spline smoothing for 1-D to N-D data.
    Smoooths the array y using paramter s. The larger s is the more smooth the array will be.
    W will not be given based on the matlab code and maxIter, and Tolz will be default values. 
    
    Args:
        -y: array
            nxd array that will be smoothed
        -W: array
            array of weights - not used here
        -s: double
            the smoothing value hard-coded to 0.5 based on matlab code
        -MaxIter: int
            maximum number of iteration set to 100
        -Tolz: float
            tolerance value set to 1e-3
    Returns:
        -z: array
            array of smoothed values
            
     Credits: 
        -Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
        -Engineered by: Deep Analytics LLC http://www.deepvt.com/
        -Ported from:
             Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
                     U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
        -Authors: Makayla Hayes
    """
    # Check input arguments
    if W is None:
        W = np.ones_like(y)
    if s is None:
        s = np.nan  # Placeholder for automatic determination of s

    # Test and prepare the variables
    y = np.double(y)
    sizy = y.shape
    noe = np.prod(sizy)
    if noe < 2:
        return y

    # Smoothness parameter and weights
    W = np.broadcast_to(W, sizy)
    if s is not np.nan:
        if not isinstance(s, (int, float)):
            raise ValueError("The smoothing parameter must be a scalar.")
        elif s < 0:
            raise ValueError("The smoothing parameter must be >= 0.")

    # "Maximal number of iterations" criterion
    if not isinstance(MaxIter, int) or MaxIter < 1:
        raise ValueError("MaxIter must be an integer >= 1")

    # "Tolerance on smoothed output" criterion
    if not 0 < TolZ < 1:
        raise ValueError("TolZ must be in ]0, 1[")

    # Initial Guess criterion
    isinitial = False

    # Weights
    IsFinite = np.isfinite(y)
    W = W * IsFinite
    if np.any(W < 0):
        raise ValueError("Weights must all be >= 0")
    else:
        W = W / np.max(W)

    # Weighted or missing data?
    isweighted = np.any(W < 1)

    # Robust smoothing?
    isrobust = False

    # Automatic smoothing?
    isauto = np.isnan(s)

    # Creation of the Lambda tensor
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in range(d):
        siz0 = np.ones((1, d), dtype=int)
        siz0[0, i] = sizy[i]
        idx = np.arange(1, sizy[i] + 1)
        res = idx.reshape((-1, 1) if i == 0 else (1, -1))
        reshaped = res - 1
        cos_vals = np.cos(np.pi * reshaped / sizy[i])
        Lambda += cos_vals
    Lambda = -2 * (d - Lambda)
    if not isauto:
        Gamma = 1 / (1 + s * Lambda**2)
    # Upper and lower bound for the smoothness parameter
    N = np.sum(np.array(sizy) != 1)
    hMin = 1e-6
    hMax = 0.99
    sMinBnd = ((
        (1 + np.sqrt(1 + 8 * hMax**(2 / N))) / 4 / hMax**(2 / N))**2 - 1) / 16
    sMaxBnd = ((
        (1 + np.sqrt(1 + 8 * hMin**(2 / N))) / 4 / hMin**(2 / N))**2 - 1) / 16

    # Initialize before iterating
    z = np.zeros(sizy)
    z0 = z
    y[~IsFinite] = 0
    Wtot = W
    tol = 1
    nit = 0
    errp = 0.1
    rf = 1 + 0.75 * (1 if isweighted else 0)
    # Main iterative process
    while tol > TolZ and nit < MaxIter:
        nit += 1
        DCTy = dctn(Wtot * (y - z) + z)

        if isauto and not np.log2(nit) % 1:
            # GCV method
            result = minimize_scalar(gcv,
                                     bounds=(np.log10(sMinBnd),
                                             np.log10(sMaxBnd)),
                                     args=(DCTy, Wtot, IsFinite))
            s = 10**result.x

        if isrobust:
            pass  # Implement robust smoothing if required

        # Update z
        z = rf * idctn(Gamma * DCTy) + (1 - rf) * z

        # Update tolerance
        tol = np.linalg.norm(z0 - z) / np.linalg.norm(z)
        z0 = z.copy()

    return z


def gcv(p, DCTy, Wtot, IsFinite, Lambda, y):

    ##isauto is set to False so should not be used
    s = 10**p
    Gamma = 1 / (1 + s * Lambda**2)

    if np.sum(Wtot) / np.sum(IsFinite) > 0.9:
        RSS = np.linalg.norm(DCTy * (Gamma - 1))**2
    else:
        yhat = idctn(Gamma * DCTy)
        RSS = np.linalg.norm(
            np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite]))**2

    TrH = np.sum(Gamma)
    GCVscore = RSS / np.sum(IsFinite) / (1 - TrH / np.prod(IsFinite.shape))**2
    return GCVscore
