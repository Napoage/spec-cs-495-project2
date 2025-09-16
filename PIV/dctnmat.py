import numpy as np


def dctn(y, w=None):
    """
    DCTN N-D discrete cosine transform.
    (his is ported matlab code)
    Args:
    y: array_like
        Input array.
    w: array_like, optional
        Weights used by the program. If not provided, the weights are calculated
        during the function call and can be reused for subsequent calls.

    Returns:
    y: ndarray
        Discrete cosine transform of the input array.
    w: list
        Weights used in the computation.
        
    Credits: 
        -Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
        -Engineered by: Deep Analytics LLC http://www.deepvt.com/
        -Ported from: 
             Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
                     U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
        -Authors: Makayla Hayes
    """
    # Convert input to double
    y = np.asarray(y, dtype=np.double)

    # Get dimensions of input
    sizy = y.shape

    # Squeeze y if it's a vector
    if y.ndim == 1:
        y = np.squeeze(y)

    dimy = y.ndim

    # Initialize weights if not provided
    if w is None:
        w = []
        for dim in range(dimy):
            n = sizy[0] if dimy == 1 else sizy[dim]
            w.append(np.exp(1j * np.arange(n) * np.pi / 2 / n))

    # Check if y is real or complex and perform DCT accordingly
    if np.iscomplexobj(y):
        y = dctn(np.real(y), w) + 1j * dctn(np.imag(y), w)
    else:
        for dim in range(dimy):
            siz = y.shape
            n = siz[0]
            idx = np.hstack((np.arange(0, n, 2), np.arange(2 * (n // 2), 0,
                                                           -2)))
            y = y[idx - 1, ...]
            y = np.reshape(y, (n, -1), order='F')  #might need to add order='F'
            y = y * np.sqrt(2 * n)
            y = np.fft.ifft(y, axis=0)  #something weird is happening here
            y = y * (w[dim].reshape((-1, 1)))
            y = np.real(y)
            y[0, ...] = y[0, ...] / np.sqrt(2)
            y = np.reshape(y, siz, order='F')
            y = np.swapaxes(y, 0, 1)

    y = np.reshape(y, sizy, order='F')

    return y
