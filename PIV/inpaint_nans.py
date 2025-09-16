from scipy.sparse import csc_matrix
import numpy as np
from scipy.sparse.linalg import lsqr


def inpaint_nans_spring(array):
    """
    Helper function to fillin holes in an array using the spring metaphor method
    (spring metaphor was the method used in matlab could not find a python function to do it so
    ported the matlab one)
    Args:
        -array: array
            a nxm array with NaNs to be filled in
    Returns:
        -B: array
            a nxm array with filled in NaNs
    
     Credits: 
        -Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
        -Engineered by: Deep Analytics LLC http://www.deepvt.com/
        -Ported from:
             Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
                     U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
        -Authors: Makayla Hayes
    """
    # Flatten the array
    n, m = array.shape[0], array.shape[1]
    A = array.flatten(order='F')
    nm = n * m

    # Find NaN values and their indices
    k = np.isnan(A)
    nan_list = np.where(k)[0]
    known_list = np.where(~k)[0]
    nan_count = len(nan_list)  #off from matlab

    # Convert linear indices to (row, column) form
    nr, nc = np.unravel_index(nan_list, (n, m), order='F')
    # Combine all forms of indices into one array
    nan_list = np.column_stack((nan_list, nr, nc))
    # Define the spring analogy matrix
    hv_list = np.array([[-1, -1, 0], [1, 1, 0], [-n, 0, -1], [n, 0, 1]])

    # Initialize an empty list to store the springs
    hv_springs = []
    # Iterate over the rows of hv_list
    for i in range(4):
        # Calculate the positions of horizontal and vertical neighbors for each node
        hvs = nan_list + np.tile(hv_list[i, :], (nan_count, 1))
        k = (hvs[:, 1] >= 0) & (hvs[:, 1] < n -
                                1) & (hvs[:, 2] >= 0) & (hvs[:, 2] < m - 1)
        first_true_index = np.argmax(k)
        hv_springs.extend(np.column_stack((nan_list[k, 0], hvs[k, 0])))

    # Convert the list to a NumPy array
    hv_springs = np.array(hv_springs)

    # Delete replicate springs
    hv_springs = np.unique(np.sort(hv_springs, axis=1), axis=0)

    # Build sparse matrix of connections
    nhv = hv_springs.shape[0]
    row_indices = np.tile(np.arange(nhv)[:, np.newaxis],
                          (1, 2)).flatten(order='F')
    data = np.tile(np.array([1, -1])[np.newaxis, :],
                   (nhv, 1)).flatten(order='F')
    col_indices = hv_springs.flatten(order='F')
    springs = csc_matrix((data, (row_indices, col_indices)), shape=(nhv, nm))

    # Eliminate knowns
    rhs = -springs[:, known_list].dot(A[known_list]).reshape(-1, 1)

    # Solve
    B = A.copy()
    B[nan_list[:, 0]] = lsqr(springs[:, nan_list[:, 0]],
                             rhs.flatten(order='F'))[0]
    B = np.reshape(B, (n, m), order='F')

    return B
