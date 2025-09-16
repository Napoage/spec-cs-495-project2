import numpy as np


def ordfilt2(A, order, domain, padding='constant', padval=None):
    """
    Function: ordfilt2

    Description:
    Perform order-statistic filtering on a 2D matrix A using a specified neighborhood.

    Args:
    - A: numpy.ndarray
        Input 2D matrix to be filtered.
    - order: int
        Order of the element to be retrieved from the sorted neighborhood.
    - domain: numpy.ndarray
        Neighborhood or filter for selecting neighboring elements.
    - padding: str, optional
        Padding method to use for handling borders. Default is 'constant'.
    - padval: {None, scalar}, optional
        Value to use for padding when padding='constant'. Default is None.

    Returns:
    - filtered: numpy.ndarray
    F   iltered matrix after order-statistic filtering.

    Exceptions:
    - ValueError: Raised if the padding type is not supported.

    Notes:
    - The function performs order-statistic filtering on the input matrix A using a specified neighborhood defined by the domain parameter.
    - The order parameter specifies the order of the element to be retrieved from the sorted neighborhood.
    - The padding parameter determines the method used for handling borders during filtering.
    - The padval parameter is only used when padding='constant', specifying the value to use for padding.
    
     Credits: 
        -Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
        -Engineered by: Deep Analytics LLC http://www.deepvt.com/
        -Ported from:
             Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
                     U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
        -Authors: Makayla Hayes
    """

    # Pad the input array
    if padding == 'constant':
        A_padded = np.pad(A,
                          pad_width=1,
                          mode='constant',
                          constant_values=padval)
    elif padding == 'zeros':
        A_padded = np.pad(A, pad_width=1, mode='constant', constant_values=0)
    else:
        raise ValueError("Padding type not supported.")

    # Initialize the filtered array
    filtered = np.zeros_like(A, dtype=A.dtype)

    # Iterate over each pixel
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood specified by the domain
            neighbors = A_padded[i:i + 3, j:j + 3][domain]

            # Sort the neighborhood and retrieve the order-th element
            sorted_neighbors = np.sort(neighbors.flatten(order='F'))
            filtered[i, j] = sorted_neighbors[order - 1]

    return filtered


def medfilt2(A, *args):
    """
Function: medfilt2

Description:
    Perform median filtering on a 2D matrix A using a specified neighborhood.

Args:
- A: numpy.ndarray
    Input 2D matrix to be filtered.

Variable Arguments:
- padding: str, optional
    Padding method to use for handling borders. Default is 'constant'.
- domain: numpy.ndarray, optional
    Neighborhood or filter for median calculation. Default is a 3x3 square.

Returns:
- retval: numpy.ndarray
    Filtered matrix after median filtering.

Exceptions:
- ValueError: Raised if the input matrix A is not a real matrix with floating-point data type.
- ValueError: Raised if an unrecognized option is passed as an argument.
- ValueError: Raised if the domain specified is not a valid filter or neighborhood.

Notes:
- The function applies median filtering on the input matrix A using a specified neighborhood defined by the domain parameter.
- The padding parameter determines the method used for handling borders during filtering.
- The domain parameter specifies the neighborhood or filter for the median calculation. It should be a boolean matrix.
- If the size of the neighborhood is even, the function performs averaging between two median values to avoid overflow issues.
"""
    if len(args) < 1 or len(args) > 3:
        print("Usage: medfilt2(A, padding='zeros', domain=np.ones((3, 3)))")

    if not isinstance(A, np.ndarray) or not np.issubdtype(A.dtype, np.floating):
        raise ValueError("medfilt2: A must be a real matrix")

    # Defaults
    padding = "constant"
    domain = np.ones((3, 3), dtype=bool)

    for opt in args:
        if isinstance(opt, str) or np.isscalar(opt):
            padding = opt
        elif isinstance(opt, (np.ndarray, np.generic)) or isinstance(opt, bool):
            if isinstance(
                    opt, np.ndarray) and not np.issubdtype(opt.dtype, np.bool_):
                if len(opt.shape) == 2 and opt.shape[0] == opt.shape[1]:
                    domain = opt.astype(bool)
                else:
                    raise ValueError(
                        "medfilt2: to specify filter instead of dimensions use a matrix of boolean class."
                    )
            else:
                domain = opt
        else:
            raise ValueError("medfilt2: unrecognized option")

    # Median filtering
    n = np.count_nonzero(domain)
    if (n - 2 * (n // 2)) == 0:  # n even
        nth = n // 2
        a = ordfilt2(A, nth, domain, padding)
        b = ordfilt2(A, nth + 1, domain, padding)
        retval = a / 2 + b / 2
    else:
        nth = n // 2 + 1
        retval = ordfilt2(A, nth, domain, padding)

    return retval
