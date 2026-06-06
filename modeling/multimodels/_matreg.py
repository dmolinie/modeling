""" Build the regression matrices for the multimodels

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2024
Last revised: April 2026

License: GPLv3
"""

__all__ = ['check_oob_index', 'matreg', 'local_matreg', 'build_inps_outs']

import numpy as np


##############################################################################
##                       Models' Regression Matrices                        ##
##############################################################################

#--------------------------   Check OOB Indexes   ---------------------------#
def check_oob_index(index, max_idx, depth=3, step=0):
    """ Remove all the out-of-bounds indexes

    Take an array of (possibly unsorted) indexes and extract any values
    smaller than `len(index) - depth - order`, as they will be out-of-
    bounds for `depth+step`-ahead prediction models.

    Parameters
    ----------
    index : np.ndarray
        The set of indexes to check, and for which to extract all the
        values that are not out-of-bounds.
    max_idx : np.ndarray
        The maximal accepted index value, without considering the pre-
        diction window (i.e. `depth` and `step`). This value is typi-
        cally the length of the original array of indexes from which
        the `index` argument has been extracted.
    [OPT] depth : int
        Depth for the local interpolated modeling.
            :Default: 2
    [OPT] step : int
        The prediction step (number of timestamps to predict ahead).
        If set to 0, it is the value at the next time (modeling).
            :Default: 0 (modeling only)

    Returns
    -------
    index : np.ndarray
        The set of indexes that are not out-of-bounds, its smaller than
        `len(index) - depth - order`.

    Examples
    --------
    >>> import numpy as np

    # Define the prediction depth and step
    >>> depth = 3
    >>> step = 3

    # Extract the not out-of-bounds indexes
    >>> index = np.arange(100)

    >>> index = check_oob_index(index, len(index), depth, step)
    >>> print(len(index))
    94

    # Extract 80% of the indexes for training
    >>> index = np.arange(100)

    >>> rng = np.random.default_rng(seed=0)
    >>> idx_trn = rng.choice(len(index), int(0.8*len(index)), replace=False)

    >>> ind_trn = index[idx_trn]
    >>> ind_trn = check_oob_index(ind_trn, len(index), depth, step)

    >>> print(len(ind_trn))
    56
    """
    return index[index < max_idx - depth - step]
#----------------------------------------------------------------------------#

#--------------------------   Regression Matrix   ---------------------------#
def matreg(values, depth=3, step=0, dims=None):
    """ Build the regression matrix

    Take a set of values and build the corresponding (global) regression
    matrix. For every value in `values`, associate it with the `depth`-1
    prior ones, and associate this so-built vector of size `depth` with
    the `depth`+`step` value ahead. Do it for every possible value, and
    ignore the out-of-range ones (i.e. ignore the last depth+step ones).
    Ex:
        If the `values` vector is:
            values = (v1, v2, ..., v9)

        And that `depth` is set to 3 and `step` to 1, then:
            inputs      outputs
          (v1 v2 v3)      v5
          (v2 v3 v4)      v6
              ...         ...
          (v5 v6 v7)      v9

    Notice that `step = 0` means that modeling is done on the direct
    following value.

    If `values` is a 1D array or that `dims` is an integer, the regres-
    sion matrix is a 2D array and the vector of outputs is a 1D array;
    if `values` is an ND array (N > 1) and that `dims` is set to None
    or is a list of integers, the vector of inputs is a 3D array, and
    the vector of outputs is a 2D array, the first axis representing
    the dimension (one regression matrix/vector of outputs per row).

    Parameters
    ----------
    values : ND np.ndarray
        The values for which to build the regression matrix.
    [OPT] depth : int
        Depth for the local interpolated modeling.
            :Default: 3
    [OPT] step : int
        The prediction step (number of timestamps to predict ahead).
        If set to 0, it is the value at the next time (modeling).
            :Default: 0 (modeling only)
    [OPT] dims : None, int or list of ints
        The dimensions of the database for which to build the regres-
        sion matrices. If None, build them for every dimension; if
        int, do it only for that specific dimension (index); if list
        of integers, do it for every dimension (index) in that list.
            :Default: None (all dimensions)

    Returns
    -------
    inp : 2D or 3D np.ndarray
        The regression matrices, one per item in `col`.
    out : 1D or 2D np.ndarray
        The outputs corresponding to the regression matrices `inp`.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data and wrap them into a set of Clusters
    >>> data = np.linspace(0, 100, 10000).reshape(-1, 10)
    >>> index = np.arange(len(data))

    # Generate a semi-random number generator
    >>> rng = np.random.default_rng(seed=0)
    >>> mask = np.full(len(index), False, dtype=bool)

    # Generate a set of training indexes (80% of the database)
    >>> idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    >>> mask[idx_trn] = True

    # Separate set of indexes into training and testing indexes
    >>> ind_trn = index[mask]
    >>> ind_gen = index[~mask]

    # Remove the out-of-bounds indexes, if any
    >>> ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    >>> ind_gen = check_oob_index(ind_gen, len(index), depth, step)

    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    >>> ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]

    # Build the regression matrix of the `cols` dimensions
    >>> inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))
    >>> print(inputs.shape, outputs.shape)
    """

    # Case `values` is a 1D line vector
    if np.ndim(values) == 1:
        inp = np.empty((len(values)-depth-step, depth), float)
        for j in range(depth):
            inp[:, j] = values[j:j-depth-step]              # Reg. matrix
        out = values[depth+step:]                           # Vec. of outputs

    # Case `values` is a 1D column vector
    elif np.shape(values)[1] == 1:
        inp = np.empty((len(values)-depth-step, depth), float)
        for j in range(depth):
            inp[:, j] = values[j:j-depth-step, 0]           # Reg. matrix
        out = values[depth+step:, 0]                        # Vec. of outputs

    # Case `dims` is set to a unique integer
    elif isinstance(dims, int):
        inp = np.empty((len(values)-depth-step, depth), float)
        for j in range(depth):
            inp[:, j] = values[j:j-depth-step, dims]        # Reg. matrix
        out = values[depth+step:, dims]                     # Vec. of outputs

    # Case there are several dimensions to retrieve
    else:

        # If dims set to None, consider all the dimensions
        if dims is None:
            dims = np.arange(values.shape[1])

        inp = np.empty((len(dims), len(values)-depth-step, depth), float)
        for i, col in enumerate(dims):
            for j in range(depth):
                inp[i, :, j] = values[j:j-depth-step, col]  # Reg. matrix
        out = np.empty((len(dims), len(values)-depth-step), float)
        for i, col in enumerate(dims):
            out[i, :] = values[depth+step:, col]            # Vec. of outputs

    return inp, out
#----------------------------------------------------------------------------#

#-------------------   Build Local Regression Matrices   --------------------#
def local_matreg(inputs, outputs, ind_trn_lst):
    """ Build the local regression matrices from the clusters' data

    Take the regression matrix of the full database (`inputs`), the cor-
    responding vector of objective values (`outputs`) and the indexes of
    the training data of the local groups (`ind_trn_lst`), and build the
    regression matrix and vector of objective outputs for every local
    group of data (i.e. the clusters).

    See function `matreg` for details about the data formats (in parti-
    cular, the when using 2D or 3D `inputs` arrays).

    Parameters
    ----------
    inputs : 2D or 3D np.ndarray
        The information matrix of every cluster (one per one).
    outputs : 1D or 2D np.ndarray
        The outputs associated with the information matrices: the
        outputs' vector at cell i corresponds to the inputs' infor-
        mation matrix at cell i. Every vector must be same length
        as the corresponding information matrix.
    ind_trn_lst : list of 1D array
        The list of the training data indexes, one array per local
        group of data (cluster).

    Returns
    -------
    mat_inputs : (list of) list of ND np.ndarrays
        The local regression matrices for any set of data (cluster).
        If `inputs` is a list (cf. function `matreg`), do it for any
        item in this list and return the list of matrices.
    mat_outputs : (list of) list of ND np.ndarrays
        The outputs corresponding to the regression matrices `mat_inputs`.
        If `outputs` if a list (cf. function `matreg`), do it for every
        item in this list and return the list of objective outputs.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data and wrap them into a set of Clusters
    >>> data = np.linspace(0, 100, 10000).reshape(-1, 10)
    >>> index = np.arange(len(data))

    # Generate a semi-random number generator
    >>> rng = np.random.default_rng(seed=0)
    >>> mask = np.full(len(index), False, dtype=bool)

    # Generate a set of training indexes (80% of the database)
    >>> idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    >>> mask[idx_trn] = True

    # Separate set of indexes into training and testing indexes
    >>> ind_trn = index[mask]
    >>> ind_gen = index[~mask]

    # Remove the out-of-bounds indexes, if any
    >>> ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    >>> ind_gen = check_oob_index(ind_gen, len(index), depth, step)

    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    >>> ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]

    # Build the regression matrix of the `cols` dimensions
    >>> inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))

    # Build the regression matrices per cluster (training data)
    >>> mat_inputs, mat_outputs = local_matreg(inputs, outputs, ind_trn_lst)
    >>> print([mat.shape for mat in mat_inputs[0]])
    [(20, 3), (20, 3), (20, 3), (14, 3)]
    >>> print([mat.shape for mat in mat_outputs[0]])
    [(20,), (20,), (20,), (14,)]
    """

    mat_inputs, mat_outputs = [], []

    # Case the original data is 1D vector (thus 2D regression matrix)
    if inputs.ndim == 2:
        matinp, matout = [], []
        for idx in ind_trn_lst:                # Build the regression matrix
            matinp.append(inputs[idx])         # and the corresponding outputs
            matout.append(outputs[idx])        # for every local group of data
        mat_inputs.append(matinp)
        mat_outputs.append(matout)

    # Case the original data is ND vector (thus 3D regression matrix)
    else:
        for inp, out in zip(inputs, outputs):  # For any dimension considered
            matinp, matout = [], []
            for idx in ind_trn_lst:            # Build the regression matrix
                matinp.append(inp[idx])        # and the corresponding outputs
                matout.append(out[idx])        # for every local group of data
            mat_inputs.append(matinp)
            mat_outputs.append(matout)

    return mat_inputs, mat_outputs
#----------------------------------------------------------------------------#

#----------------------   Inputs & Outputs Matrices   -----------------------#
def build_inps_outs(inputs, outputs, ind_trn, ind_gen):
    """ Build the matrices of inputs & outputs

    Take the regression matrix of the full database (`inputs`), the cor-
    responding vector of objective values (`outputs`), and the indexes
    of the training (`ind_trn`) and testing (`ind_gen`) data, and build
    the matrices of the training inputs and their corresponding outputs,
    and the testing inputs and their corresponding outputs.

    If `inputs` and `outputs` are lists of arrays (e.g. one array per
    dimension), build the inputs and outputs matrices for each of them.

    Parameters
    ----------
    inputs : 2D or 3D np.ndarray
        The information matrix of every cluster (one per one).
    outputs : 1D or 2D np.ndarray
        The outputs associated with the information matrices: the
        outputs' vector at cell i corresponds to the inputs' infor-
        mation matrix at cell i. Every vector must be same length
        as the corresponding information matrix.
    ind_trn : 1D np.ndarray
        The indexes of the training data in the full database.
    ind_gen : 1D np.ndarray
        The indexes of the testing data in the full database.

    Returns
    -------
    inp_trn : (list of) np.ndarray(s)
        The training inputs; same type as `inputs`.
    out_trn : (list of) np.ndarray(s)
        The training outputs; same type as `outputs`.
    inp_gen : (list of) np.ndarray(s)
        The testing inputs; same type as `inputs`.
    out_gen : (list of) np.ndarray(s)
        The testing outputs; same type as `outputs`.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data and wrap them into a set of Clusters
    >>> data = np.linspace(0, 100, 10000).reshape(-1, 10)
    >>> index = np.arange(len(data))

    # Generate a semi-random number generator
    >>> rng = np.random.default_rng(seed=0)
    >>> mask = np.full(len(index), False, dtype=bool)

    # Generate a set of training indexes (80% of the database)
    >>> idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    >>> mask[idx_trn] = True

    # Separate set of indexes into training and testing indexes
    >>> ind_trn = index[mask]
    >>> ind_gen = index[~mask]

    # Remove the out-of-bounds indexes, if any
    >>> ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    >>> ind_gen = check_oob_index(ind_gen, len(index), depth, step)

    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    >>> ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]

    # Build the regression matrix of the `cols` dimensions
    >>> inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))

    # Build the regression matrices per cluster (training data)
    >>> mat_inputs, mat_outputs = local_matreg(inputs, outputs, ind_trn_lst)

    # Build the Training & Testing sets
    >>> inp_trn, out_trn, inp_gen, out_gen =\
    ...     build_inps_outs(inputs, outputs, ind_trn, ind_gen)
    >>> print(inp_trn.shape, out_trn.shape, inp_gen.shape, out_gen.shape)
    (4, 795, 3) (4, 795) (4, 199, 3) (4, 199)
    """

    # Case data are 1D vectors
    if inputs.ndim == 2:
        inp_trn, inp_gen = inputs[ind_trn], inputs[ind_gen]
        out_trn, out_gen = outputs[ind_trn], outputs[ind_gen]

    # Case data are ND vectors
    else:
        inp_trn, inp_gen = inputs[:, ind_trn], inputs[:, ind_gen]
        out_trn, out_gen = outputs[:, ind_trn], outputs[:, ind_gen]

    return inp_trn, out_trn, inp_gen, out_gen
#----------------------------------------------------------------------------#

##############################################################################
