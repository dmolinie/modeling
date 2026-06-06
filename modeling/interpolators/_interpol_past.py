""" Multi-Models using past data

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2022
Last revised: April 2026

License: GPLv3
"""

__all__ = ['InterpolatorPast', 'MultiModelPast']

import numpy as np

import modeling.optimization._optimizers as opt

# Default cost function
cost_func = opt.mse


##############################################################################
##                          Past-based Multi-Model                          ##
##############################################################################

class InterpolatorPast():
    r""" Regression Matrix-based past-values Model

    Take a regression matrix and its corresponding objective values and
    train the model. A regression matrix is composed of N vectors, each
    composed of L past values. The objective values are vectors of size
    N, in which the value at line n is the regression matrix's vector at
    line n. The temporal gap between the most recent value in the regres-
    sion vector and the objective value is the prediction step.

    The local model is defined as a polynomial interpolation of the L
    prior values of the regression vector, each powered to every value
    up to the degree `order` (cf. class's constructor), denoted as D,
    and weighted by the `theta` coefficients:
        ŷ(val, theta) = sum_d^D sum_m^L theta[d*D+m] * val[m]^{d+1}

    Note that the exponentiation starts from 1 and ends with `order+1`.

    Constructor
    -----------
    __init__(order=1)

    Attributes
    ----------
    regressors : np.ndarray, getter only
        The regressors' parameter vectors.

    Methods
    -------
    fit(inputs, outputs)
        Select the modeling function and train the model.
    predict(inputs)
        Use the Model to predict the data in response to `inputs`.

    Examples
    --------
    >>> import numpy as np
    >>> import modeling.optimization as opt

    >>> def function(vals):
    ...     ''' Function to interpolate '''    
    ...     return vals**3 - 5.*vals**2 - 10.*vals

    >>> def matreg(vals, depth, step):
    ...     ''' Build the regression matrix '''
    ...     inputs = np.empty((len(vals)-depth-step, depth), float)
    ...     for i in range(depth):
    ...         inputs[:, i] = vals[i:-depth-step+i]
    ...     outputs = vals[depth+step:]
    ...     return inputs, outputs

    # Generate dummy data
    >>> values = function(np.linspace(-1, +1, 1000, float))

    # Build the global regression matrix
    >>> inputs, outputs = matreg(values, depth:=3, step:=0)

    # Split the dataset (regression matrix) into a trainig & testing sets
    >>> frac = int(len(inputs) * 0.8)
    >>> inps_trn, inps_gen = inputs[:frac], inputs[frac:]
    >>> outs_trn, outs_gen = outputs[:frac], outputs[frac:]

    # Instantiate & train the model
    >>> model = InterpolatorPast(order=3)
    >>> model.fit(inps_trn, outs_trn)

    # Evaluate the model's accuracy
    >>> print("Training", f"%.5e" %opt.mse(outs_trn, model.predict(inps_trn)))
    >>> print("Testing", f"%.5e" %opt.mse(outs_gen, model.predict(inps_gen)))
    Training 1.49936e-04
    Testing 3.67683e-02
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, order=1):
        """ Instantiate an InterpolatorPast object (constructor)

        Parameters
        ----------
        [OPT] order : int
            The maximal exponentiation to which the inputs are raised
            to. The estimation is a linear combination of the values
            raised to every power up to `order`.
                :Default: 1

        Examples
        --------
        >>> model = InterpolatorPast()
        >>> model = InterpolatorPast(order=5)
        """
        self._order = order                     # Max power for input values
        self._regressors = None                 # Local models coefficients
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def regressors(self):
        """ Get the coefficients of the regressors """
        return self._regressors
    #------------------------------------------------------------------------#

    #---------------------   Initialization Functions   ---------------------#
    def _matinfo(self, values):
        """ Information Matrix of the local models """
        depth, pos = values.shape[-1], 0
        phi = np.empty((len(values), depth*self._order), float)
        for j in range(1, self._order+1):           # Powered inputs
            phi[:, pos:pos+depth] = values**j
            pos += depth
        return phi
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Functions   -----------------------#
    def _estimate(self, values, theta):
        """ Model estimate """
        depth, pos = values.shape[-1], 0
        acc = np.zeros(len(values), float)
        for j in range(1, self._order+1):                    # Powered inputs
            acc += (theta[pos:pos+depth]*(values**j)).sum(1)   # Local model
            pos += depth
        return acc

    def predict(self, inputs):
        """ Use the Model to predict the data in response to `inputs`

        Parameters
        ----------
        inputs : np.ndarray
            The past data, typically sorted in time order.

        Returns
        -------
        pred : np.ndarray
            The estimated data.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        >>> def function(vals):
        ...     ''' Function to interpolate '''    
        ...     return vals**3 - 5.*vals**2 - 10.*vals

        >>> def matreg(vals, depth, step):
        ...     ''' Build the regression matrix '''
        ...     inputs = np.empty((len(vals)-depth-step, depth), float)
        ...     for i in range(depth):
        ...         inputs[:, i] = vals[i:-depth-step+i]
        ...     outputs = vals[depth+step:]
        ...     return inputs, outputs

        # Generate dummy data
        >>> value = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inputs, outputs = matreg(value, depth:=3, step:=0)

        # Instantiate & train the model
        >>> model = InterpolatorPast(order=3)
        >>> model.fit(inputs, outputs)

        # Evaluate the model's accuracy
        >>> print(f"%.5e" %opt.mse(outputs, model.predict(inputs)))
        1.55335e-04
        """
        if np.ndim(inputs) == 1:
            inputs = np.reshape(inputs, (1, -1))
        return self._estimate(inputs, self._regressors)
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, inputs, outputs):
        """ Select the modeling function and train the model

        Parameters
        ----------
        inputs : 2D np.ndarrays
            The information matrix of the inputs; any row of this matrix
            is assumed to be the vector inputs for the model.
        outputs : 1D np.ndarrays
            The outputs associated with `inputs`: the outputs' vector at
            cell i corresponds to the information matrix's cell i.

        Returns
        -------
        None : directly set the `regressors` attribute.

        Examples
        --------
        >>> import numpy as np
        >>> from modeling.optimization import cost

        >>> def function(vals):
        ...     ''' Function to interpolate '''    
        ...     return vals**3 - 5.*vals**2 - 10.*vals

        >>> def matreg(vals, depth, step):
        ...     ''' Build the regression matrix '''
        ...     inputs = np.empty((len(vals)-depth-step, depth), float)
        ...     for i in range(depth):
        ...         inputs[:, i] = vals[i:-depth-step+i]
        ...     outputs = vals[depth+step:]
        ...     return inputs, outputs

        # Generate dummy data
        >>> value = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inputs, outputs = matreg(value, depth:=3, step:=0)

        # Instantiate & train the model
        >>> model = InterpolatorPast(order=3)
        >>> model.fit(inputs, outputs)
        >>> print(model.regressors)
        [ 1.41640642e-01  3.34237273e-01  5.27271264e-01 -2.28240661e-02
          2.42496583e-03  2.00551302e-02 -9.08100471e-03 -3.84705093e-04
          9.31621297e-03]
        """
        self._regressors = opt.least_squares(self._matinfo(inputs), outputs)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Past-based Multi-Model                          ##
##############################################################################

class MultiModelPast():
    r""" Regression Matrix-based past-values Multi-Model

    Take a set of regression matrices and the corresponding objective
    values as a set of vectors of outputs and train the multi-model.
    Any regression matrix is composed of N vectors, each composed of L
    past values. The objective values are vectors of size N, in which
    the value at line n corresponds to the regression matrix's vector
    at line n. The temporal gap between the most recent value in the
    regression vector and the objective value is the prediction step.

    By denoting `val` a regression vector, the multi-model is defined as
    a linear combination of the estimates outputted by any of the K local
    models, each multiplied by a Gaussian-like membership function (MF):
        ŷ(val, theta, beta) = sum_k^K w_k(val, beta_k) * f_k(val, theta_k)

    where `beta` and `theta` are the sets of the K vectors `beta_k` and
    `theta_k`, each being the parameter vector of the membership function
    and local model, respectively, of the k-th regression matrix.

    The local models are defined as a polynomial interpolation of the L
    prior values of the regression vector, each powered to every value
    up to the degree `order` (cf. class's constructor), denoted as D
    (note that the exponentiation starts from 1 and ends with `order+1`):
        f_k(val, theta_k) = sum_d^D sum_m^L theta_k[d*D+m] * val[m]^{d+1}

    The membership functions are Gaussian-like functions, defined as:
        rho_k(val, beta_k) = exp( -(val - beta_k[0])^2 / (2*beta_k[1]^2) )

    with `w_k` the weight of model `p` defined by:
        w_k(val, beta_k) = rho_k(val, beta_k) / (sum_k rho_k(val, beta_k))

    In practice, both `theta` and `beta` are flattened in the implementation
    for practical reasons, but the `regressors` (theta) and `membership`
    (beta) class' properties are reshaped into K sets of (local) vectors.

    Constructor
    -----------
    __init__(order=1, level='local_beta', stop=100)

    Attributes
    ----------
    level : str, getter & setter
        The optimization level.
    regressors : np.ndarray, getter only
        The regressors' parameter vectors.
    membership : np.ndarray, getter only
        The membership functions' parameter vectors.

    Methods
    -------
    fit(inputs, outputs)
        Select the modeling function and train the multi-model.
    predict(inputs)
        Multi-Model estimate (local estimate + membership function).

    Examples
    --------
    >>> import numpy as np
    >>> import modeling.optimization as opt

    >>> def function(vals):
    ...     ''' Function to interpolate '''    
    ...     return vals**3 - 5.*vals**2 - 10.*vals

    >>> def matreg(vals, depth, step):
    ...     ''' Build the regression matrix '''
    ...     inps_glb = np.empty((len(vals)-depth-step, depth), float)
    ...     for i in range(depth):
    ...         inps_glb[:, i] = vals[i:-depth-step+i]
    ...     outs_glb = vals[depth+step:]
    ...     return inps_glb, outs_glb

    # Generate dummy data
    >>> values = function(np.linspace(-1, +1, 1000, float))

    # Build the global regression matrix
    >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

    # Split the dataset into K subsets
    >>> K = 3
    >>> N = len(inps_glb) // K
    >>> inps_loc, outs_loc = [], []
    >>> for i in range(K):
    ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
    ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

    # Build the local regression matrices
    >>> mat_inps_trn, mat_outs_trn = [], []
    >>> mat_inps_gen, mat_outs_gen = [], []
    >>> for inp, out in zip(inps_loc, outs_loc):
    ...     # inp_trn, inp_gen, out_trn, out_gen = \
    ...     #    sklearn.model_selection.train_test_split(inp, out)
    ...     frac = int(len(inp) * 0.8)
    ...     inp_trn, inp_gen = inp[:frac], inp[frac:]
    ...     out_trn, out_gen = out[:frac], out[frac:]
    ...     mat_inps_trn.append(inp_trn)
    ...     mat_inps_gen.append(inp_gen)
    ...     mat_outs_trn.append(out_trn)
    ...     mat_outs_gen.append(out_gen)

    >>> for level in ('local', 'local_theta', 'local_beta', 'global'):>
    ...     # Instantiate & train the multimodel
    ...     model = MultiModelPast(order=3, level=level, stop=100)
    ...     model.fit(mat_inps_trn, mat_outs_trn)
    ...     # Evaluate the multimodel's accuracy
    ...     cost = 0.
    ...     for inp, out in zip(mat_inps_gen, mat_outs_gen):
    ...         cost += opt.mse(out, model.predict(inp))
    ...     print(level, '\t', f"%.5e" %cost)
    local           2.35734e+00
    local_theta     1.82544e+00
    local_beta      8.24367e-03
    global          8.24298e-03
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, order=1, level='local_beta', stop=100):
        """ Instantiate a MultiModelPast object (constructor)

        Parameters
        ----------
        [OPT] order : int
            The maximal power to which the inputs are raised to. The
            estimation is a linear combination of the values raised
            to every power up to `order`.
                :Default: 1
        [OPT] level : str
            The modeling level: local, semi-global or global:
              - 'local':       both theta and beta local;
              - 'local_theta': theta local and beta global;
              - 'local_beta':  theta global and beta local;
              - 'global':      both theta and beta global.
            The last one often gives the best results, but also is the
            slowest; the 'local_beta' variant is often a good trade-off
            between accuracy and execution speediness.
                :Default: 'local_beta'
        [OPT] stop : int
            Maximal number of iterations for iterative methods.
                :Default: 100

        Examples
        --------
        >>> model = MultiModelPast(order=2, level='local')
        >>> model = MultiModelPast(order=5, level='global', stop=100)
        """
        self._stop = stop                   # Stopping criterion
        self._level = level                 # Level of the training
        self._order = order                 # Max power for input values
        self._regressors = None             # Local models coefficients
        self._membership = None             # Membership funcs. parameters
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def level(self):
        """ Get the optimization level """
        return self._level

    @level.setter
    def level(self, level):
        """ Set the optimization level """
        self._level = level

    @property
    def regressors(self):
        """ Get the coefficients of the regressors """
        return self._regressors.reshape(
            -1, 2 * len(self._regressors) // len(self._membership))

    @property
    def membership(self):
        """ Get the coefficients of the membership functions """
        return self._membership.reshape(-1, 2)
    #------------------------------------------------------------------------#

    #---------------------   Initialization Functions   ---------------------#
    def _beta0(self, inputs):
        """ Initial membership function parameters """
        bcu = np.empty((len(inputs), 2), float)
        for i, inp in enumerate(inputs):
            bcu[i, 0] = inp.mean()                  # Cluster's mean
            std = inp.std()                         # Cluster's std (set to
            bcu[i, 1] = std if std != 0 else 0.01   # 0.01 if null to avoid
        return bcu.ravel()                          # dividing by 0 in exp)

    def _matinfo_theta(self, values):
        """ Information Matrix of the local models """
        depth, pos = values.shape[1], 0
        phi = np.empty((len(values), depth*self._order), float)
        for j in range(1, self._order+1):           # Powered inputs
            phi[:, pos:pos+depth] = values**j
            pos += depth
        return phi

    def _matinfo(self, values, beta):
        """ Information Matrix of the models masked by their memberships """

        depth, mod, pos = values.shape[1], len(beta)//2, 0
        phi = np.empty((len(values), mod*depth*self._order), float)

        # All products between the local models and the memberships sum
        den = np.zeros(len(values), float)
        for bcu in beta.reshape(-1, 2):                     # Local MF params
            apt = np.exp(-0.5*((values-bcu[0])/bcu[1])**2)  # Local MF
            for j in range(1, self._order+1):               # Powered inputs
                phi[:, pos:pos+depth] = apt*(values**j)     # Local model
                pos += depth
            den += apt.sum(1)                               # MF summation

        # Denominator (sum of the memberships)
        return depth * phi / den.reshape(-1, 1)
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Functions   -----------------------#
    def _memberships(self, inputs, beta, *args):
        """ Gaussian-like membership function """
        # pylint: disable=unused-argument
        return np.exp(-0.5*((inputs-beta[0])/beta[1])**2).mean(1)

    def _estimate(self, values, beta, theta):
        """ Multi-Model estimate (local estimate + membership function) """
        depth, pos = values.shape[-1], 0
        acc = np.zeros(len(values), float)
        den = np.zeros(len(values), float)
        for bcu in beta.reshape(-1, 2):                     # Local MF params
            apt = np.exp(-0.5*((values-bcu[0])/bcu[1])**2)  # Local MF
            for j in range(1, self._order+1):               # Powered inputs
                tcu = theta[pos:pos+depth]                  # Loc model params
                acc += (apt*tcu*(values**j)).sum(1)         # Local model
                pos += depth
            den += apt.sum(1)                               # MF summation
        return depth * acc / den

    def predict(self, inputs):
        """ Multi-Model estimate (local estimate + membership function)

        Parameters
        ----------
        inputs : np.ndarray
            The past data, typically sorted in time order.

        Returns
        -------
        pred : np.ndarray
            The estimated data.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        >>> def function(vals):
        ...     ''' Function to interpolate '''    
        ...     return vals**3 - 5.*vals**2 - 10.*vals

        >>> def matreg(vals, depth, step):
        ...     ''' Build the regression matrix '''
        ...     inps_glb = np.empty((len(vals)-depth-step, depth), float)
        ...     for i in range(depth):
        ...         inps_glb[:, i] = vals[i:-depth-step+i]
        ...     outs_glb = vals[depth+step:]
        ...     return inps_glb, outs_glb

        # Generate dummy data
        >>> values = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

        # Split the dataset into K subsets
        >>> K = 3
        >>> N = len(inps_glb) // K
        >>> inps_loc, outs_loc = [], []
        >>> for i in range(K):
        ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
        ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

        # Instantiate & train the multimodel
        >>> model = MultiModelPast(order=3, level='local_beta', stop=100)
        >>> model.fit(inps_loc, outs_loc)

        # Evaluate the multimodel's accuracy
        >>> cost = 0.
        >>> for inp, out in zip(inps_loc, outs_loc):
        ...     cost += opt.mse(out, model.predict(inp))
        >>> print(f"%.5e" %cost)
        3.05279e-04
        """
        if np.ndim(inputs) == 1:
            inputs = np.reshape(inputs, (1, -1))
        return self._estimate(inputs, self._membership, self._regressors)
    #------------------------------------------------------------------------#

    #-------------------   Local Modeling & Membership   --------------------#
    def _model_local(self, inputs, outputs):
        """ Local Modeling and Membership (cf. `fit` method) """

        # Local models (one per regression matrix)
        tcu = np.empty((len(inputs), self._order*inputs[0].shape[1]), float)
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            tcu[i] = opt.least_squares(self._matinfo_theta(inp), out)

        # Initial membership parameters
        bcu = self._beta0(inputs).reshape(-1, 2)

#        # Local memberships (one per regression matrix)
#        for i, (inp, out) in enumerate(zip(inputs, outputs)):
#            bcu[i] = opt.gradient_descent_2nd(
#                inp, out, bcu[i], None,
#                festim=self._memberships, fcost=cost_func, method='gn')

        self._regressors, self._membership = tcu.ravel(), bcu.ravel()
    #------------------------------------------------------------------------#

    #----------------   Local Modeling & Global Membership   ----------------#
    def _model_theta(self, inputs, outputs):
        """ Local Modeling and Global Membership (cf. `fit` method) """

        # Local models (one per regression matrix)
        tcu = np.empty((len(inputs), self._order*inputs[0].shape[1]), float)
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            tcu[i] = opt.least_squares(self._matinfo_theta(inp), out)
        tcu = tcu.ravel()

        # Initial membership parameters
        bcu = self._beta0(inputs)

        # Global memberships (on the whole regression matrix)
        inp, out = np.vstack(inputs), np.hstack(outputs)
        tmp = opt.gradient_descent_2nd(
            inp, out, bcu, tcu,
            festim=self._estimate, fcost=cost_func, method='gn')
        if (cost_func(out, self._estimate(inp, tmp, tcu))
            < cost_func(out, self._estimate(inp, bcu, tcu))):
            bcu = tmp

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #----------------   Global Modeling & Local Membership   ----------------#
    def _model_beta(self, inputs, outputs):
        """ Global Modeling and Local Membership (cf. `fit` method) """

        # Initial membership parameters
        bcu = self._beta0(inputs).reshape(-1, 2)

#        # Local memberships (one per regression matrix)
#        for i, (inp, out) in enumerate(zip(inputs, outputs)):
#            bcu[i] = opt.gradient_descent_2nd(
#                inp, out, bcu[i], None,
#                festim=self._memberships, fcost=cost_func, method='gn')
        bcu = bcu.ravel()

        # Global models (on the whole regression matrix)
        tcu = opt.least_squares(
            self._matinfo(np.vstack(inputs), bcu), np.hstack(outputs))

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-------------------   Global Modeling & Membership   -------------------#
    def _model_global(self, inputs, outputs):
        """ Global Modeling and Membership (cf. `fit` method) """

        # Rebuild the whole regression matrices (inputs & outputs)
        inp, out = np.vstack(inputs), np.hstack(outputs)

        # Initial membership parameters
        bcu = self._beta0(inputs)

        # Initial local modeling
        tcu = opt.least_squares(self._matinfo(inp, bcu), out)
        ccu = cost_func(out, self._estimate(inp, bcu, tcu))

        # Multi-modeling the data
        for _ in range(self._stop):

            # Global membership (on the whole regression matrix)
            tmp = opt.gradient_descent_2nd(
                inp, out, bcu, tcu,
                festim=self._estimate, fcost=cost_func, method='gn')
            cst = cost_func(out, self._estimate(inp, tmp, tcu))
            if cst < ccu:                           # If smaller error,
                bcu, ccu = tmp, cst                 # update MF's params
            else:                                   # and current error
                break

            # Global modeling (on the whole regression matrix)
            tmp = opt.least_squares(self._matinfo(inp, bcu), out)
            cst = cost_func(out, self._estimate(inp, bcu, tmp))
            if cst < ccu:                           # If smaller error,
                tcu, ccu = tmp, cst                 # update models' params
            else:                                   # and current error
                break

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, inputs, outputs):
        """ Select the modeling function and train the multi-model

        Parameters
        ----------
        inputs : list of 2D np.ndarrays
            The information matrix of every cluster (one per one): any
            row of a matrix is assumed to be the vector inputs for the
            training/modeling multi-model.
        outputs : list of 1D np.ndarrays
            The outputs associated with the information matrices: the
            outputs' vector at cell i corresponds to the inputs' infor-
            mation matrix at cell i. Any vector must be the same length
            as the corresponding information matrix.

        Returns
        -------
        None : directly set the `regressors` and `membership` attributes.

        Examples
        --------
        >>> import numpy as np
        >>> from modeling.optimization import cost

        >>> def function(vals):
        ...     ''' Function to interpolate '''    
        ...     return vals**3 - 5.*vals**2 - 10.*vals

        >>> def matreg(vals, depth, step):
        ...     ''' Build the regression matrix '''
        ...     inps_glb = np.empty((len(vals)-depth-step, depth), float)
        ...     for i in range(depth):
        ...         inps_glb[:, i] = vals[i:-depth-step+i]
        ...     outs_glb = vals[depth+step:]
        ...     return inps_glb, outs_glb

        # Generate dummy data
        >>> values = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

        # Split the dataset into K subsets
        >>> K = 3
        >>> N = len(inps_glb) // K
        >>> inps_loc, outs_loc = [], []
        >>> for i in range(K):
        ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
        ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

        # Instantiate & train the multimodel
        >>> model = MultiModelPast(order=3, level='local_beta', stop=100)
        >>> model.fit(inps_loc, outs_loc)
        >>> print(model.regressors)
        [[ 0.0930415   0.08124261  0.06663715  0.14981489  0.11712187  0.08840886
           0.00798625 -0.01844478 -0.03121083]
         [ 0.24316025  0.32813615  0.41913222  0.02319519 -0.00060374 -0.02840735
          -0.03979227 -0.00144675  0.04413742]
         [ 0.30621026  0.31744345  0.32905996 -0.00346391 -0.00634599 -0.00902045
          -0.00098606  0.01606458 -0.01635465]]
        """
        if self._level == 'local':              # Both theta and beta local
            self._model_local(inputs, outputs)

        elif self._level == 'local_theta':      # Theta local and beta global
            self._model_theta(inputs, outputs)

        elif self._level == 'local_beta':       # Theta global and beta local
            self._model_beta(inputs, outputs)

        elif self._level == 'global':           # Both theta and beta global
            self._model_global(inputs, outputs)
    #------------------------------------------------------------------------#

##############################################################################
