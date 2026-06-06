""" Gating Networks class

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2024
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['GatingNetworks']

import numpy as np

import modeling.optimization._optimizers as opt


##############################################################################
##                             Gating Networks                              ##
##############################################################################

class GatingNetworks():
    r""" Gating Networks Multi-Model

    Take a set of regression matrices and the corresponding objective
    values as a set of vectors of outputs and train the Gating Network.
    Any regression matrix is composed of N vectors, each composed of L
    past values. The objective values are vectors of size N, in which
    the value at line n corresponds to the regression matrix's vector
    at line n. The temporal gap between the most recent value in the
    regression vector and the objective value is the prediction step.

    By denoting `val` a regression vector, a Gating Network is defined
    as a linear combination of the estimates outputted by any of the K
    local models, each multiplied by a static weighting coefficient:
        ŷ(val, theta) = sum_k^K w_k(val) * f_k(val, theta_k)

    where `theta` is the set of the K vectors `theta_k`, each being the
    parameter vector of the local model of the k-th regression matrix.
    These parameter vectors are obtained by training the local models on
    their respective regression matrices.
    
    The weights `w_k` are static coefficients obtained by searching for
    the best values minimizing the approximation error between the esti-
    mation function ŷ and the objective outputs. This is done by optimi-
    zation using the Ordinary Least Squares (ŷ being linear in w_k).

    Important: the functions `f` can be obtained by using any regression
    model, e.g. Decision Trees, Random Forests, MLPs, RBF Networks, etc.
    Notice that any regressor must feature a `fit` method to train it,
    and a `predict` method to estimate the outputs to a set of inputs;
    as such, these methods must be implemented in the regressor's class.
      Ex:
        reg = model(**params)           # Instantiate the regressor
        reg.fit(inputs, outputs)        # Train the regressor
        outs = reg.predict(inputs)      # Use the regressor for prediction

    Constructor
    -----------
    __init__(model, params)

    Attributes
    ----------
    model : class object, getter & setter
        The (local) regression model.
    params : dict, getter & setter
        The parameters to be used by the regressor.
    regressors : np.ndarray, getter only
        The regressors' parameter vectors.
    weights : np.ndarray, getter only
        The weighting coefficients of the local models.

    Methods
    -------
    fit(inputs, outputs)
        Train the Gating Network.
    predict(inputs)
        Use the Gating Network for prediction.

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
    >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

    # Split database into K sub-datasets
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

    #----- Select the interpolator (local model) -----#
    # Local project generic interpolator
    >>> from modeling.interpolators import Interpolator
    >>> method = Interpolator
    >>> params = {'order': 2, 'interpolator': 'polynomial'}

    ## Local project past-data interpolator 
    >>> from modeling.interpolators import InterpolatorPast
    >>> #method = InterpolatorPast
    >>> #params = {'order': 2}

    ## Local project RBF Networks
    >>> #from modeling.rbf_net import *
    >>> #method = RBFNet
    >>> #params = {'kernel': 'linear',
    ... #          'centers': np.linspace(0, 1.5, 10, dtype=float)}

    #--- Scikit-Learn models

    ## Decision Trees
    >>> #from sklearn.tree import DecisionTreeRegressor
    >>> #method = DecisionTreeRegressor
    >>> #params = {'max_depth': 10}

    ## Random Forests
    >>> #from sklearn.ensemble import RandomForestRegressor
    >>> #method = RandomForestRegressor
    >>> #params = {'n_estimators': 20, 'max_depth': 8}

    ## Standard MLP
    >>> #from sklearn.neural_network import MLPRegressor
    >>> #method = MLPRegressor
    >>> #params = {'hidden_layer_sizes': 250, 'activation': 'logistic',
    ... #          'solver': 'adam', 'max_iter': 1000, 'verbose': False}
    #-------------------------------------------------#

    # Instantiate & train the gating network
    >>> model = GatingNetworks(method, params)
    >>> model.fit(mat_inps_trn, mat_outs_trn)

    # Evaluate the gating network's accuracy
    >>> cost = 0.
    >>> for inp, out in zip(mat_inps_trn, mat_outs_trn):
    ...     cost += opt.mse(out, model.predict(inp))
    >>> print("Training", f"%.5e" %cost)
    Training 8.98423e-04

    >>> cost = 0.
    >>> for inp, out in zip(mat_inps_gen, mat_outs_gen):
    ...     cost += opt.mse(out, model.predict(inp))
    >>> print("Testing", f"%.5e" %cost)
    Testing 1.95735e-03
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, model, params):
        """ Instantiate a GatingNetworks object (constructor)

        Parameters
        ----------
        model : class object
            The regression model to use.
                Ex: `Interpolator` from the `interpolators` module.
        params : dict
            The parameters of the regression `model` in use.
                Ex: {'order': 2, 'interpolator': 'polynomial'}
                    (for the interpolators.Interpolator regression model).

        Examples
        --------
        #--- Locally-implemented models
        >>> from modeling.interpolators import Interpolator

        # Use a simple interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate the Gating Networks with local Interpolators
        >>> gnets = GatingNetworks(method, params)

        #--- Scikit-Learn models
        >>> from sklearn.tree import DecisionTreeRegressor

        # Use simple Decision Trees
        >>> method = DecisionTreeRegressor
        >>> params = {'max_depth': 10}

        # Instantiate the Gating Networks with local Decision Trees
        >>> gnets = GatingNetworks(method, params)
        """

        if not callable(model):
            raise TypeError("Invalid type for `model`, the model to use "
                + f" must be a callable (received {type(model)})")

        self._model = model                     # (Local) Regression model
        self._params = params                   # Parameters of the model
        self._regressors = None                 # Local models coefficients
        self._weights = None                    # Weighting coefficients
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def model(self):
        """ Get the local model in use """
        return self._model

    @model.setter
    def model(self, model):
        """ Set the local model to use """
        if not callable(model):
            raise TypeError(f"Invalid type, need callable (received {type(model)})")
        self._model = model

    @property
    def params(self):
        """ Get the local model's parameters in use """
        return self._params

    @params.setter
    def params(self, params):
        """ Set the local model's parameters to use """
        self._params = params

    @property
    def regressors(self):
        """ Get the set of the local models (regressors) """
        return self._regressors

    @property
    def weights(self):
        """ Get the weighting coefficients of the local models """
        return self._weights
    #------------------------------------------------------------------------#

    #------------------------   Modeling Functions   ------------------------#
    def _local_models(self, inputs, outputs):
        """ Train the regressors on the given set of inputs & outputs """
        regressors = []
        for inps, outs in zip(inputs, outputs):       # For every model matrix
            regressor = self._model(**self._params)   # Instantiate regressor
            regressor.fit(inps, outs)                 # Train the regressor
            regressors.append(regressor)
        return regressors

    def _local_matreg(self, inputs):
        """ Build the regression matrix as the vertical concatenation of
            the estimates of every local model """
        mat_trn = np.empty((len(inputs), len(self._regressors)), float)
        for k, model in enumerate(self._regressors):  # Estimate the outputs
            mat_trn[:, k] = model.predict(inputs)     # using every loc. model
        return mat_trn                                # (one per column)
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Function   ------------------------#
    def predict(self, inputs):
        """ Use the Gating Network to estimate the outputs to `inputs`

        Parameters
        ----------
        inputs : list of 2D np.ndarrays
            The information matrix for which to predict the outputs
            (the matrix shape must be the same as those used when
             training the Gating Network using the `fit` method).

        Returns
        -------
        pred : np.ndarray
            The estimated outputs to the `inputs`.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        # Generate dummy data
        >>> values = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

        # Split database into K sub-datasets
        >>> K = 3
        >>> N = len(inps_glb) // K
        >>> inps_loc, outs_loc = [], []
        >>> for i in range(K):
        ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
        ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

        # Local project generic interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate & train the gating network
        >>> model = GatingNetworks(method, params)
        >>> model.fit(inps_loc, outs_loc)

        # Evaluate the gating network's accuracy
        >>> cost = 0.
        >>> for inp, out in zip(inps_loc, outs_loc):
        ...     cost += opt.mse(out, model.predict(inp))
        >>> print(f"%.5e" %cost)
        9.97769e-04
        """
        return np.sum(self._local_matreg(inputs) * self._weights, 1)
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, inputs, outputs):
        """ Train the Gating Network

        Parameters
        ----------
        inputs : list of 2D np.ndarrays
            The information matrix of every cluster (one per one):
            every row of a matrix is assumed to be the vector inputs
            for the training/modeling multi-model.
        outputs : list of 1D np.ndarrays
            The outputs associated with the information matrices:
            the outputs' vector at cell i corresponds to the inputs'
            information matrix at cell i. Every vector must be the
            same length as the corresponding information matrix.
            For a vector V of outputs, a value at cell j corresponds
            to the objective output of the estimation based on line j
            of the corresponding matrix list inputs.

        Returns
        -------
        None : directly set the `regressors` and `weights` attributes.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> values = function(np.linspace(-1, +1, 1000, float))

        # Build the global regression matrix
        >>> inps_glb, outs_glb = matreg(values, depth:=3, step:=0)

        # Split database into K sub-datasets
        >>> K = 3
        >>> N = len(inps_glb) // K
        >>> inps_loc, outs_loc = [], []
        >>> for i in range(K):
        ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
        ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

        # Local project generic interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate the gating network
        >>> model = GatingNetworks(method, params)

        # Train the gating network
        >>> model.fit(inps_loc, outs_loc)
        >>> print(model.weights)
        [0.22747836 0.45059337 0.32795688]
        """

        # Train the local models on the local matrices
        self._regressors = self._local_models(inputs, outputs)

        # Train the Gating Network by using the OLS to find the coefficients
        self._weights = opt.least_squares(
            self._local_matreg(np.vstack(inputs)), np.hstack(outputs))
    #------------------------------------------------------------------------#

##############################################################################
