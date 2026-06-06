""" Multi-Models class that wraps a set of models into a multi-model

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2024
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['MultiModel']

import numpy as np

import modeling.optimization._optimizers as opt

# Default cost function
cost_func = opt.mse


##############################################################################
##                      Membership-based Multi-Models                       ##
##############################################################################

class MultiModel():
    r""" Membership-based Multi-Model class

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
    
    The membership functions are Gaussian-like functions, defined as:
        rho_k(val, beta_k) = exp( -(val - beta_k[0])^2 / (2*beta_k[1]^2) )

    with `w_k` the weight of model `p` defined by:
        w_k(val, beta_k) = rho_k(val, beta_k) / (sum_k rho_k(val, beta_k))

    In practice, `beta` is flattened in the implementation for practical
    reasons, but the corresponding `membership` property is reshaped
    into K sets of (local) vectors, each being of size 2 (mean and std).

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
    __init__(model, params, level='local_beta', stop=100)

    Attributes
    ----------
    level : str, getter & setter
        The optimization level.
    model : class object, getter & setter
        The (local) regression models.
    params : dict, getter & setter
        The parameters to be used by the regressor.
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

    >>> for level in ('local', 'local_theta', 'local_beta', 'global'):
    ...     # Instantiate & train the multimodel
    ...     model = MultiModel(method, params, level)
    ...     model.fit(mat_inps_trn, mat_outs_trn)
    ...     # Evaluate the multimodel's accuracy
    ...     cost = 0.
    ...     for inp, out in zip(mat_inps_gen, mat_outs_gen):
    ...         cost += opt.mse(out, model.predict(inp))
    ...     print(level, '\t', f"%.5e" %cost)
    local 	        7.44479e-04
    local_theta 	7.44479e-04
    local_beta 	    1.08057e-03
    global 	        1.08057e-03
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, model, params, level='local_beta', stop=100):
        """ Instantiate a MultiModel object (constructor)

        Parameters
        ----------
        model : class object
            The regression model to use.
                Ex: `Interpolator` from the `interpolators` module.
        params : dict
            The parameters of the regression `model` in use.
                Ex: {'order': 2, 'interpolator': 'polynomial'}
                    (for the interpolators.Interpolator regression model).
        [OPT] level : str
            The modeling level: local, semi-global or global:
              - 'local':       both local models and memberships;
              - 'local_theta': local models and global memberships;
              - 'local_beta':  global models and local memberships;
              - 'global':      both global models and memberships.
            The last one should give the best results, but is the
            slowest; the 'local_beta' variant is generally a good
            trade-off between accuracy and execution speediness.
                :Default: 'local_beta'
        [OPT] stop : int
            Maximal number of iterations for the iterative methods.
                :Default: 100

        Examples
        --------
        #--- Locally-implemented models
        >>> from modeling.interpolators import Interpolator

        # Use a simple interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate the Multi-Model Networks with local Interpolators
        # with local optimization for both models and memberships
        >>> mmodel = MultiModel(method, params, 'local')

        #--- Scikit-Learn models
        >>> from sklearn.tree import DecisionTreeRegressor

        # Use simple Decision Trees
        >>> method = DecisionTreeRegressor
        >>> params = {'max_depth': 10}

        # Instantiate the Multi-Model with local Decision Trees
        # with local optimization for models and global for memberships
        >>> mmodel = MultiModel(method, params, 'local_beta')
        """

        if not callable(model):
            raise TypeError("Invalid type for `model`, the model to use "
                + f" must be a callable (received {type(model)})")

        self._stop = stop                       # Stopping criterion
        self._level = level                     # Level of the training
        self._model = model                     # (Local) Regression model
        self._params = params                   # Parameters of the model
        self._regressors = None                 # Local models coefficients
        self._membership = None                 # Membership funcs. parameters
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
    def membership(self):
        """ Get the membership functions' parameter vector """
        return self._membership.reshape(-1, 2)
    #------------------------------------------------------------------------#

    #------------------------   Modeling Functions   ------------------------#
    def _beta0(self, inputs):
        """ Initial membership function parameters """
        beta = np.empty((len(inputs), 2), float)
        for i, inp in enumerate(inputs):
            beta[i, 0] = inp.mean()                 # Cluster's mean
            std = inp.std()                         # Cluster's std (set to
            beta[i, 1] = std if std != 0 else 0.01  # 0.01 if null to avoid
        return beta                                 # dividing by 0 in exp)

    def _local_models(self, inputs, outputs):
        """ Train the regressor on the given inputs & outputs """
        regressor = self._model(**self._params)     # Instantiate regressor
        regressor.fit(inputs, outputs)              # Train the regressor
        return regressor

    def _global_models(self, inputs, outputs, beta):
        """ Train the regressor on the whole database """

        # Compute the denominator
        den = np.zeros(len(outputs), float)
        for bcu in beta.reshape(-1, 2):                         # MF's params
            apt = np.exp(-0.5*((outputs-bcu[0]))**2 / bcu[1])   # Current MF
            den += apt                                          # Denominator

        # Train the local models with masked objective values
        regressors = []
        for bcu in beta.reshape(-1, 2):
            apt = np.exp(-0.5*((outputs-bcu[0]))**2 / bcu[1]) / den
            regressors.append(self._local_models(inputs, apt*outputs))

        return regressors

    def _memberships(self, inputs, beta, *args):
        """ Gaussian-like membership function """
        # pylint: disable=unused-argument
        return np.exp(-0.5*((inputs-beta[0])/beta[1])**2).mean(1)
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Functions   -----------------------#
    def _predict_local(self, inputs, beta, models):
        """ Local prediction function (wrt the MFs)"""
        den = np.zeros(len(inputs), float)
        acc = np.zeros(len(inputs), float)
        for model, bcu in zip(models, beta.reshape(-1, 2)):
            apt = np.sum(np.exp(-0.5*((inputs-bcu[0]))**2 / bcu[1]), 1) # + Efficient ?
            acc += apt*model.predict(inputs)        # Masked estimates
            den += apt                              # Cumulated memberships
        return acc / den                            # Normalized estimates

    def _predict_global(self, inputs, models):
        """ Global prediction function (wrt the MFs) """
        acc = np.zeros(len(inputs), float)
        for model in models:
            acc += model.predict(inputs)            # Cumulated estimates
        return acc

    def predict(self, inputs):
        """ Multi-Model estimate (local estimate + membership function)

        Parameters
        ----------
        inputs : list of 2D np.ndarrays
            The information matrix for which to predict the outputs
            (the matrix shape must be the same as those used when
             training the Multi-Model using the `fit` method).

        Returns
        -------
        pred : np.ndarray
            The estimated outputs to the `inputs`.

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
        inps_loc, outs_loc = [], []
        >>> for i in range(K):
        ...     inps_loc.append(inps_glb[i*N:(i+1)*N])
        ...     outs_loc.append(outs_glb[i*N:(i+1)*N])

        # Local project generic interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate & train the multimodel
        >>> model = MultiModel(method, params, 'local_beta')
        >>> model.fit(inps_loc, outs_loc)

        # Evaluate the multimodel's accuracy
        >>> cost = 0.
        >>> for inp, out in zip(inps_loc, outs_loc):
        ...     cost += opt.mse(out, model.predict(inp))
        >>> print(f"%.5e" %cost)
        5.55278e-04
        """
        if self._level in ('local_beta', 'global'):
            return self._predict_global(inputs, self._regressors)
        return self._predict_local(inputs, self._membership, self._regressors)
    #------------------------------------------------------------------------#

    #-------------------   Local Modeling & Membership   --------------------#
    def _fit_local(self, inputs, outputs):
        """ Local Modeling and Membership (cf. `fit` method) """

        # Local models (one per regression matrix)
        regressors = []
        for inp, out in zip(inputs, outputs):
            regressors.append(self._local_models(inp, out))

        # Initial membership parameters
        bcu = self._beta0(inputs)

#        # Local memberships (one per regression matrix)
#        for i, (inp, out) in enumerate(zip(inputs, outputs)):
#            bcu[i] = opt.gradient_descent_2nd(
#                inp, out, bcu[i], None,
#                festim=self._memberships, fcost=cost_func, method='gn')

        self._regressors, self._membership = regressors, bcu.ravel()
    #------------------------------------------------------------------------#

    #----------------   Local Modeling & Global Membership   ----------------#
    def _fit_localmodels(self, inputs, outputs):
        """ Local Modeling and Global Membership (cf. `fit` method) """

        # Local models (one per regression matrix)
        regressors = []
        for inp, out in zip(inputs, outputs):
            regressors.append(self._local_models(inp, out))

        # Global memberships (on the whole regression matrix)
        bcu = self._beta0(inputs).ravel()

        # Global memberships (on the whole regression matrix)
        bcu = opt.gradient_descent_2nd(
            np.vstack(inputs), np.hstack(outputs), bcu, regressors,
            festim=self._predict_local, fcost=cost_func, method='gn')

        self._regressors, self._membership = regressors, bcu
    #------------------------------------------------------------------------#

    #----------------   Global Modeling & Local Membership   ----------------#
    def _fit_globalmodels(self, inputs, outputs):
        """ Global Modeling and Local Membership (cf. `fit` method) """

        # Initial membership parameters
        bcu = self._beta0(inputs)

#        # Local memberships (one per regression matrix)
#        for i, (inp, out) in enumerate(zip(inputs, outputs)):
#            bcu[i] = opt.gradient_descent_2nd(
#                inp, out, bcu[i], None,
#                festim=self._memberships, fcost=cost_func, method='gn')
        bcu = bcu.ravel()

        regressors = self._global_models(
            np.vstack(inputs), np.hstack(outputs), bcu)

        self._regressors, self._membership = regressors, bcu
    #------------------------------------------------------------------------#

    #-------------------   Global Modeling & Membership   -------------------#
    def _fit_global(self, inputs, outputs):
        """ Global Modeling and Membership (cf. `fit` method) """

        # Rebuild the whole regression matrices (inputs & outputs)
        inp, out = np.vstack(inputs), np.hstack(outputs)

        # Initial membership parameters
        bcu = self._beta0(inputs).ravel()

        # Initial local modeling
        regressors = self._global_models(inp, out, bcu)
        ccu = cost_func(out, self._predict_local(inp, bcu, regressors))

        # Multi-modeling the data
        for _ in range(self._stop):

            # Global membership (on the whole regression matrix)
            tmp = opt.gradient_descent_2nd(
                inp, out, bcu, regressors,
                festim=self._predict_local, fcost=cost_func, method='gn')
            cst = cost_func(out, self._predict_local(inp, tmp, regressors))
            if cst < ccu:                           # If smaller error
                bcu = tmp                           # Update the MFs' params
                ccu = cst                           # Update the error
            else:
                break

            # Global modeling (on the whole regression matrix)
            reg_tmp = self._global_models(inp, out, bcu)
            cst = cost_func(out, self._predict_local(inp, bcu, reg_tmp))
            if cst < ccu:                           # If smaller error
                regressors = reg_tmp                # Update the regressors
                ccu = cst                           # Update the error
            else:
                break

        self._regressors, self._membership = regressors, bcu
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, inputs, outputs):
        """ Select the modeling function and train the multi-model

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

        # Local project generic interpolator
        >>> method = Interpolator
        >>> params = {'order': 2, 'interpolator': 'polynomial'}

        # Instantiate the multimodel
        >>> model = MultiModel(method, params, 'local_beta')

        # Train the multimodel
        >>> model.fit(inps_loc, outs_loc)
        >>> print(model.membership)
        [[ 3.8904273   0.43686067]
         [-0.16430747  1.9093609 ]
         [-8.64214007  2.92648418]]
        """

        # Both theta and beta local
        if self._level == 'local':
            self._fit_local(inputs, outputs)

        # Theta local and beta global
        if self._level == 'local_theta':
            self._fit_localmodels(inputs, outputs)

        # Theta global and beta local
        if self._level == 'local_beta':
            self._fit_globalmodels(inputs, outputs)

        # Both theta and beta global
        if self._level == 'global':
            self._fit_global(inputs, outputs)
    #------------------------------------------------------------------------#

##############################################################################
