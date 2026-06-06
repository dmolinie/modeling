""" Radial Basis Function (RBF) Networks

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2024
Last revised: April 2026

License: GPLv3
"""

__all__ = ['STOP', 'RBFNet']

import numpy as np

import modeling.optimization._optimizers as opt
from ._kernels import get_ker_func

# Max number of iterations
STOP = 100

# Default cost function
cost_func = opt.mse


##############################################################################
##                               RBF Networks                               ##
##############################################################################

class RBFNet():
    """ Radial Basis Function (RBF) Networks

    Instantiate a single-layer network in which every neuron is an RBF.
    All the neurons share the same RBF kernel, and the synaptic weights
    are trained using the OLS. If activated, the centers of the RBFs are
    optimized by a 2nd order Gradient Descent.

    By denoting `f_i` the RBF associated with the i-th neuron, `c_i` its
    center and `w_i` its weighting coefficient, the RBF Network output y
    to an input x is given by:

        y(x) = sum_i w_i * f_i(||x - c_i||)

    where `||.||` is the module (absolute value) operator.

    Constructor
    -----------
    __init__(kernel, centers)

    Attributes
    ----------
    kernel : str, getter & setter
        The RBF kernel in use.
    centers : np.ndarray, getter & setter
        The centers of the RBFs.
    weights : np.ndarray, getter only
        The synaptic weights of the Network.

    Methods
    -------
    fit(inputs, outputs, fcost=None, step=0.1, coef=0., method='LM', local=True)
        Train the RBF Network.
    predict(inputs)
        Output the predicted values of the RBF Network to the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> import modeling.optimization as opt

    # Generate dummy data
    >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

    # Instantiate the RBF Network
    >>> net = rbf.RBFNet('linear', np.linspace(0, 100, 5, dtype=float))

    # Train the network
    >>> net.fit(inps, inps, local='both')
    >>> print(opt.mse(inps, net.predict(inps)))
    3.975989894735239e-26

    # Column array (1xN)
    >>> vec = inps.reshape(-1, 1)
    >>> net.fit(vec, inps)
    >>> print(opt.mse(inps, net.predict(vec)))
    3.975989894735239e-26

    # Multi-column array (MxN)
    >>> vec = np.hstack((inps.reshape(-1, 1), inps.reshape(-1, 1)))
    >>> net.fit(vec, inps)
    >>> print(opt.mse(inps, net.predict(vec)))
    1.0807310883548893e-07

    # Dummy examples on array shapes
    >>> a = np.arange(10)
    >>> net.fit(a, a)
    >>> print(opt.mse(a, net.predict(a)))
    0.00019063857510143704

    >>> b = a.reshape(-1, 1)
    >>> net.fit(b, a)
    >>> print(opt.mse(a, net.predict(b)))
    0.00019063857510143704

    >>> c = np.hstack((a.reshape(-1, 1), a.reshape(-1, 1)))
    >>> net.fit(c, a)
    >>> print(opt.mse(a, net.predict(c)))
    4.796651703711746e-05
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, kernel, centers):
        """ Instantiate an RBFNet object (constructor)

        Parameters
        ----------
        kernel : str
            The type of RBFs to be used. Possibilities:
              - 'linear': y(r) = -r
              - 'thin_plate_spline': y(r) = r² * log(r)
              - 'cubic': y(r) = -r³
              - 'quintic': y(r) = -r⁵
              - 'multiquadric': y(r) = -√(1+r²)
              - 'inverse_multiquadric': y(r) = 1/√(1+r²)
              - 'inverse_quadratic': y(r) = 1/(1+r²)
              - 'gaussian': y(r) = -exp(-r²)
             See the `get_ker_func` function for details.
                :Default: linear
        centers : tuple, list or np.ndarray of floats
            The centers C = {c_i} of the RBFs: r(x) = ||x - c_i||.
            Also provides the number of neurons for the network, as
            of the number of centers.

        Examples
        --------
        # RBF Network with a linear kernel and 5 functions
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))

        # RBF Network with a gaussian kernel and 10 functions
        >>> net = RBFNet('gaussian', np.linspace(0, 100, 10, dtype=float))
        """

        # Get the kernel function
        self._fker = get_ker_func(kernel)
        self._ker = kernel

        # Instantiate the RBF centers and weights
        self._centers = centers                 # Centers of the RBFs
        self._weights = None                    # Network's RBFs coefficients
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def kernel(self):
        """ Get the mathematical kernel in use """
        return self._ker

    @kernel.setter
    def kernel(self, kernel):
        """ Set the mathematical kernel to use """
        self._fker = get_ker_func(kernel)
        self._ker = kernel

    @property
    def centers(self):
        """ Get the centers of the RBFs """
        return self._centers

    @centers.setter
    def centers(self, centers):
        """ Set the centers of the RBFs """
        self._centers = centers

    @property
    def weights(self):
        """ Get the weighting coefficients of the Network's RBFs """
        return self._weights
    #------------------------------------------------------------------------#

    #------------------------   Train the Weights   -------------------------#
    def _train_weights(self, inputs, outputs):
        """ Train the synaptic weights of the RBF Network

        Take the input values and the corresponding outputs, build the
        information matrix, and train the RBF Network synaptic weights
        using the OLS. If the inputs are 1D, pass them to each of the
        RBF neurons of the Network, and build the information matrix as
        the outputs of one distinct RBF for every column of the matrix;
        if the inputs are multidimensional, treat every dimension as a
        distinct input vector, build the corresponding matrix for each
        of them, and final concatenate them all into one single matrix,
        which is eventually used by the OLS.

        Parameters
        ----------
        See method `fit`.

        Returns
        -------
        weights : np.ndarray of floats
            The synaptic weights of the Network optimized by the OLS.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

        # Instantiate the RBF Network and train its weights
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
        >>> net._weights = net._train_weights(inps, inps)
        """

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            network = np.empty(
                (len(inputs), np.shape(inputs)[1]*len(self._centers)), float)

            # Build the information matrix for every col of the input matrix
            cpt = 0
            for i in range(np.shape(inputs)[1]):
                inps = inputs[:, i]             # Current column
                for c in self._centers:         # Pass every col to every RBF
                    network[:, cpt] = self._fker(np.abs(inps-c))    # RBF
                    cpt += 1

        # Case the input data are a 1D vector
        else:
            network = np.empty((len(inputs), len(self._centers)), float)

            # Pass the inputs to every RBF of the network
            for i, c in enumerate(self._centers):
                network[:, i] = self._fker(np.abs(inputs-c))        # RBF

        # Optimize the synaptic weights
        return opt.least_squares(network, outputs)
    #------------------------------------------------------------------------#

    #------------------------   Train the Centers   -------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def _train_centers(self,
        inputs, outputs, fcost=None, step=0.1, coef=0., method='LM'):
        """ Train the RBF's centers of the Network

        Take the input values and corresponding outputs, and optimize
        the centers of the RBFs using a 2nd order Gradient Descent. The
        estimation function is the method `_summation`.
        See `gradient_descent_2nd` from `optimization` module for details.

        Parameters
        ----------
        See method `fit`.

        Returns
        -------
        weights : np.ndarray of floats
            The synaptic weights of the Network optimized by the OLS.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        # Generate dummy data
        >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

        # Instantiate the RBF Network and train both centers & weights
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
        >>> net._weights = net._train_weights(inps, inps)
        >>> net._centers = net._train_centers(inps, inps, opt.mse)
        """
        return opt.gradient_descent_2nd(
            inputs, outputs, self._centers, self._weights,
            step, coef, self._summation, fcost, method)
    #------------------------------------------------------------------------#

    #------------------------   Train RBF Network   -------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def fit(self,
        inputs, outputs, fcost=None, step=0.1, coef=0., method='LM', local=True):
        """ Train the RBF Network

        Take the input values and corresponding outputs, and train the
        Network's synaptic weights using the OLS. If `local` is set to
        `False`, improve the training by iteratively optimizing the syn-
        aptic weights (OLS) and the RBF's center (2nd order GD).

        N.B.: the arguments `fcost`, `step`, `coef` and `method` are not
             used if `local` is set to `True`, since they are only used
             by the 2nd order GD when improving the RBF's centers.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to train the RBF Network with. If 1D array, 
            build the information matrix by passing the input vector to
            any RBF of the network. If ND array, build the matrix for
            every dimension of the input matrix, concatenate them all,
            and pass it to the OLS for optimization.
        outputs : list, tuple of np.ndarray of floats
            The objective values for the training.
        [OPT] fcost : reference to a function
            The cost function. Default to the Mean Squared Error.
                :Default: None (Mean Squared Error)
        [OPT] step : float
            The step for the 2nd order gradient descent.
                :Default: 1.
        [OPT] coef : float
            The regularization coefficient; used only with LM.
                :Default: 0. (no reg. // LM <--> GN)
        [OPT] method : str
            The method to be used; possibilities (no case-sensitive):
              - 'Newton': Newton method
              - 'Gauss' : Gauss-Newton method
              - 'LM'    : Levenberg-Marquardt method
            Any other entry ignored and default value used instead.
                :Default: 'LM'
        [OPT] local : bool
            Only train the synaptic weights (True) or both weights
            and RBF's centers in an iterative fashion (False).
                :Default: True (synaptic weights only)

        Returns
        -------
        None : directly update the `weights` attribute.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

        # Instantiate the RBF Network
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))

        # Train the RBF Network's synaptic weights
        >>> net.fit(inps, inps, local='weights')

        # Train the RBF Network's synaptic weights & RBF centers
        >>> net.fit(inps, inps, local='both')
        """

        # Initial training of the synaptic weights
        self._weights = self._train_weights(inputs, outputs)

        # Train the RBF's centers and synaptic weights iteratively
        if not local:

            # Use the default cost function if none is provided
            if fcost is None:
                fcost = cost_func
            cbf, ccu = np.inf, fcost(outputs, self.predict(inputs))

            # Optimize the parameters until no further improvement observed
            # or after STOP (global variable) iterations at maximum
            cpt = 0
            while ccu < cbf and cpt < STOP:
                cpt += 1

                # Train the RBF's centers
                centers = self._train_centers(
                    inputs, outputs, fcost, step, coef, method)
                caf = fcost(
                    outputs, self._summation(inputs, centers, self._weights))

                # Check if estimates improved by updating centers
                if caf < ccu:
                    cbf, ccu = ccu, caf            # Update the errors (costs)
                    self._centers = centers        # Save the new centers

                # Train the Network synaptic weights
                weights = self._train_weights(inputs, outputs)
                caf = fcost(
                    outputs, self._summation(inputs, self._centers, weights))

                # Check if estimates improved by updating weights
                if caf < ccu:
                    cbf, ccu = ccu, caf            # Update the errors (costs)
                    self._weights = weights        # Save the new weights

                # Compute the errors with the new parameters
                cbf, ccu = ccu, fcost(outputs, self.predict(inputs))
    #------------------------------------------------------------------------#

    #-------------------------   Output Function   --------------------------#
    def _summation(self, inputs, centers, weights):
        """ Provide the output of the RBF Network to a set of inputs

        Parametric version of the method `predict`, which is used for
        the optimizing of the RBF's centers in the methods `fit` and
        `_train_centers`. This method is required to allow to pass the
        parameter vectors `weights` and `centers` as `theta` and `beta`,
        respectively, in the 2nd order GD optimization algorithms.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to train the RBF Network.
        centers : np.ndarray of floats
            The RBF's centers.
        weights : np.ndarray of floats
            The RBF Network's synaptic weights.

        Returns
        -------
        outputs : np.ndarray of the same dimension as `inputs`
            The predicted estimates to the input values.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        # Generate dummy data
        >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

        # Instantiate the RBF Network
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))

        # Train the RBF Network's synaptic weights & RBF centers
        >>> net.fit(inps, inps, local='both')
        >>> pred = net._summation(inps, net.centers, net.weights)
        >>> print(opt.mse(inps, pred))
        2.0084190091825144e-26
        """

        outputs = np.zeros(len(inputs), dtype=float)

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            cpt = 0
            # Pass every col of the input matrix to every RBF of the Network
            for i in range(np.shape(inputs)[1]):
                inps = inputs[:, i]
                for c in centers:
                    outputs += weights[cpt] * self._fker(np.abs(inps-c))
                    cpt += 1

        # Case the input data are a 1D vector
        else:
            # Pass the inputs to every RBF of the network
            for i, c in enumerate(centers):
                outputs += weights[i] * self._fker(np.abs(inputs-c))

        return outputs
    #------------------------------------------------------------------------#

    #-------------------------   Predict Outputs   --------------------------#
    def predict(self, inputs):
        """ Predict the values of the RBF Network to inputs

        Take a set of inputs, and pass them to the different RBFs of the
        Network before summing their weighted respective outputs.
        Note: RBF Network must be trained using the `fit` method prior.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to be used for prediction. If ND array,
            pass every dim. to the Network before summing them all.

        Returns
        -------
        pred : np.ndarray of the same dimension as `inputs`
            The predicted estimates to the input values.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        # Generate dummy data
        >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

        # Instantiate the RBF Network
        >>> net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))

        # Train the RBF Network's synaptic weights & RBF centers
        >>> net.fit(inps, inps, local='both')
        >>> print(opt.mse(inps, net.predict(inps)))
        4.3544624246272465e-26
        """
        if self._weights is None:
            raise AssertionError("RBF Network not trained, run `fit` method prior")
        return self._summation(inputs, self._centers, self._weights)
    #------------------------------------------------------------------------#

##############################################################################
