""" Interpolation class

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: March 2024
Last revised: April 2026

License: GPLv3
"""

__all__ = ['Interpolator']

import numpy as np

from modeling.optimization import _optimizers as opt


##############################################################################
##                            Interpolation class                           ##
##############################################################################

class Interpolator():
    """ Polynomial/Sinusoidal Interpolation

    Instantiate an interpolator, which can either be a polynomial or a
    sinusoidal decomposition (similar to a partial Fourier series). The
    interpolation is linear in the coefficient vector, thus use the OLS
    to operate training. Provide tools to build the Information Matrix,
    operate training, and predict a time series.

    Constructor
    -----------
    __init__(order=1, interpolator='polynomial')

    Attributes
    ----------
    interpolator : str, getter & setter
        The interpolator to use.
    order : int, getter & setter
        The order of the interpolator.
    theta : np.ndarray, getter only
        The interpolator's coefficient vector.

    Methods
    -------
    matinfo(inputs)
        Build the Information Matrix.
    fit(inputs, outputs)
        Train the Interpolator.
    predict(inputs)
        Predict the values of the Interpolator to inputs.

    Examples
    --------
    >>> import numpy as np
    >>> import modeling.optimization as opt

    # Instantiate the interpolator
    >>> inter = Interpolator(5)
    >>> inter = Interpolator(5, 'poly')
    >>> inter = Interpolator(3, 'polynomial')
    >>> inter = Interpolator(5, 'sinus')
    >>> inter = Interpolator(3, 'polynomial')
    >>> inter.interpolator = 'poly'

    # Train the interpolator and use it for prediction
    >>> inps = np.arange(10)
    >>> mat = inter.matinfo(inps)
    >>> inter.fit(inps, inps)
    >>> print(inter.theta)
    [ 1.00000000e+00 -2.49245069e-14  1.30451205e-15]
    >>> inter.predict(inps)

    # Column array (1xN)
    >>> vec = inps.reshape(-1, 1)
    >>> mat = inter.matinfo(vec)
    >>> inter.fit(vec, inps)
    >>> print(opt.mse(inps, inter.predict(vec)))
    9.834382053111089e-26

    # Multi-column array (MxN)
    >>> vec = np.hstack((inps.reshape(-1, 1), inps.reshape(-1, 1)))
    >>> mat = inter.matinfo(vec)
    >>> inter.fit(vec, inps)
    >>> print(opt.mse(inps, inter.predict(vec)))
    0.0031552988340089183

    # Dummy examples on array shapes
    >>> a = np.arange(10)
    >>> inter.fit(a, a)
    >>> resa = inter.predict(a)
    >>> print(opt.mse(a, resa))
    9.834382053111089e-26

    >>> b = a.reshape(-1, 1)
    >>> inter.fit(b, a)
    >>> resb = inter.predict(b)
    >>> print(opt.mse(a, resb))
    9.834382053111089e-26

    >>> c = np.hstack((a.reshape(-1, 1), a.reshape(-1, 1)+3))
    >>> inter.fit(c, a)
    >>> resc = inter.predict(c)
    >>> print(opt.mse(a, resc))
    0.003776182404149145
    """

    #---------------------------   Constructor   ----------------------------#
    @staticmethod
    def __check_order(order):
        """ Check if order argument is positive """
        if order >= 1:
            return order
        raise ValueError(f"Wrong value `{order}` for `order`, "
            + "order must be strictly positive")

    def __check_inter(self, interpolator):
        """ Check is the interpolator is valid """
        if 'poly' in interpolator:  # Polynomial
            return {'inter': 'polynomial',
                    'matinfo': self._matinfo_polynomial,
                    'predict': self._predict_polynomial}
        if 'sinus' in interpolator: # Sinusoidal
            return {'inter': 'sinusoidal',
                    'matinfo': self._matinfo_sinusoidal,
                    'predict': self._predict_sinusoidal}
        raise ValueError(
            f"Unknown value {interpolator} for `interpolator`; options are:\n"
            + "\t{'polynomial' and 'sinusoidal'}")

    def __init__(self, order=1, interpolator='polynomial'):
        """ Instantiate an Interpolator object (constructor)

        Parameters
        ----------
        [OPT] order : int
            The order of the interpolator (maximal exponentiation).
                :Default: 1
        [OPT] interpolator : str
            The interpolator name, among: {'polynomial', 'sinusoidal'}.
                :Default: 'polynomial'

        Examples
        --------
        # Order 3 polynomial interpolator
        >>> interpol = Interpolator(3, 'polynomial')

        # Order 1 sinusoidal interpolator
        >>> interpol = Interpolator(1, 'sinusoidal')
        """
        self._order = self.__check_order(order)
        self._inter = self.__check_inter(interpolator.lower())
        self._theta = None
        self.__flag = True
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def interpolator(self):
        """ Get the interpolator's type """
        return self._inter['inter']

    @interpolator.setter
    def interpolator(self, interpolator):
        """ Set the interpolator's type """
        self.__flag = True
        self._inter = self.__check_inter(interpolator.lower())

    @property
    def order(self):
        """ Get the interpolator's order """
        return self._order

    @order.setter
    def order(self, order):
        """ Set the interpolator's order """
        self.__flag = True
        self._order = self.__check_order(order)

    @property
    def theta(self):
        """ Get the coefficient vector theta """
        return self._theta
    #------------------------------------------------------------------------#

    #-----------------------   Information Matrices   -----------------------#
    def _matinfo_polynomial(self, inputs):
        """ IM of the polynomial interpolator """

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            depth = np.shape(inputs)[1]
            mat = np.empty((len(inputs), depth*self._order), float)

            # Build the information matrix for every col of the input matrix
            pos = 0
            for i in range(depth):
                inps = inputs[:, i]                     # Current dim
                for j in range(1, self._order+1):
                    mat[:, pos] = inps**j               # Monomials
                    pos += 1

        # Case the input data are a 1D vector
        else:
            mat = np.empty((len(inputs), self._order), float)

            # Pass the inputs to the interpolator
            for j in range(self._order):
                mat[:, j] = inputs**(j+1)               # Monomials

        return mat

    def _matinfo_sinusoidal(self, inputs):
        """ IM of the sinusoidal interpolator """

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            depth = np.shape(inputs)[1]
            mat = np.empty((len(inputs), 2*depth*self._order), float)

            # Build the information matrix for every col of the input matrix
            pos = 0
            for i in range(depth):                      # Through every col
                inps = 2*np.pi*inputs[:, i]             # Trigo values
                for j in range(1, self._order+1):
                    mat[:, pos] = np.cos(j*inps)        # Cosine coef
                    mat[:, pos+1] = np.sin(j*inps)      # Sine coef
                    pos += 2

        # Case the input data are a 1D vector
        else:
            mat = np.empty((len(inputs), 2*self._order), float)

            # Pass the inputs to the interpolator
            pos = 0
            inps = 2*np.pi*inputs                       # Trigo values
            for j in range(1, self._order+1):
                mat[:, pos] = np.cos(j*inps)            # Cosine coef
                mat[:, pos+1] = np.sin(j*inps)          # Sine coef
                pos += 2

        return mat

    def matinfo(self, inputs):
        """ Build the Information Matrix

        Create and return the Information Matrix for the OLS. Following
        a linear interpolation (in the parameters), each column of the
        IM is a unique transform applied to any 1D feature. Two types
        are available: Polynomial and Sinusoidal.

        Polynomial: operate a polynomial interpolation:
                Phi = (inp_i^j)_{j in [[0, order]]}
            For instance (order = 2):
                      (1    i0    i0^2)
                Phi = (|     |     |  )
                      (1    iN    iN^2)

        Sinusoidal: mimic a partial Fourier Series decomposition:
            By posing inp = 2pi*inputs
                Phi = (cos(j*inp_i) sin(j*inp_i))_{j in [[0, order]]}
            For instance (order = 4):
                      (cos(t0)  sin(t0)  cos(2t0)   sin(2t0))
                Phi = (     |        |        |          |    )
                      (cos(tN)  sin(tN)  cos(2tN)   sin(2tN))

        Note that the maximal order is one lower than that specified in
        the constructor to account for the constant of the polynomial.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to build the Information Matrix. If ND
            array, build the matrix for every dimension of the input
            matrix and concatenate all of them.

        Returns
        -------
        mat : 2D np.ndarray of size order*len(inputs)
          The Information Matrix.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> inps = np.arange(10)

        # Build the Interpolator & its information matrix
        >>> inter = Interpolator(3, 'polynomial')
        >>> mat = inter.matinfo(inps)
        >>> print(mat)
        [[  0.   0.   0.]
         [  1.   1.   1.]
         [  2.   4.   8.]
         [  3.   9.  27.]
         [  4.  16.  64.]
         [  5.  25. 125.]
         [  6.  36. 216.]
         [  7.  49. 343.]
         [  8.  64. 512.]
         [  9.  81. 729.]]
        """
        return self._inter['matinfo'](inputs)
    #------------------------------------------------------------------------#

    #---------------------   Interpolator's Training   ----------------------#
    def fit(self, inputs, outputs):
        """ Train the Interpolator

        Take the inputs and the corresponding values to interpolate, and
        train the interpolator by finding the best coefficients with the
        OLS. The interpolator is the `interpolator` property.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to train the Interpolator. If 1D array,
            build the information matrix with the only input vector.
            If ND array, build the matrix for every dimension of the
            input matrix, concatenate them all and pass it to the OLS
            for optimization.
        outputs : list, tuple of np.ndarray of floats
            The objective values for the training.

        Returns
        -------
        None : directly set the parameter vector `theta`.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> inps = np.arange(10)

        # Build the Interpolator and train it
        >>> inter = Interpolator(3, 'polynomial')
        >>> inter.fit(inps, inps)
        >>> print(inter.theta)
        [ 1.00000000e+00 -2.49245069e-14  1.30451205e-15]
        """
        self.__flag = False                                    # Training done
        if self._order == 1:
            self._theta = opt.least_squares(inputs, outputs)
        else:
            self._theta = opt.least_squares(self.matinfo(inputs), outputs)
    #------------------------------------------------------------------------#

    #--------------------   Interpolator's Forecasting   --------------------#
    def _predict_polynomial(self, inputs):
        """ Polynomial Interpolation """

        outputs = np.zeros(len(inputs), float)

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            # Sum the estimates for every col of the input matrix
            pos = 0
            for i in range(np.shape(inputs)[1]):
                inps = inputs[:, i]
                for j, thi in enumerate(self._theta[pos:pos+self._order]):
                    outputs += thi*(inps**(j+1))        # Weighted monomials
                pos += self._order

        # Case the input data are a 1D vector
        else:
            # Sum the weighted powered input values
            for j, thi in enumerate(self._theta):
                outputs += thi*(inputs**(j+1))          # Weighted monomials

        return outputs

    def _predict_sinusoidal(self, inputs):
        """ Sinusoidal interpolation (~ partial Fourier series) """

        outputs = np.zeros(len(inputs), float)

        # Case the input data are an ND matrix
        if np.ndim(inputs) > 1:
            # Sum the estimates for every col of the input matrix
            pos = 0
            for i in range(np.shape(inputs)[1]):
                inps = 2*np.pi*inputs[:, i]
                thi = self._theta[pos:pos+2*self._order]
                for j, (th1, th2) in enumerate(zip(thi[0::2], thi[1::2]), 1):
                    outputs += th1*np.cos(j*inps) + th2*np.sin(j*inps) # Trigs
                pos += 2*self._order

        # Case the input data are a 1D vector
        else:
            # Sum the weighted powered input values
            inps = 2*np.pi*inputs
            for j, (th1, th2) in enumerate(
                zip(self._theta[0::2], self._theta[1::2]), 1):
                outputs += th1*np.cos(j*inps) + th2*np.sin(j*inps)     # Trigs

        return outputs

    def predict(self, inputs):
        """ Predict the values of the Interpolator to inputs

        Take a set of inputs, and pass them to the interpolator to get
        the corresponding predicted outputs.
        Note: Interpolator must be trained using the `fit` method first.

        Parameters
        ----------
        inputs : list, tuple or np.ndarray of floats
            The input values to be used for prediction. If ND array,
            pass every dim. to the Interpolator before summing them.

        Returns
        -------
        pred : np.ndarray of the same dimension as `inputs`
            The predicted estimates to the input values.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        # Generate dummy data
        >>> inps = np.arange(10)
        >>> vec = inps.reshape(-1, 1)

        # Build the Interpolator, train it and use it for prediction
        >>> inter = Interpolator(3, 'polynomial')
        >>> inter.fit(vec, inps)
        >>> print(opt.mse(inps, inter.predict(vec)))
        9.834382053111089e-26
        """
        self.__flag = False

        # Check if the order or interpolator have been modified
        if self._theta is None or self.__flag:
            raise AssertionError(
                "Interpolator not trained, or parameters changed; "
                + "please run `fit` method prior")

        # Predict the values to the inputs
        return self._inter['predict'](inputs)
    #------------------------------------------------------------------------#

##############################################################################
