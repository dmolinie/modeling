""" Derivative tools

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2024
Last revised: April 2026

License: GPLv3
"""

__all__ = ['matinv', 'Derivative', 'Hessian']

import numpy as np


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

#---------------------------   Matrix Inversion   ---------------------------#
def matinv(mat):
    """ Invert a matrix possibly containing all-zeros rows/cols

    A matrix with an all-zeros row or column is non-invertible (det=0);
    thus, identify the all-zeros rows/cols, set its diagonal component
    to one so as to ignore the corresponding dimension while inverting
    the matrix, proceed to the inversion and reset to zero the diagonal
    component to undo the influence of the corresponding dimension.

    This is similar to the simple deletion of the all-zeros rows/cols,
    to the inversion of the reduced matrix and to its final extension.

    Parameters
    ----------
    mat : np.ndarray
        The matrix to be inverted.

    Returns
    -------
    matinv : np.ndarray
        The inverted matrix.

    Examples
    --------
    >>> import numpy as np

    >>> mat = np.random.random(9).reshape(3, 3)     # Dummy data
    >>> minv = matinv(mat)                          # Inverse matrix
    >>> (mat @ minv).round(3)                       # Check that M x MI = Id
    array([[ 1.,  0., -0.],
           [-0.,  1., -0.],
           [ 0.,  0.,  1.]])
    """
    mat = mat.copy()
    zeros = np.logical_or(np.around(mat.sum(axis=0)) == 0., # All-zeros cols
                          np.around(mat.sum(axis=1)) == 0.) # All-zeros rows
    mat[zeros, zeros] = 1.      # Replace the diagonal 0 by 1
    mat = np.linalg.inv(mat)    # Invert the matrix
    mat[zeros, zeros] = 0.      # Delete the diagonal 1
    return mat
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Derivative class                             ##
##############################################################################

class Derivative():
    """ Derivative & gradient class

    Provide tools to compute the partial derivative or gradient of an
    estimation or cost function. The derivatives are computed using the
    trapezoidal rule.

    Constructor
    -----------
    __init__(festim, fcost, beta=None)

    Constants
    ---------
    EPS = 1e-6
        Step for the derivatives.

    Attributes
    ----------
    festim : function reference, getter & setter
        The estimation function.
    fcost : function reference, getter & setter
        The cost function.
    beta : np.ndarray, getter & setter
        The second parameter vector.

    Methods
    -------
    derivative(inps, theta, partial, values=None)
        Compute the 1st order partial p-th derivative.
    derivative_2d(inps, theta, partial, values=None)
        Compute the 2nd order 2D partial derivative.
    gradient(inps, theta, values=None)
        Compute the gradient of an estimation or cost function.
    gradient_quad(inps, theta, values)
        Compute the gradient of the quadratic cost function.

    Examples
    --------
    >>> import numpy as np

    # Estimation function
    >>> def festim(inps, theta, *args):
    ...     return inps*(2*theta[0]**2 + theta[1]**3 + theta[0] * theta[1]**2)

    >>> def derx(inps, theta, *args):
    ...     return inps*(4*theta[0] + theta[1]**2)

    >>> def dery(inps, theta, *args):
    ...     return inps*(3*theta[1]**2 + 2*theta[0]*theta[1])

    >>> def derxx(inps, theta, *args):
    ...     return 4*inps

    >>> def deryy(inps, theta, *args):
    ...     return inps*(6*theta[1] + 2*theta[0])

    >>> def derxy(inps, theta, *args):
    ...     return 2*inps*theta[1]

    # Cost function
    >>> def fcost(values, estims):
    ...     return np.sum(values**2 + estims**2 + values*estims)

    >>> def derjx(values, inps, theta):
    ...     return np.sum((2*festim(inps, theta)+values) * derx(inps, theta), 0)

    >>> def derjy(values, inps, theta):
    ...     return np.sum((2*festim(inps, theta)+values) * dery(inps, theta), 0)

    >>> def derjxx(values, inps, theta):
    ...     return np.sum(2*derx(inps, theta) * derx(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * derxx(inps, theta), 0)

    >>> def derjyy(values, inps, theta):
    ...     return np.sum(2*dery(inps, theta) * dery(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * deryy(inps, theta), 0)

    >>> def derjxy(values, inps, theta):
    ...     return np.sum(2*dery(inps, theta) * derx(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * derxy(inps, theta), 0)


    # Generate dummy data
    >>> inps = np.arange(10, dtype=float)
    >>> theta = (1., 2.)

    # Derivative object
    >>> der = Derivative(festim, fcost, None)


    #--- Estimates' derivatives

    # 1st order derivatives
    >>> derx_th = derx(inps, theta)
    >>> derx_ex = der.derivative(inps, theta, 0)
    >>> dery_th = dery(inps, theta)
    >>> dery_ex = der.derivative(inps, theta, 1)

    >>> print(sum(derx_th - derx_ex))
    3.3168703339470085e-08
    >>> print(sum(dery_th - dery_ex))
    5.923197932133917e-08

    # 2nd order derivatives
    >>> derxx_th = derxx(inps, theta)
    >>> derxx_ex = der.derivative_2d(inps, theta, (0, 0))
    >>> deryy_th = deryy(inps, theta)
    >>> deryy_ex = der.derivative_2d(inps, theta, (1, 1))
    >>> derxy_th = derxy(inps, theta)
    >>> derxy_ex = der.derivative_2d(inps, theta, (0, 1))

    >>> print(sum(derxx_th - derxx_ex))
    -0.030212959536584094
    >>> print(sum(deryy_th - deryy_ex))
    -0.06311279423243832
    >>> print(sum(derxy_th - derxy_ex))
    -0.008896677463781089


    #--- Cost function derivative

    # 1st order derivatives
    >>> derjx_th = derjx(inps, inps, theta)
    >>> derjx_ex = der.derivative(inps, theta, 0, inps)
    >>> derjy_th = derjy(inps, inps, theta)
    >>> derjy_ex = der.derivative(inps, theta, 1, inps)
    >>> derjxy_th = derjxy(inps, inps, theta)
    >>> derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)

    >>> print(derjx_th - derjx_ex)
    1.8689315766096115e-05
    >>> print(derjy_th - derjy_ex)
    9.988434612751007e-07
    >>> print(derjxy_th - derjxy_ex)
    -5.254353880882263

    # 2nd order derivatives
    >>> derjxx_th = derjxx(inps, inps, theta)
    >>> derjxx_ex = der.derivative_2d(inps, theta, (0, 0), inps)
    >>> derjyy_th = derjyy(inps, inps, theta)
    >>> derjyy_ex = der.derivative_2d(inps, theta, (1, 1), inps)
    >>> derjxy_th = derjxy(inps, inps, theta)
    >>> derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)

    >>> print(derjxx_th - derjxx_ex)
    -10.87883397936821
    >>> print(derjyy_th - derjyy_ex)
    -13.435806035995483
    >>> print(derjxy_th - derjxy_ex)
    -5.254353880882263


    #--- Estimates' gradient
    >>> grad_th = np.array([derx(inps, theta), dery(inps, theta)]).T
    >>> grad_ex = der.gradient(inps, theta)
    >>> print(np.sum(grad_th - grad_ex))
    9.240068266080925e-08

    # Cost function gradient
    >>> gradj_th = np.array([derjx(inps, inps, theta), derjy(inps, inps, theta)]).T
    >>> gradj_ex = der.gradient(inps, theta, inps)
    >>> print(np.sum(gradj_th - gradj_ex))
    1.9688159227371216e-05

    # Quadratic cost function gradient
    >>> grad_quad_th = [np.sum((festim(inps, theta, 0) - inps)*derx(inps, theta), 0),
    ...                 np.sum((festim(inps, theta, 1) - inps)*dery(inps, theta), 0)]
    >>> grad_quad_ex = der.gradient_quad(inps, theta, inps)
    >>> print(grad_quad_th - grad_quad_ex)
    [2.43838440e-06 5.15388092e-06]
    """

    # Step for the derivatives
    EPS = 1e-6

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, festim, fcost, beta=None):
        """ Instantiate a Derivative object (constructor)

        Parameters
        ----------
        festim : reference to a function
            The estimation function to use for the computation of the
            derivatives. Must take 3 arguments: the inputs, the para-
            meter vector (wrt which the derivatives are computed) and
            any additional parameter (arg. `beta` below).
        fcost : reference to a function
            The cost function to use to compute the derivatives. Must
            take 2 arguments: the reference values and the estimates.
        [OPT] beta : any type
            Any additional parameter (e.g. parameter vector) which would
            be required by the estimation function (e.g. the multimodel-
            oriented estimators). Passed as 3rd argument to `festim` to
            calculate the derivatives.
                :Default: None

        Examples
        --------
        >>> def festim(inps, theta, *args):
        ...     return inps*(2*theta[0]**2 + theta[1]**3 + theta[0] * theta[1]**2)
        >>> def fcost(values, estims):
        ...     return np.sum(values**2 + estims**2 + values*estims)

        # Instantiate a `Derivative` (cf. class doc for `festim` & `fcost`)
        >>> der = Derivative(festim, fcost, None)

        # Properties
        >>> der.festim = festim
        >>> der.fcost = fcost
        >>> der.beta = 123
        """

        # Check the estimation and cost functions
        if not callable(festim):
            raise TypeError("Invalid type for `festim`, the estimation "
                + f"function must be a callable (received {type(festim)})")
        if not callable(fcost):
            raise TypeError("Invalid type for `fcost`, the cost "
                + f"function must be a callable (received {type(fcost)})")

        self._festim = festim           # Estimation function
        self._fcost = fcost             # Cost function
        self._beta = beta               # Second parameter vector
    #------------------------------------------------------------------------#

    #----------------------------   Properties   ----------------------------#
    @property
    def festim(self):
        """ Get the estimation function festim """
        return self._festim

    @festim.setter
    def festim(self, festim):
        """ Set the estimation function festim """
        if not callable(festim):
            raise TypeError(
                f"Invalid type, need callable (received {type(festim)})")
        self._festim = festim

    @property
    def fcost(self):
        """ Get the cost function fcost"""
        return self._fcost

    @fcost.setter
    def fcost(self, fcost):
        """ Set the cost function fcost"""
        if not callable(fcost):
            raise TypeError(
                f"Invalid type, need callable (received {type(fcost)})")
        self._fcost = fcost

    @property
    def beta(self):
        """ Get the second parameter vector beta """
        return self._beta

    @beta.setter
    def beta(self, beta):
        """ Set the second parameter vector beta """
        self._beta = beta
    #------------------------------------------------------------------------#

    #-------------------   1st Order Partial Derivative   -------------------#
    def derivative(self, inps, theta, partial, values=None):
        """ Compute the 1st order partial p-th derivative

        Compute the partial derivative wrt the parameter vector `theta`
        of a cost function or of vector estimates. The estimation func.
        is the `festim` attribute and is assumed to require two sets of
        parameters (festim(inps, theta, beta)): the first is that with
        respect to which the derivative is computed, and the second is
        assumed to be a constant here (`beta` attribute). The cost func.
        is the `fcost` attribute and is assumed to require the estimates
        (obtained by passing the `inps` and the parameter vectors to the
        estimation function and the ref. `values`); if no ref. values are
        provided, compute the partial derivative of the estimate for each
        input; otherwise, compute the cost function derivative.

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        partial : int
            Index of the partial derivative (the derivative is computed
            with respect to the partial p-th component of theta).
        [OPT] values : list, tuple or np.ndarray of floats
            Reference values. If provided, compute the cost function par-
            tial derivative between them and the estimate ones. If not
            provided, compute the partial derivative for each input.
                :Default: None (estimate partial derivatives)

        Returns
        -------
        der : float or np.ndarray
            The cost function (float) or estimates' (np.ndarray) derivative.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Derivative` (cf. class doc for `festim` & `fcost`)
        >>> der = Derivative(festim, fcost, None)

        # Compute the 1st order derivatives
        >>> derx_ex = der.derivative(inps, theta, 0)
        >>> dery_ex = der.derivative(inps, theta, 1)
        >>> derjx_ex = der.derivative(inps, theta, 0, inps)
        >>> derjy_ex = der.derivative(inps, theta, 1, inps)
        """

        # Vector of theta variations
        dth = np.full((2, len(theta)), theta, dtype=float)
        dth[:, partial] += [-self.EPS/2, self.EPS/2]

        # Estimate differentiation
        if values is None:
            values = [-self._festim(inps, dth[0], self._beta),
                      +self._festim(inps, dth[1], self._beta)]

        # Cost function differentiation
        else:
            values =\
                [-self._fcost(values, self._festim(inps, dth[0], self._beta)),
                 +self._fcost(values, self._festim(inps, dth[1], self._beta))]

        # Estimate/Cost derivative
        return np.sum(values, 0) / self.EPS
    #------------------------------------------------------------------------#

    #----------------------   2D Partial Derivative   -----------------------#
    def derivative_2d(self, inps, theta, partial, values=None):
        """ Compute the 2nd order 2D partial derivative

        Compute the cost function 2D derivative or that of estimates.
        First derivative wrt i-th theta component and second wrt j-th
        one. See `derivative` method for details.

        If theta is the current parameter vector, differentiate it wrt
        the i-th component and differentiate it anew wrt the j-th one:

            d²f/djdi = d/j(df/di)
                     = d/dj ( (f(th[i+, j]) - f(th[i-, j])) / eps )
                     = (  (f(th[i+, j+]) - f(th[i+, j-])) / eps
                        - (f(th[i-, j+]) - f(th[i-, j-])) / eps ) / eps
                     = (  f(th[i+, j+]) - f(th[i+, j-])
                        - f(th[i-, j+]) + f(th[i-, j-]) ) / eps**2

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        partial : 2-tuple
            Indices (i, j) of the partial derivatives of theta.
        [OPT] values : list, tuple or np.ndarray of floats
            Reference values. If provided, compute the cost function
            partial 2D derivatives between them and the estimates. If
            not provided, compute them for every input value.
                :Default: None (estimate partial 2D derivatives)

        Returns
        -------
        der : float or np.ndarray
            The cost function (float) or estimates' (array) derivatives.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Derivative` (cf. class doc for `festim` & `fcost`)
        >>> der = Derivative(festim, fcost, None)

        # Compute the 2nd order derivatives
        >>> derxx_ex = der.derivative_2d(inps, theta, (0, 0))
        >>> derxy_ex = der.derivative_2d(inps, theta, (0, 1))
        >>> deryy_ex = der.derivative_2d(inps, theta, (1, 1))
        >>> derjxx_ex = der.derivative_2d(inps, theta, (0, 0), inps)
        >>> derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)
        >>> derjyy_ex = der.derivative_2d(inps, theta, (1, 1), inps)
        """

        # Variation around theta
        eps = self.EPS / 2
        dth = np.full((4, len(theta)), theta)
        dth[:, partial[0]] += [-eps, -eps, eps, eps]          # ∂ϑi
        dth[:, partial[1]] += [-eps, eps, -eps, eps]          # ∂ϑj

        # Estimate 2D differentiation
        # [+f(th[i+, j+]), -f(th[i+, j-]), -f(th[i-, j+]), +f(th[i-, j-])]
        if values is None:
            values =\
                [+self._festim(inps, dth[0], self._beta),
                 -self._festim(inps, dth[1], self._beta),
                 -self._festim(inps, dth[2], self._beta),
                 +self._festim(inps, dth[3], self._beta)]

        # Cost function 2D differentiation
        # [+J(th[i+, j+]), -J(th[i+, j-]), -J(th[i-, j+]), +J(th[i-, j-])]
        else:
            values =\
                [+self._fcost(values, self._festim(inps, dth[0], self._beta)),
                 -self._fcost(values, self._festim(inps, dth[1], self._beta)),
                 -self._fcost(values, self._festim(inps, dth[2], self._beta)),
                 +self._fcost(values, self._festim(inps, dth[3], self._beta))]

        # 2D partial derivative d²f/djdi
        return np.sum(values, 0) / self.EPS**2
    #------------------------------------------------------------------------#

    #----------------------   Estimate/Cost Gradient   ----------------------#
    def gradient(self, inps, theta, values=None):
        """ Compute the gradient of an estimation or cost function

        Compute either the cost function gradient or that of estimates.
        For each component of `theta`, compute the partial derivative of
        the  estimate or cost (see `derivative` for details).

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        [OPT] values : list, tuple or np.ndarray of floats
            Reference values. If provided, compute the cost function
            gradient between them and the estimated ones. Otherwise,
            compute the estimates' gradient for every input value.
                :Default: None (estimates' gradient)

        Returns
        -------
        grad : 1D-ND np.ndarray
            The cost function (1D array) or estimates' (ND array) gradient.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Derivative` (cf. class doc for `festim` & `fcost`)
        >>> der = Derivative(festim, fcost, None)

        # Compute the gradient
        >>> grad_ex = der.gradient(inps, theta)

        # Cost function gradient
        >>> gradj_ex = der.gradient(inps, theta, inps)
        """

        # Any partial derivative vector (gradient)
        return np.array([self.derivative(inps, theta, i, values)
                         for i in range(np.size(theta))], float).T
    #------------------------------------------------------------------------#

    #------------------------   Quadratic Gradient   ------------------------#
    def gradient_quad(self, inps, theta, values):
        """ Compute the gradient of the quadratic cost function

        Restriction of the `gradient` function to the quadratic case;
        more efficient but more specific (not usable for estimation).
        If J = (1/2)*sum_n {eps(n)**2}, with eps(n) = (ys(n) - ŷ(n)),
        then deps(n)/dt = -dŷ(n)/dt and

            dJ/dt =  sum_n {eps(n) * deps(n)/dt}
                  = -sum_n {eps(n) * dŷ(n)/dt}
                  =  sum_n {(ŷ(n) - ys(n)) * dŷ(n)/dt}

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        values : list, tuple or np.ndarray of floats
            Reference values for which the cost function is computed.

        Returns
        -------
        grad : 1D np.ndarray
            The quadratic cost function gradient.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Derivative` (cf. class doc for `festim` & `fcost`)
        >>> der = Derivative(festim, fcost, None)

        # Quadratic cost function gradient
        >>> grad_quad_ex = der.gradient_quad(inps, theta, inps)
        """

        grad = np.empty(np.size(theta), float)
        for i in range(np.size(theta)):
            grad[i] = np.sum(
                (self._festim(inps, theta, self._beta) - values)    # ŷ - ys
                 * self.derivative(inps, theta, i))                 # dŷ/dt

        return grad
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Hessian Matrix Class                           ##
##############################################################################

class Hessian(Derivative):
    """ Hessian Matrix class

    Provide tools to compute the Hessian matrix or its approximate using
    the tools implemented in the class Derivative.

    If theta = th is a vector, with th^T its transpose, then the Hessian
    matrix is defined as H = d²f/dth^Tdth = d/dth^T (df/dth):

            ( d²f/dth[0]dth[0]    ...    d²f/dth[0]dth[K] )
        H = (         |            |             |        )
            ( d²f/dth[K]dth[0]    ...    d²f/dth[K]dth[K] )

    Note: according to the Schwarz's theorem (∂²f/∂i∂j = ∂²f/∂j∂i), the
          Hessian is a symmetric matrix (H[i, j] = H[j, i]).

    Constructor
    -----------
    See `Derivative` class.

    Constants
    ---------
    See `Derivative` class.

    Attributes
    ----------
    See `Derivative` class.

    Methods
    -------
    hessian(inps, theta, values)
        Compute the Hessian of an estimation or a cost function.
    hessian_quad(inps, theta, values)
        Compute the Hessian matrix of the Quadratic Cost Function.
    hessian_a(inps, theta)
        Approximate the Hessian for the Quadratic Cost Function.
    hessian_r(inps, theta, coef=0.)
        Regularize the Hessian for the Quadratic Cost Function.
    + See `Derivative` class.

    Examples
    --------
    >>> import numpy as np

    # Estimation function
    >>> def festim(inps, theta, *args):
    ...     return inps*(2*theta[0]**2 + theta[1]**3 + theta[0] * theta[1]**2)

    >>> def derx(inps, theta, *args):
    ...     return inps*(4*theta[0] + theta[1]**2)

    >>> def dery(inps, theta, *args):
    ...     return inps*(3*theta[1]**2 + 2*theta[0]*theta[1])

    >>> def derxx(inps, theta, *args):
    ...     return 4*inps

    >>> def deryy(inps, theta, *args):
    ...     return inps*(6*theta[1] + 2*theta[0])

    >>> def derxy(inps, theta, *args):
    ...     return 2*inps*theta[1]

    # Cost function
    >>> def fcost(values, estims):
    ...     return np.sum(values**2 + estims**2 + values*estims)

    >>> def derjx(values, inps, theta):
    ...     return np.sum((2*festim(inps, theta)+values) * derx(inps, theta), 0)

    >>> def derjy(values, inps, theta):
    ...     return np.sum((2*festim(inps, theta)+values) * dery(inps, theta), 0)

    >>> def derjxx(values, inps, theta):
    ...     return np.sum(2*derx(inps, theta) * derx(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * derxx(inps, theta), 0)

    >>> def derjyy(values, inps, theta):
    ...     return np.sum(2*dery(inps, theta) * dery(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * deryy(inps, theta), 0)

    >>> def derjxy(values, inps, theta):
    ...     return np.sum(2*dery(inps, theta) * derx(inps, theta)
    ...                   + (2*festim(inps, theta)+values) * derxy(inps, theta), 0)


    # Generate dummy data
    >>> inps = np.arange(10, dtype=float)
    >>> theta = (1., 2.)

    # Hessian matrix
    >>> hess = Hessian(festim, fcost, None)

    # General Hessian
    >>> mat_th = [derjxx(inps, inps, theta), derjxy(inps, inps, theta),
    ...           derjxy(inps, inps, theta), derjyy(inps, inps, theta)]
    >>> mat_ex = hess.hessian(inps, theta, inps)
    >>> print(sum(mat_th - mat_ex.ravel()))
    -34.82334777712822

    # Approximate Hessian
    >>> mata_th = [sum(derx(inps, theta) * derx(inps, theta)),
    ...            sum(derx(inps, theta) * dery(inps, theta)),
    ...            sum(dery(inps, theta) * derx(inps, theta)),
    ...            sum(dery(inps, theta) * dery(inps, theta))]
    >>> mata_ex = hess.hessian_a(inps, theta)
    >>> print(sum(mata_th - mata_ex.ravel()))
    2.8032969566993415e-05

    # Regularized Approximate Hessian
    >>> matr_th = [sum(derx(inps, theta) * derx(inps, theta))+10.,
    ...            sum(derx(inps, theta) * dery(inps, theta)),
    ...            sum(dery(inps, theta) * derx(inps, theta)),
    ...            sum(dery(inps, theta) * dery(inps, theta))+10.]
    >>> matr_ex = hess.hessian_r(inps, theta, 10.)
    >>> print(sum(matr_th - matr_ex.ravel()))
    2.8032969566993415e-05

    # Quadratic Hessian
    >>> mat_quad_th = [sum(derx(inps, theta) * derx(inps, theta)
    ...                    - (inps - festim(inps, theta))*derxx(inps, theta)),
    ...                sum(derx(inps, theta) * dery(inps, theta)
    ...                    - (inps - festim(inps, theta))*derxy(inps, theta)),
    ...                sum(dery(inps, theta) * derx(inps, theta)
    ...                    - (inps - festim(inps, theta))*derxy(inps, theta)),
    ...                sum(dery(inps, theta) * dery(inps, theta)
    ...                    - (inps - festim(inps, theta))*deryy(inps, theta))]
    >>> mat_quad_ex = hess.hessian_quad(inps, theta, inps)
    >>> print(sum(mat_quad_th - mat_quad_ex.ravel()))
    -8.74850617525226
    """

    #-------------------   Estimate/Cost Hessian Matrix   -------------------#
    def hessian(self, inps, theta, values):
        """ Compute the Hessian of an estimation or a cost function

        Compute the cost function Hessian matrix or that of estimates.
        See method `derivative_2d` for partial derivative computation.

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        values : list, tuple or np.ndarray of floats
            Reference values for which to compute the Hessian matrix.

        Returns
        -------
        hess : one or a list of 2D np.ndarray
            The cost function (1 array) or estimates' (N arrays) Hessian.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Hessian` (cf. class doc for `festim` & `fcost`)
        >>> hess = Hessian(festim, fcost, None)

        # Compute the hessian matrix
        >>> mat_ex = hess.hessian(inps, theta, inps)
        """

        size = np.size(theta)

        # Compute the upper triangle values
        hess = np.zeros((size, size), float)
        for i in range(size):
            for j in range(i+1, size):
                hess[i, j] = self.derivative_2d(inps, theta, (i, j), values)

        # Copy the upper values to the lower triangle
        hess += hess.T

        # Compute the diagonal components
        for i in range(size):
            hess[i, i] = self.derivative_2d(inps, theta, (i, i), values)

        return hess
    #------------------------------------------------------------------------#

    #------------------------   Quadratic Hessian   -------------------------#
    def hessian_quad(self, inps, theta, values):
        """ Compute the Hessian matrix of the Quadratic Cost Function

        Compute the Hessian matrix of the quadratic cost function only;
        if J = (1/2)*sum_n {eps(n)**2}, with eps(n) = (ys(n) - ŷ(n)),
        then deps(n)/dt = -dŷ(n)/dt and

            H(J) = d²J/dth^Tdth = (1/2) sum_n {d/dth^T (dJ/dth)}
                 = sum_n {d/dth^T (eps(n) deps(n)/dth))}
                 = sum_n {deps(n)/dth^T deps(n)/dth
                          + eps(n) d²eps(n)/dth^Tdth}
                 = sum_n {dŷ(n)/dth^T dŷ(n)/dth
                          - eps(n) d²ŷ(n)/dth^Tdth}
                 = sum_n {H(ŷ(n)) - eps(n)*d²ŷ(n)/dth^Tdth}
                 = sum_n {dŷ(n)/dth^T dŷ(n)/dth - (ys(n)-ŷ(n))*H(ŷ(n))}
                 = sum_n {Ha(ŷ(n)) - (ys(n)-ŷ(n))*H(ŷ(n))}

        where H(ŷ) = d²ŷ/dth^Tdth is the Hessian matrix of ŷ
        and Ha(ŷ) = dŷ/dth^T dŷ/dth is the approximate Hessian of ŷ.

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        values : list, tuple or np.ndarray of floats
            Reference values for which to compute the Hessian matrix.

        Returns
        -------
        hess : 2D np.ndarray
            The quadratic cost function Hessian matrix.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Hessian` (cf. class doc for `festim` & `fcost`)
        >>> hess = Hessian(festim, fcost, None)

        # Compute the quadratic Hessian
        >>> mat_quad_ex = hess.hessian_quad(inps, theta, inps)
        """

        size = np.size(theta)
        hess = np.zeros((size, size), float)
        ders = self.gradient(inps, theta)
        errs = values - self._festim(inps, theta, self._beta)     # Error eps

        # Compute the upper triangle values
        for i in range(size):
            for j in range(i+1, size):
                hess[i, j] = np.sum(
                    ders[:, i]*ders[:, j]
                    - errs*self.derivative_2d(inps, theta, (i, j)))

        # Copy the upper values to the lower triangle
        hess += hess.T

        # Compute the diagonal components
        for i in range(size):
            hess[i, i] = np.sum(
                ders[:, i]**2 - errs*self.derivative_2d(inps, theta, (i, i)))

        return hess
    #------------------------------------------------------------------------#

    #-----------------------   Approximate Hessian   ------------------------#
    def hessian_a(self, inps, theta):
        """ Approximate the Hessian for the Quadratic Cost Function

        Compute the Approximate Hessian of the quadratic cost function.
        See `hessian_quad` for the detailed math; Ha is the matrix mul-
        tiplication of the gradient transpose and the gradient itself:

            Ha(ŷ) = dŷ/dth^T dŷ/dth

        Assuming the second order derivative is negligible compared to
        the first order derivative, the Approximate Hessian can be used
        as a first order estimate of the real Hessian, for instance in
        the Gauss-Newton optimization algorithm:

            H(J) = sum_n {Ha(ŷ(n)) - (ys(n)-ŷ(n))*H(ŷ(n))}
                 ~ sum_n {Ha(ŷ(n))}

        Note: the real measures are not required here, for they do not
              participate in the computation of the estimates' gradient.

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.

        Returns
        -------
        hess : 2D np.ndarray
            The quadratic cost function Approximate Hessian matrix.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Hessian` (cf. class doc for `festim` & `fcost`)
        >>> hess = Hessian(festim, fcost, None)

        # Compute the hessian matrix
        >>> mat_ex = hess.hessian_a(inps, theta)
        """

        size = np.size(theta)
        hess = np.zeros((size, size), float)
        ders = self.gradient(inps, theta)

        # Compute the upper triangle values
        for i in range(size):
            for j in range(i+1, size):
                hess[i, j] = np.sum(ders[:, i]*ders[:, j])

        # Copy the upper values to the lower triangle
        hess += hess.T

        # Compute the diagonal components
        for i in range(size):
            hess[i, i] = np.sum(ders[:, i]**2)

        return hess
    #------------------------------------------------------------------------#

    #--------------------   Regularized Hessian Matrix   --------------------#
    def hessian_r(self, inps, theta, coef=0.):
        """ Regularize the Hessian for the Quadratic Cost Function

        Compute the Regularized Hessian of the quadratic cost function.
        See `hessian_quad` and `hessian_a` for the detailed math; Hr is
        defined as the regularized Approximate Hessian matrix:

            Hr(ŷ) = Ha(ŷ) + lambda*Id(K)    -- Id(K) = Identity matrix

        Adding the Identity matrix allows to regularize the Approximate
        Hessian in order to reduce its condition number; an example of
        use is the Levenberg-Marquardt optimization algorithm.

        Note: the real measures are not required here, for they do not
              participate in the computation of the estimates' gradient.

        Parameters
        ----------
        inps : list, tuple or np.ndarray of floats
            Inputs values for which the derivative is computed.
        theta : list, tuple or np.ndarray of floats
            Vector with respect to which the derivative is computed.
        [OPT] coef : float
            The lambda regulator coefficient.
                :Default: 0. (no regularization)

        Returns
        -------
        hess : 2D np.ndarray
            The quadratic cost function Regularized Hessian matrix.

        Examples
        --------
        >>> import numpy as np

        # Generate dummy ata
        >>> inps = np.arange(10, dtype=float)
        >>> theta = (1., 2.)

        # Instantiate a `Hessian` (cf. class doc for `festim` & `fcost`)
        >>> hess = Hessian(festim, fcost, None)

        # Compute the hessian matrix
        >>> matr_ex = hess.hessian_r(inps, theta, 10.)
        """
        return self.hessian_a(inps, theta) + coef*np.eye(np.size(theta))
    #------------------------------------------------------------------------#

##############################################################################
