""" Optimization algorithms

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: November 2021
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=too-many-arguments, too-many-positional-arguments

__all__ = ['STOP',
    'mse', 'linear', 'sinus', 'gaussian',
    'backtracking', 'least_squares',
    'gradient_descent_1st', 'gradient_descent_2nd']

import numpy as np

from . import _derivative as drv

# Max number of iterations for iterative methods
STOP = 100


##############################################################################
##                              Cost Functions                              ##
##############################################################################

#-----------------------   Quadratic Cost function   ------------------------#
def mse(values, estims, *args):
    """ Quadratic error cost function """
    # pylint: disable=unused-argument
    # J = (1/2)*sum_n {(ys(n)-ŷ(n))**2}
    return sum((values-estims)**2) / (2*len(values))
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Estimation Functions                           ##
##############################################################################

#-------------------------   Polynomial Estimate   --------------------------#
def linear(inputs, theta, *args):
    """ Linear (polynomial) estimation function """
    # pylint: disable=unused-argument
    return theta[0] + sum(p*inputs**(j+1) for j, p in enumerate(theta[1:]))
#----------------------------------------------------------------------------#

#-------------------------   Sinusoidal Estimate   --------------------------#
def sinus(inputs, theta, *args):
    """ Sinus estimation function """
    # pylint: disable=unused-argument
    return (theta[0]                                 # Even + Odd indexes
            + sum(p*np.cos((j+1)*inputs) for j, p in enumerate(theta[1::2]))
            + sum(p*np.sin((j+1)*inputs) for j, p in enumerate(theta[2::2])))
#----------------------------------------------------------------------------#

#--------------------------   Gaussian estimate   ---------------------------#
def gaussian(inputs, theta, *args):
    """ Gaussian estimation function """
    # pylint: disable=unused-argument
    gauss = theta[0]
    for j, (par1, par2) in enumerate(zip(theta[1::2], theta[2::2])):
        gauss += par1 * np.exp(-(j+1)*0.5*par1**2*(inputs-par2)**2)
    return gauss
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Gradient Descent                             ##
##############################################################################

#-----------------------   Backtracking Line Search   -----------------------#
def backtracking(inputs, values, theta0,
                 beta=None, coef=0.5, festim=None, fcost=None):
    """ Backtracking line search

    Estimate the optimal step for Gradient Descent. Take the values to
    interpolate, the corresponding timestamps and the current estimate
    of the parameter vector theta, and proceed to a simplified Gradient
    Descent optimization until satisfying Armijo–Goldstein condition.

    Parameters
    ----------
    inputs : list, tuple or np.ndarray of floats
        Inputs values to use for the optimization.
    values : list, tuple or np.ndarray of floats
        Reference values to interpolate.
    theta0 : any type
        Initial parameters for the 1st order Gradient Descent.
    [OPT] beta : any type
        Second parameter vector (constant for theta's optimization).
            :Default: None
    [OPT] coef : float
        Local slope for the backtracking; a higher value means a more
        accurate estimate, but also longer to obtain. Should be between
        0 and 1, but not equal to 0 nor 1.
            :Default: 0.5
    festim : reference to a function
        The estimation function to use for the computation of the deri-
        vatives. Must accept 3 arguments: the inputs, the parameter vec-
        tor (wrt which the derivatives are computed) and any additional
        parameter (arg. `beta` below).
    fcost : reference to a function
        The cost function to use for the computation of the derivatives.
        Must take 2 arguments: the reference values and the estimates.

    Returns
    -------
    theta : np.ndarray
        The optimal vector of parameters.
    alpha : float
        The optimal step.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

    # Compute the optimal parameter vector
    >>> theta, step = backtracking(
    ...     inps, inps, np.arange(3, dtype=float), None, 0.5, linear, mse)
    >>> estims = linear(inps, theta)
    >>> print(mse(inps, estims))
    2593508.390108966
    """

    # Constants
    alpha = 10.                                         # Maximal step
    cst = fcost(values, festim(inputs, theta0, beta))   # Cost to minimize

    # Initialization
    der = drv.Derivative(festim, fcost, beta)
    grad = der.gradient(inputs, theta0, values)         # Gradient to follow
    tau = coef * sum(grad**2)                           # Descent local slope
    theta = theta0 - alpha*grad                         # Initial descent

    # Iterative descent until the Armijo–Goldstein condition is satisfied
    cpt = 0
    while cst-fcost(values, festim(inputs, theta, beta)) < alpha*tau\
          and cpt < 100:
        cpt += 1
        alpha = alpha * 0.5                             # Reduce the step
        theta = theta0 - alpha*grad                     # Descent/New vector

    # Return the parameter vector and step size
    return theta, alpha
#----------------------------------------------------------------------------#

#----------------------   Variable Gradient Descent   -----------------------#
def _gd_backtracking(inputs, values, theta0,
                     beta=None, coef=0.5, festim=None, fcost=None):
    """ Variable Gradient descent (cf. `gradient_descent_1st`) """

    # Initialization
    tbf = theta0.copy()                                                   #t-2
    tcu = backtracking(inputs, values, tbf, beta, coef, festim, fcost)[0] #t-1
    taf = backtracking(inputs, values, tcu, beta, coef, festim, fcost)[0] #t

    # Iterative gradient descent (Backtracking)
    cpt = 0
    while not(all(np.round(taf, 5) == np.round(tbf, 5))) and cpt < STOP:
        cpt += 1
        tbf[:], tcu[:] = tcu, taf
        taf = backtracking(inputs, values, taf, beta, coef, festim, fcost)[0]

    return taf
#----------------------------------------------------------------------------#

#----------------------   Momentum Gradient Descent   -----------------------#
def _gd_momentum(inputs, values, theta0,
                 beta=None, rho=0.5, festim=None, fcost=None):
    """ Momentum Gradient Descent (cf. `gradient_descent_1st`) """

    # Initialization
    tbf = theta0.copy()
    step = backtracking(inputs, values, tbf, beta, rho, festim, fcost)[1] #t-2

    der = drv.Derivative(festim, fcost, beta)
    tcu = tbf - step * der.gradient(inputs, tbf, values)                  #t-1
    taf = tcu - step * der.gradient(inputs, tcu, values)                  #t

    # Iterative gradient descent (Momentum)
    cpt = 0
    while not(all(np.round(taf, 5) == np.round(tbf, 5))) and cpt < STOP:
        cpt += 1
        eps = tcu - tbf
        tbf[:], tcu[:] = tcu, taf
        taf -= step*der.gradient(inputs, taf, values) - rho*eps     # Descent

    return taf
#----------------------------------------------------------------------------#

#----------------------   Adaptive Gradient Descent   -----------------------#
def _gd_adagrad(inputs, values, theta0,
                beta=None, step=0.5, festim=None, fcost=None):
    """ Adaptive Gradient Descent (AdaGrad) (cf. `gradient_descent_1st`) """

    # Initialization (time t-2)
    tbf = theta0.copy()
    der = drv.Derivative(festim, fcost, beta)

    # First descent (time t-1)
    grad = der.gradient(inputs, tbf, values)
    mat = grad**2
    tcu = tbf - step*np.sign(grad)

    # Second descent (time t)
    grad = der.gradient(inputs, tcu, values)
    mat += grad**2
    taf = tcu - step*grad / np.sqrt(mat/2)

    # Iterative Adaptive Gradient Descent
    cpt = 2
    while not(all(np.round(taf, 3) == np.round(tbf, 3))) and cpt < STOP:
        cpt += 1
        eps = tcu - tbf
        tbf[:], tcu[:] = tcu, taf
        grad = der.gradient(inputs, taf, values)
        mat += grad**2
        taf -= step*grad / np.sqrt(mat/cpt) - step*eps              # Descent

    return taf
#----------------------------------------------------------------------------#

#----------------------   Recursive Gradient Descent   ----------------------#
def _gd_recursive(phi, values, theta0, step=1.):
    """ Recursive Gradient Descent (cf. `gradient_descent_1st`) """

    # Iterative recursive update
    taf = theta0.copy()
    for arr, val in zip(phi, values):
        taf += step * (val-np.sum(arr*taf)) * arr / (1+step*np.sum(arr**2))

    return taf
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Optimization methods                           ##
##############################################################################

#------------------------   Ordinary Least Squares   ------------------------#
def least_squares(phi, values, reg=0., method='OLS'):
    """ Ordinary/Recursive Least Squares (OLS/RLS)

    Optimization for linear estimation problems, such as polynomials.
    Take the objective values and that used for regression (contained
    in the information matrix `phi`) and estimate the optimal coefs for
    the estimation function. If Phi is the matrix composed of all the
    regression vectors at any time t, Phi = (Phi(0)^T, ..., Phi(N)^T)^T,
    it yields ys = Phi*theta, thus:

        ys = Phi * theta <=> Phi^T * ys = Phi^T * Phi * theta
            => theta = (Phi^T * Phi)^{-1} * Phi^T * ys

    For the Recursive Least Squares, the relation is not causal, thus
    one has to use an estimate of the next parameter vector:
        th{k+1} = th{k} + A{k+1} * Phi(k) * err(k+1)
    with err(k+1) = ys(k+1) - Phi(k)^T * th{k} the a priori error, and
    A{k+1} a direction matrix, such that:
        N = A{k} * Phi(k) * Phi(k)^T * A{k}
        D = 1 + Phi(k)^T * A{k} * Phi(k)
        A{k+1} = A{k} - N / D

    Parameters
    ----------
    phi : np.ndarray
        The information matrix (~ estimate's shape).
        Ex: a polynomial estimation can be given by:
            phi = np.ones((len(inputs), order+1), float)
            for i, stp in enumerate(inputs):
                phi[i, 1:] = [stp**p for p in range(1, order+1)]
    values : list, tuple or np.ndarray of floats
        Reference values to interpolate.
    [OPT] reg : float
        Regularization coefficient. If provided, it is applied in any
        case; if not, a simple addition of the identity matrix is made
        if the condition number of (Phi.T @ Phi) is too high (>1e6).
            :Default: 0. (no regularization)
    [OPT] method : str
        The Least Squares variant: Ordinary (OLS) or Recursive (RLS).
            :Default: 'OLS'

    Returns
    -------
    theta : np.ndarray
        The optimal set of parameters.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

    # Build the information matrix
    >>> phi = np.ones((len(inps), 3), dtype=float)
    >>> for i in range(1, 3):
    ...     phi[:, i] = inps**i

    # Ordinary Least Squares OLS
    >>> theta = least_squares(phi, inps, method='OLS')
    >>> estims = linear(inps, theta)
    >>> print(mse(inps, estims))
    8.453682035906844e-07

    # Recurrent Least Squares OLS
    >>> theta = least_squares(phi, inps, method='RLS')
    >>> estims = linear(inps, theta)
    >>> print(mse(inps, estims))
    8.453682036223716e-07
    """

    # Recursive Least Squares
    if method.lower() == 'rls':
        taf = np.zeros(phi.shape[1], dtype=float).reshape(-1, 1)
        mat = np.eye(phi.shape[1], dtype=float)

        # Iterative recursive update
        for i, arr in enumerate(phi):
            vec = arr.reshape(1, -1)
            mvc = mat @ vec.T
            mat -= mvc @ vec @ mat / (1. + vec @ mvc)
            taf += mat @ vec.T * (values[i] - arr @ taf)

        return taf.reshape(-1)

    # Ordinary Least Squares (default)
    # Condition number and regularization
    mat = phi.T @ phi

    if reg != 0.:
        mat += reg*np.eye(len(mat), dtype=float)
    elif np.linalg.cond(mat) > 1e6:
        mat += np.eye(len(mat), dtype=float)

    # Theta (polynomial coefficients)
    return (np.linalg.inv(mat) @ phi.T) @ values
#----------------------------------------------------------------------------#

#----------------------   1st Order Gradient Descent   ----------------------#
def gradient_descent_1st(inputs, values, theta0, beta=None, step=0.5,
                         festim=None, fcost=None, method="momentum"):
    """ First order Gradient Descent optimization

    First order nonlinear optimization algorithm. Take the objective
    values, the associated inputs and the estimation and cost functions,
    and estimate the optimal set of parameters in an iterative manner
    using the Backtracking, Momentum, AdaGrad or Recursive variant.

    The Gradient Descent consists of varying the parameter vector theta
    following the gradient's direction (sign and amplitude):
        theta{k+1} = theta{k} - mu*gradient{k}
    where mu is the descent step size.

    Since the step size is of major importance (if not correctly chosen,
    the descent diverges), it is automatically estimated by Backtracking
    Line Search when a step size is required within the variants.

    The four possible variants (`method` argument) are:
      - 'Backtracking': estimate the step for each iteration of the
         descent by backtracking; accurate but resource demanding
      - 'Momentum': estimate the step only at the beginning by back-
         tracking, and proceed the descent such that:
            th{k+1} = th{k} - mu*grad{k} - (th{k-1}-th{k-2})
      - 'AdaGrad': the step is now a 2nd order derivative estimate,
         as the square of the gradient:
            th{k+1} = th{k} - (mu/est)*grad{k}
         where est = sum_{k} grad^T grad.
         Note that the implemented version is a bit different: it is
         the mean of the `est` matrix which is used to compensate the
         step, and the momentum mechanism is also used:
            th{k+1} = th{k} - mu*grad{k}*k/est - (th{k-1}-th{k-2})
      - 'Recursive': compute the gradient of the cost function with
         regard to its next evolution dJ/dth{k+1}. Since this is not
         causal, it needs to predict the evolution of J based on the
         current state of theta and sample. The update function is:
            th{k+1} = th{k} + err(k+1)*mu*Phi(k)/(1+mu*Phi(k)^T*Phi(k))
         where err(k+1) = ys(k+1) - Phi(k)^T*th{k}, with Phi the matrix
         composed of all the regression vectors at any time t:
            Phi = (Phi(0)^T, ..., Phi(N)^T)^T
         This variant should be used when the whole database is not
         available, for instance with a recursive learning model.

    Parameters
    ----------
    inputs : list, tuple or np.ndarray of floats
        Inputs values to use for the optimization. If the method is
        `recursive`, the inputs should be an information matrix (cf.
        the `least_squares` function).
    values : list, tuple or np.ndarray of floats
        Reference values to interpolate.
    theta0 : any type
        Initial parameters for the 1st order Gradient Descent.
    [OPT] beta : any type
        Second parameter vector (constant for theta's optimization).
            :Default: None
    [OPT] step : float
        Serve a different purpose depending on the variant:
          - 'Backtracking': local slope for the backtracking
          - 'Momentum': last update multiplier rho*(th{k-1}-th{k-2})
          - 'AdaGrad': fixed step size (flattened by `est` matrix)
          - 'Recursive': fixed step size
            :Default: 0.5
        Note: minor importance (local adjustment).
    [OPT] festim : reference to a function
        Reference to an estimation function. Default to the Linear regressor.
            :Default: None (Linear regressor)
    [OPT] fcost : reference to a function
        Reference to a cost function. Default to the Mean Squared Error.
            :Default: None (Mean Squared Error)
    [OPT] method : str
        The method to be used; possibilities (no case-sensitive):
            {'Backtracking', 'Momentum', 'AdaGrad', 'Recursive'}
        Any other entry ignored and default method used instead.
            :Default: 'Momentum'

    Returns
    -------
    theta : np.ndarray
        The optimal vector of parameters.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

    # Compute the optimal parameter vector
    >>> theta = gradient_descent_1st(
    ...     inps, inps, np.zeros(3), None, 1.0, None, mse, 'backtracking')
    >>> theta = gradient_descent_1st(
    ...     inps, inps, np.zeros(3), None, 1., linear, mse, 'momentum')
    >>> theta = gradient_descent_1st(
    ...     inps, inps, np.zeros(3), None, 1.0, None, mse, 'adagrad')
    >>> theta = gradient_descent_1st(
    ...     inps, inps, np.zeros(3), None, 1.0, linear, mse, 'recursive')
    >>> estims = linear(inps, theta)
    >>> print(mse(inps, estims))
    5.858467123603192e+63
    """

    # Use the default estimation function if none is provided
    if festim is None:
        festim = linear

    # Use the default cost function if none is provided
    if fcost is None:
        fcost = mse

    # Recursive Gradient Descent
    if method.lower() == "recursive":
        return _gd_recursive(inputs, values, theta0, step)

    # Backtracking-based Gradient Descent
    if method.lower() == "backtracking":
        return _gd_backtracking(inputs, values, theta0, beta, step, festim, fcost)

    # Momentum-based Gradient Descent
    if method.lower() == "momentum":
        return _gd_momentum(inputs, values, theta0, beta, step, festim, fcost)

    # Adaptive Gradient Descent (default)
    return _gd_adagrad(inputs, values, theta0, beta, step, festim, fcost)
#----------------------------------------------------------------------------#

#----------------------   2nd Order Gradient Descent   ----------------------#
def gradient_descent_2nd(inputs, values, theta0, beta=None, step=1., coef=0.,
                         festim=None, fcost=None, method="LM"):
    """ Second order, Hessian-based nonlinear Gradient Descent optimization

    Second order nonlinear optimization algorithm. Take the objective
    values, the associated inputs and the estimation and cost functions,
    and estimate the optimal set of parameters in an iterative manner
    using the Newton, Gauss-Newton or Levenberg-Marquardt method. Stop
    after `STOP` (global variable) iterations max if no convergence.

    The Newton method uses any cost function provided, whilst the GN and
    LM variants are defined to use the Mean Squared Error only.

    Parameters
    ----------
    inputs : list, tuple or np.ndarray of floats
        Inputs values to use for the optimization.
    values : list, tuple or np.ndarray of floats
        Reference values to interpolate.
    theta0 : any type
        Initial parameters for the 2nd order Gradient Descent.
    [OPT] beta : any type
        Second parameter vector (constant for theta's optimization).
            :Default: None
    [OPT] step : float
        The step for the 2nd order gradient descent.
            :Default: 1.
    [OPT] coef : float
        The regularization coefficient; used only with LM.
            :Default: 0. (no reg. // LM <--> GN)
    [OPT] festim : reference to a function
        Reference to an estimation function. Default to the Linear regressor.
            :Default: None (Linear regressor)
    [OPT] fcost : reference to a function
        Reference to a cost function. Default to the Mean Squared Error.
            :Default: None (Mean Squared Error)
    [OPT] method : str
        The method to be used; possibilities (no case-sensitive):
          - 'Newton': Newton method
          - 'Gauss' : Gauss-Newton method
          - 'LM'    : Levenberg-Marquardt method
        Any other entry ignored and default value used instead.
            :Default: 'LM'

    Returns
    -------
    theta : np.ndarray
        The optimal vector of parameters.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> inps = np.arange(N:=100, dtype=float) + np.random.random(N)

    # Compute the optimal parameter vector
    >>> theta = gradient_descent_2nd(inps, inps, np.zeros(3), None)
    >>> theta = gradient_descent_2nd(
    ...     inps, inps, np.zeros(3), None, 1.0, 0.0, None, mse, 'GN')
    >>> theta = gradient_descent_2nd(
    ...     inps, inps, np.zeros(3), None, 1.0, 0.0, linear, mse, 'GN')
    >>> theta = gradient_descent_2nd(
    ...     inps, inps, np.zeros(3), None, 1.0, 0.0, None, mse, 'Newton')
    >>> theta = gradient_descent_2nd(
    ...     inps, inps, np.zeros(3), None, 1.0, 0.0, linear, mse, 'Newton')
    >>> estims = linear(inps, theta)
    >>> print(mse(inps, estims))
    2.858560749584777e-29
    """
    # pylint: disable=too-many-locals

    # Use the default estimation function if none is provided
    if festim is None:
        festim = linear

    # Use the default cost function if none is provided
    if fcost is None:
        fcost = mse

    # Instantiate a Derivative/Hessian objective
    der = drv.Hessian(festim, fcost, beta)

    # Select the optimization algorithm
    if method.lower() == 'newton':
        fhess = der.hessian
        param = values
    elif method.lower() in ('gauss', 'gn'):
        fhess = der.hessian_r
        param = 0.
    else: # Default method "LM"
        fhess = der.hessian_r
        param = coef

    # Initialization
    tbf = theta0.copy()
    ccu = fcost(values, festim(inputs, tbf, beta))

    # First Descent
    tmp = tbf - step*(drv.matinv(fhess(inputs, tbf, param))         # Theta
                      @ der.gradient(inputs, tbf, values))
    cst = fcost(values, festim(inputs, tmp, beta))                  # Cost
    if cst < ccu:       # Check if the update gives better results
        tcu, ccu = tmp, cst
    else:
        return tbf

    # Second Descent
    tmp = tcu - step*(drv.matinv(fhess(inputs, tcu, param))         # Theta
                      @ der.gradient(inputs, tcu, values))
    cst = fcost(values, festim(inputs, tmp, beta))                  # Cost
    if cst < ccu:       # Check if the update gives better results
        taf, ccu = tmp, cst
    else:
        return tcu

    # Iterative descent (2nd order)
    cpt = 0
    while not(all(np.round(taf, 3) == np.round(tbf, 3))) and cpt < STOP:
        cpt += 1
        tbf[:], tcu[:] = tcu, taf
        tmp = taf - step*(drv.matinv(fhess(inputs, taf, param))     # Theta
                          @ der.gradient(inputs, taf, values))
        cst = fcost(values, festim(inputs, tmp, beta))              # Cost
        if cst < ccu:   # Check if the update gives better results
            taf = tmp
            ccu = cst
        else:
            return taf

    return taf
#----------------------------------------------------------------------------#

##############################################################################
