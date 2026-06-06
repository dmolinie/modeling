import numpy as np
from modeling.optimization._optimizers import *

#def linear(inputs, theta, *args):
#    vals = np.full(len(inputs), theta[0])
#    for i, th in enumerate(theta):
#        vals += th*inputs**i
#    return vals

festim = linear


def test_backtracking():
    """ Backtracking line search """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Compute the optimal parameter vector
    theta, step = backtracking(
        inps, inps, np.arange(3, dtype=float), None, 0.5, linear, mse)
    estims = linear(inps, theta)
    print(mse(inps, estims))

def test_least_squares():
    """ Ordinary/Recursive Least Squares (OLS/RLS) """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Build the information matrix
    phi = np.ones((len(inps), 3), dtype=float)
    for i in range(1, 3):
        phi[:, i] = inps**i
    # Ordinary Least Squares OLS
    theta = least_squares(phi, inps, method='OLS')
    estims = linear(inps, theta)
    print(mse(inps, estims))
    # Recurrent Least Squares OLS
    theta = least_squares(phi, inps, method='RLS')
    estims = linear(inps, theta)
    print(mse(inps, estims))

def test_gradient_descent_1st():
    """ Gradient Descent optimization """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Compute the optimal parameter vector
    theta = gradient_descent_1st(
        inps, inps, np.zeros(3), None, 1.0, None, mse, 'backtracking')
    theta = gradient_descent_1st(
        inps, inps, np.zeros(3), None, 1., linear, mse, 'momentum')
    theta = gradient_descent_1st(
        inps, inps, np.zeros(3), None, 1.0, None, mse, 'adagrad')
    theta = gradient_descent_1st(
        inps, inps, np.zeros(3), None, 1.0, linear, mse, 'recursive')
    estims = linear(inps, theta)
    print(mse(inps, estims))

def test_gradient_descent_2nd():
    """ Hessian-based nonlinear optimization (Newton, Gauss, LM) """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Compute the optimal parameter vector
    theta = gradient_descent_2nd(inps, inps, np.zeros(3), None)
    theta = gradient_descent_2nd(
        inps, inps, np.zeros(3), None, 1.0, 0.0, None, mse, 'GN')
    theta = gradient_descent_2nd(
        inps, inps, np.zeros(3), None, 1.0, 0.0, linear, mse, 'GN')
    theta = gradient_descent_2nd(
        inps, inps, np.zeros(3), None, 1.0, 0.0, None, mse, 'Newton')
    theta = gradient_descent_2nd(
        inps, inps, np.zeros(3), None, 1.0, 0.0, linear, mse, 'Newton')
    estims = linear(inps, theta)
    print(mse(inps, estims))



# Launch test/example functions
test_backtracking()

test_least_squares()

test_gradient_descent_1st()

test_gradient_descent_2nd()

