import numpy as np
from modeling.optimization._derivative import *

# Estimation function
def festim(inps, theta, *args):
    return inps*(2*theta[0]**2 + theta[1]**3 + theta[0] * theta[1]**2)

def derx(inps, theta, *args):
    return inps*(4*theta[0] + theta[1]**2)

def dery(inps, theta, *args):
    return inps*(3*theta[1]**2 + 2*theta[0]*theta[1])

def derxx(inps, theta, *args):
    return 4*inps

def deryy(inps, theta, *args):
    return inps*(6*theta[1] + 2*theta[0])

def derxy(inps, theta, *args):
    return 2*inps*theta[1]

# Cost function
def fcost(values, estims):
    return np.sum(values**2 + estims**2 + values*estims)

def derjx(values, inps, theta):
    return np.sum((2*festim(inps, theta)+values) * derx(inps, theta), 0)

def derjy(values, inps, theta):
    return np.sum((2*festim(inps, theta)+values) * dery(inps, theta), 0)

def derjxx(values, inps, theta):
    return np.sum(2*derx(inps, theta) * derx(inps, theta)
                  + (2*festim(inps, theta)+values) * derxx(inps, theta), 0)

def derjyy(values, inps, theta):
    return np.sum(2*dery(inps, theta) * dery(inps, theta)
                  + (2*festim(inps, theta)+values) * deryy(inps, theta), 0)

def derjxy(values, inps, theta):
    return np.sum(2*dery(inps, theta) * derx(inps, theta)
                  + (2*festim(inps, theta)+values) * derxy(inps, theta), 0)


def test_matinv():
    """ Invert a matrix possibly containing all-zeros rows/cols """
    mat = np.random.random(25).reshape(5, 5)    # Dummy data
    minv = matinv(mat)                          # Inverse matrix
    print((mat @ minv).round(3))                # Check that M x MI = Id

def test_Derivative():
    """ Derivative class """
    # Generate dummy data
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Derivative object
    der = Derivative(festim, fcost, None)
    # Estimates' derivatives
    # 1st order derivatives
    derx_th = derx(inps, theta)
    derx_ex = der.derivative(inps, theta, 0)
    dery_th = dery(inps, theta)
    dery_ex = der.derivative(inps, theta, 1)
    print(sum(derx_th - derx_ex))
    print(sum(dery_th - dery_ex))
    # 2nd order derivatives
    derxx_th = derxx(inps, theta)
    derxx_ex = der.derivative_2d(inps, theta, (0, 0))
    deryy_th = deryy(inps, theta)
    deryy_ex = der.derivative_2d(inps, theta, (1, 1))
    derxy_th = derxy(inps, theta)
    derxy_ex = der.derivative_2d(inps, theta, (0, 1))
    print(sum(derxx_th - derxx_ex))
    print(sum(deryy_th - deryy_ex))
    print(sum(derxy_th - derxy_ex))
    # Cost function derivative
    # 1st order derivatives
    derjx_th = derjx(inps, inps, theta)
    derjx_ex = der.derivative(inps, theta, 0, inps)
    derjy_th = derjy(inps, inps, theta)
    derjy_ex = der.derivative(inps, theta, 1, inps)
    derjxy_th = derjxy(inps, inps, theta)
    derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)
    print(derjx_th - derjx_ex)
    print(derjy_th - derjy_ex)
    print(derjxy_th - derjxy_ex)
    # 2nd order derivatives
    derjxx_th = derjxx(inps, inps, theta)
    derjxx_ex = der.derivative_2d(inps, theta, (0, 0), inps)
    derjyy_th = derjyy(inps, inps, theta)
    derjyy_ex = der.derivative_2d(inps, theta, (1, 1), inps)
    derjxy_th = derjxy(inps, inps, theta)
    derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)
    print(derjxx_th - derjxx_ex)
    print(derjyy_th - derjyy_ex)
    print(derjxy_th - derjxy_ex)
    # Estimates' gradient
    grad_th = np.array([derx(inps, theta), dery(inps, theta)]).T
    grad_ex = der.gradient(inps, theta)
    print(np.sum(grad_th - grad_ex))
    # Cost function gradient
    gradj_th = np.array([derjx(inps, inps, theta), derjy(inps, inps, theta)]).T
    gradj_ex = der.gradient(inps, theta, inps)
    print(np.sum(gradj_th - gradj_ex))
    # Quadratic cost function gradient
    grad_quad_th = [np.sum((festim(inps, theta, 0) - inps)*derx(inps, theta), 0),
                    np.sum((festim(inps, theta, 1) - inps)*dery(inps, theta), 0)]
    grad_quad_ex = der.gradient_quad(inps, theta, inps)
    print(grad_quad_th - grad_quad_ex)

def test_Derivative_init():
    """ Instantiate a Derivative object (constructor) """
    der = Derivative(festim, fcost, None)
    der.festim = festim
    der.fcost = fcost
    der.beta = 123

def test_Derivative_meth():
    """ Compute the 1st order partial p-th derivative """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Derivative` object
    der = Derivative(festim, fcost, None)
    # Compute the 1st order derivatives
    derx_ex = der.derivative(inps, theta, 0)
    dery_ex = der.derivative(inps, theta, 1)
    derjx_ex = der.derivative(inps, theta, 0, inps)
    derjy_ex = der.derivative(inps, theta, 1, inps)

def test_Derivative_2d_meth():
    """ Compute the 2nd order 2D partial derivative """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Derivative` object
    der = Derivative(festim, fcost, None)
    # Compute the 2nd order derivatives
    derxx_ex = der.derivative_2d(inps, theta, (0, 0))
    derxy_ex = der.derivative_2d(inps, theta, (0, 1))
    deryy_ex = der.derivative_2d(inps, theta, (1, 1))
    derjxx_ex = der.derivative_2d(inps, theta, (0, 0), inps)
    derjxy_ex = der.derivative_2d(inps, theta, (0, 1), inps)
    derjyy_ex = der.derivative_2d(inps, theta, (1, 1), inps)

def test_gradient_meth():
    """ Compute the gradient of an estimation or cost function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Derivative` object
    der = Derivative(festim, fcost, None)
    # Compute the gradient
    grad_ex = der.gradient(inps, theta)
    # Cost function gradient
    gradj_ex = der.gradient(inps, theta, inps)

def test_gradient_quad_meth():
    """ Compute the gradient of the quadratic cost function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Derivative` object
    der = Derivative(festim, fcost, None)
    # Quadratic cost function gradient
    grad_quad_ex = der.gradient_quad(inps, theta, inps)

def test_Hessian():
    """ Hessian Matrix class """
    # Generate dummy data
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Hessian matrix
    hess = Hessian(festim, fcost, None)
    # General Hessian
    mat_th = [derjxx(inps, inps, theta), derjxy(inps, inps, theta),
              derjxy(inps, inps, theta), derjyy(inps, inps, theta)]
    mat_ex = hess.hessian(inps, theta, inps)
    print(sum(mat_th - mat_ex.ravel()))
    # Approximate Hessian
    mata_th = [sum(derx(inps, theta) * derx(inps, theta)),
               sum(derx(inps, theta) * dery(inps, theta)),
               sum(dery(inps, theta) * derx(inps, theta)),
               sum(dery(inps, theta) * dery(inps, theta))]
    mata_ex = hess.hessian_a(inps, theta)
    print(sum(mata_th - mata_ex.ravel()))
    # Regularized Approximate Hessian
    matr_th = [sum(derx(inps, theta) * derx(inps, theta))+10.,
               sum(derx(inps, theta) * dery(inps, theta)),
               sum(dery(inps, theta) * derx(inps, theta)),
               sum(dery(inps, theta) * dery(inps, theta))+10.]
    matr_ex = hess.hessian_r(inps, theta, 10.)
    print(sum(matr_th - matr_ex.ravel()))
    # Quadratic Hessian
    mat_quad_th = [sum(derx(inps, theta) * derx(inps, theta)
                       - (inps - festim(inps, theta))*derxx(inps, theta)),
                   sum(derx(inps, theta) * dery(inps, theta)
                       - (inps - festim(inps, theta))*derxy(inps, theta)),
                   sum(dery(inps, theta) * derx(inps, theta)
                       - (inps - festim(inps, theta))*derxy(inps, theta)),
                   sum(dery(inps, theta) * dery(inps, theta)
                       - (inps - festim(inps, theta))*deryy(inps, theta))]
    mat_quad_ex = hess.hessian_quad(inps, theta, inps)
    print(sum(mat_quad_th - mat_quad_ex.ravel()))

def test_Hessian_meth():
    """ Compute the Hessian of an estimation or a cost function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Hessian` object
    hess = Hessian(festim, fcost, None)
    # Compute the hessian matrix
    mat_ex = hess.hessian(inps, theta, inps)

def test_Hessian_quad_meth():
    """ Compute the Hessian matrix of the Quadratic Cost Function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Hessian` object
    hess = Hessian(festim, fcost, None)
    # Compute the quadratic Hessian
    mat_quad_ex = hess.hessian_quad(inps, theta, inps)

def test_Hessian_a_meth():
    """ Approximate the Hessian for the Quadratic Cost Function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Hessian` object
    hess = Hessian(festim, fcost, None)
    # Compute the hessian matrix
    mat_ex = hess.hessian_a(inps, theta)

def test_Hessian_r_meth():
    """ Regularize the Hessian for the Quadratic Cost Function """
    # Generate dummy ata
    inps = np.arange(10, dtype=float)
    theta = (1., 2.)
    # Instantiate a `Hessian` object
    hess = Hessian(festim, fcost, None)
    # Compute the hessian matrix
    matr_ex = hess.hessian_r(inps, theta, 10.)



# Launch test/example functions
test_matinv()

test_Derivative()

test_Derivative_init()

test_Derivative_meth()

test_Derivative_2d_meth()

test_gradient_meth()

test_gradient_quad_meth()

test_Hessian()

test_Hessian_meth()

test_Hessian_quad_meth()

test_Hessian_a_meth()

test_Hessian_r_meth()

