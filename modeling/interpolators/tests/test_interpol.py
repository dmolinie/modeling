import numpy as np
import modeling.optimization._optimizers as opt
from modeling.interpolators._interpol import *


def test_Interpolator():
    """ Polynomial/Sinusoidal Interpolation """
    # Instantiate the interpolator
    inter = Interpolator(5)
    inter = Interpolator(5, 'poly')
    inter = Interpolator(3, 'polynomial')
    inter = Interpolator(5, 'sinus')
    inter = Interpolator(3, 'polynomial')
    inter.interpolator = 'poly'
    # Train the interpolator and use it for prediction
    inps = np.arange(10)
    mat = inter.matinfo(inps)
    inter.fit(inps, inps)
    print(inter.theta)
    inter.predict(inps)
    # Column array (1xN)
    vec = inps.reshape(-1, 1)
    mat = inter.matinfo(vec)
    inter.fit(vec, inps)
    print(opt.mse(inps, inter.predict(vec)))
    # Multi-column array (MxN)
    vec = np.hstack((inps.reshape(-1, 1), inps.reshape(-1, 1)))
    mat = inter.matinfo(vec)
    inter.fit(vec, inps)
    print(opt.mse(inps, inter.predict(vec)))
    # Dummy examples on array shapes
    a = np.arange(10)
    inter.fit(a, a)
    resa = inter.predict(a)
    print(opt.mse(a, resa))
    b = a.reshape(-1, 1)
    inter.fit(b, a)
    resb = inter.predict(b)
    print(opt.mse(a, resb))
    c = np.hstack((a.reshape(-1, 1), a.reshape(-1, 1)+3))
    inter.fit(c, a)
    resc = inter.predict(c)
    print(opt.mse(a, resc))

def test_Interpolator_init():
    """ Instantiate an Interpolator object (constructor) """
    # Order 3 polynomial interpolator
    interpol = Interpolator(3, 'polynomial')
    # Order 1 sinusoidal interpolator
    interpol = Interpolator(1, 'sinusoidal')

def test_Interpolator_matinfo():
    """ Build the Information Matrix """
    # Generate dummy data
    inps = np.arange(10)
    # Build the Interpolator & its information matrix
    inter = Interpolator(3, 'polynomial')
    mat = inter.matinfo(inps)
    print(mat)

def test_Interpolator_fit():
    """ Train the Interpolator """
    # Generate dummy data
    inps = np.arange(10)
    # Build the Interpolator and train it
    inter = Interpolator(3, 'polynomial')
    inter.fit(inps, inps)
    print(inter.theta)

def test_Interpolator_predict():
    """ Predict the values of the Interpolator to inputs """
    # Generate dummy data
    inps = np.arange(10)
    vec = inps.reshape(-1, 1)
    # Build the Interpolator, train it and use it for prediction
    inter = Interpolator(3, 'polynomial')
    inter.fit(vec, inps)
    print(opt.mse(inps, inter.predict(vec)))



# Launch test/example functions
test_Interpolator()

test_Interpolator_init()

test_Interpolator_matinfo()

test_Interpolator_fit()

test_Interpolator_predict()

