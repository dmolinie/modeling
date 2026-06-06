import numpy as np
import modeling.optimization._optimizers as opt
from modeling.rbf_nets._kernels import *
from modeling.rbf_nets._rbf_net import *


def test_get_ker_func():
    """ Get the reference to the kernel function """
    # Linear kernel
    kernel = get_ker_func('linear')
    kernel(1.23)
    # Gaussian kernel
    kernel = get_ker_func('gaussian')
    kernel(1.23)

def test_RBFNet():
    """ Radial Basis Function (RBF) Networks """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    # Train the network
    net.fit(inps, inps, local='both')
    print(opt.mse(inps, net.predict(inps)))
    # Column array (1xN)
    vec = inps.reshape(-1, 1)
    net.fit(vec, inps)
    print(opt.mse(inps, net.predict(vec)))
    # Multi-column array (MxN)
    vec = np.hstack((inps.reshape(-1, 1), inps.reshape(-1, 1)))
    net.fit(vec, inps)
    print(opt.mse(inps, net.predict(vec)))
    # Dummy examples on array shapes
    a = np.arange(10)
    net.fit(a, a)
    print(opt.mse(a, net.predict(a)))
    b = a.reshape(-1, 1)
    net.fit(b, a)
    print(opt.mse(a, net.predict(b)))
    c = np.hstack((a.reshape(-1, 1), a.reshape(-1, 1)))
    net.fit(c, a)
    print(opt.mse(a, net.predict(c)))

def test_RBFNet_init():
    """ Instantiate an RBFNet object (constructor) """
    # RBF Network with a linear kernel and 5 functions
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    # RBF Network with a gaussian kernel and 10 functions
    net = RBFNet('gaussian', np.linspace(0, 100, 10, dtype=float))

def test_RBFNet_train_weights():
    """ Train the synaptic weights of the RBF Network """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network and train its weights
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    net._weights = net._train_weights(inps, inps)

def test_RBFNet_train_centers():
    """ Train the RBF's centers of the Network """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network and train both centers & weights
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    net._weights = net._train_weights(inps, inps)
    net._centers = net._train_centers(inps, inps, opt.mse)

def test_RBFNet_fit():
    """ Train the RBF Network """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    # Train the RBF Network's synaptic weights
    net.fit(inps, inps, local='weights')
    # Train the RBF Network's synaptic weights & RBF centers
    net.fit(inps, inps, local='both')

def test_RBFNet_summation():
    """ Provide the output of the RBF Network to a set of inputs """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    # Train the RBF Network's synaptic weights & RBF centers
    net.fit(inps, inps, local='both')
    pred  = net._summation(inps, net.centers, net.weights)
    print(opt.mse(inps, pred))

def test_RBFNet_predict():
    """ Predict the values of the RBF Network to inputs """
    # Generate dummy data
    inps = np.arange(N:=100, dtype=float) + np.random.random(N)
    # Instantiate the RBF Network
    net = RBFNet('linear', np.linspace(0, 100, 5, dtype=float))
    # Train the RBF Network's synaptic weights & RBF centers
    net.fit(inps, inps, local='both')
    print(opt.mse(inps, net.predict(inps)))



# Launch test/example functions
test_get_ker_func()

test_RBFNet()

test_RBFNet_init()

test_RBFNet_train_weights()

test_RBFNet_train_centers()

test_RBFNet_fit()

test_RBFNet_summation()

test_RBFNet_predict()

