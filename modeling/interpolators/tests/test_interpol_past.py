import numpy as np
import modeling.optimization._optimizers as opt
from modeling.interpolators._interpol_past import *

def function(vals):
    """ Function to interpolate """    
    return vals**3 - 5.*vals**2 - 10.*vals

def matreg(vals, depth, step):
    """ Build the regression matrix """
    inputs = np.empty((len(vals)-depth-step, depth), float)
    for i in range(depth):
        inputs[:, i] = vals[i:-depth-step+i]
    outputs = vals[depth+step:]
    return inputs, outputs


def test_InterpolatorPast():
    """ Regression Matrix-based past-values Model """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inputs, outputs = matreg(values, depth:=3, step:=0)
    # Split the dataset (regression matrix) into a trainig & testing sets
    frac = int(len(inputs) * 0.8)
    inps_trn, inps_gen = inputs[:frac], inputs[frac:]
    outs_trn, outs_gen = outputs[:frac], outputs[frac:]
    # Instantiate & train the multimodel
    model = InterpolatorPast(order=3)
    model.fit(inps_trn, outs_trn)
    # Evaluate the model's accuracy
    print("Training", f"%.5e" %opt.mse(outs_trn, model.predict(inps_trn)))
    print("Testing", f"%.5e" %opt.mse(outs_gen, model.predict(inps_gen)))

def test_InterpolatorPast_init():
    """ Instantiate an InterpolatorPast object (constructor) """
    model = InterpolatorPast()
    model = InterpolatorPast(order=5)

def test_InterpolatorPast_fit():
    """ Select the modeling function and train the model """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inputs, outputs = matreg(values, depth:=3, step:=0)
    # Instantiate & train the multimodel
    model = InterpolatorPast(order=3)
    model.fit(inputs, outputs)
    print(model.regressors)

def test_InterpolatorPast_predict():
    """ Use the Model to predict the data in response to `inputs` """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inputs, outputs = matreg(values, depth:=3, step:=0)
    # Instantiate & train the multimodel
    model = InterpolatorPast(order=3)
    model.fit(inputs, outputs)
    # Evaluate the model's accuracy
    print(f"%.5e" %opt.mse(outputs, model.predict(inputs)))

def test_MultiModelPast():
    """ Regression Matrix-based past-values Multi-Model """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inps_glb, outs_glb = matreg(values, depth:=3, step:=0)
    # Split database into K sub-datasets
    K = 3
    N = len(inps_glb) // K
    inps_loc, outs_loc = [], []
    for i in range(K):
        inps_loc.append(inps_glb[i*N:(i+1)*N])
        outs_loc.append(outs_glb[i*N:(i+1)*N])
    # Build the local regression matrices
    mat_inps_trn, mat_outs_trn = [], []
    mat_inps_gen, mat_outs_gen = [], []
    for inp, out in zip(inps_loc, outs_loc):
        # inp_trn, inp_gen, out_trn, out_gen = \
        #    sklearn.model_selection.train_test_split(inp, out)
        frac = int(len(inp) * 0.8)
        inp_trn = inp[:frac]
        inp_gen = inp[frac:]
        out_trn = out[:frac]
        out_gen = out[frac:]
        mat_inps_trn.append(inp_trn)
        mat_inps_gen.append(inp_gen)
        mat_outs_trn.append(out_trn)
        mat_outs_gen.append(out_gen)
    for level in ('local', 'local_theta', 'local_beta', 'global'):
        # Instantiate & train the multimodel
        model = MultiModelPast(order=3, level=level, stop=100)
        model.fit(mat_inps_trn, mat_outs_trn)
        # Evaluate the multimodel's accuracy
        cost = 0.
        for inp, out in zip(mat_inps_gen, mat_outs_gen):
            cost += opt.mse(out, model.predict(inp))
        print(level, '\t', f"%.5e" %cost)

def test_MultiModelPast_fit():
    """ Select the modeling function and train the multi-model """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inps_glb, outs_glb = matreg(values, depth:=3, step:=0)
    # Split database into K sub-datasets
    K = 3
    N = len(inps_glb) // K
    inps_loc, outs_loc = [], []
    for i in range(K):
        inps_loc.append(inps_glb[i*N:(i+1)*N])
        outs_loc.append(outs_glb[i*N:(i+1)*N])
    # Instantiate & train the multimodel
    model = MultiModelPast(order=3, level='local_beta', stop=100)
    model.fit(inps_loc, outs_loc)
    print(model.regressors)

def test_MultiModelPast_predict():
    """ Multi-Model estimate (local estimate + membership function) """
    # Generate dummy data
    values = function(np.linspace(-1, +1, 1000, float))
    # Build the global regression matrix
    inps_glb, outs_glb = matreg(values, depth:=3, step:=0)
    # Split database into K sub-datasets
    K = 3
    N = len(inps_glb) // K
    inps_loc, outs_loc = [], []
    for i in range(K):
        inps_loc.append(inps_glb[i*N:(i+1)*N])
        outs_loc.append(outs_glb[i*N:(i+1)*N])
    # Instantiate & train the multimodel
    model = MultiModelPast(order=3, level='local_beta', stop=100)
    model.fit(inps_loc, outs_loc)
    # Evaluate the multimodel's accuracy
    cost = 0.
    for inp, out in zip(inps_loc, outs_loc):
        cost += opt.mse(out, model.predict(inp))
    print(f"%.5e" %cost)



# Launch test/example functions
test_InterpolatorPast()

test_InterpolatorPast_init()

test_InterpolatorPast_fit()

test_InterpolatorPast_predict()

test_MultiModelPast()

test_MultiModelPast_fit()

test_MultiModelPast_predict()

