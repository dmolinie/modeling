import numpy as np
import modeling.optimization._optimizers as opt
from modeling.interpolators._interpol import Interpolator
from modeling.multimodels._multimodels import *

def function(vals):
    """ Function to interpolate """    
    return vals**3 - 5.*vals**2 - 10.*vals

def matreg(vals, depth, step):
    """ Build the regression matrix """
    inps_glb = np.empty((len(vals)-depth-step, depth), float)
    for i in range(depth):
        inps_glb[:, i] = vals[i:-depth-step+i]
    outs_glb = vals[depth+step:]
    return inps_glb, outs_glb


def test_MultiModel():
    """ Membership-based Multi-Model class """
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
        inp_trn, inp_gen = inp[:frac], inp[frac:]
        out_trn, out_gen = out[:frac], out[frac:]
        mat_inps_trn.append(inp_trn)
        mat_inps_gen.append(inp_gen)
        mat_outs_trn.append(out_trn)
        mat_outs_gen.append(out_gen)
    #----- Select the interpolator (local model) -----#
    # Local project generic interpolator
    from modeling.interpolators._interpol import Interpolator
    method = Interpolator
    params = {'order': 2, 'interpolator': 'polynomial'}
    ## Local project past-data interpolator 
    #from modeling.interpolators._interpol_past import InterpolatorPast
    #method = InterpolatorPast
    #params = {'order': 2}
    ## Local project RBF Networks
    #from modeling.rbf_net import *
    #method = RBFNet
    #params = {'kernel': 'linear', 'centers': np.linspace(0, 1.5, 10, dtype=float)}
    ## Scikit-Learn models
    ## Decision Trees
    #from sklearn.tree import DecisionTreeRegressor
    #method = DecisionTreeRegressor
    #params = {'max_depth': 10}
    ## Random Forests
    #from sklearn.ensemble import RandomForestRegressor
    #method = RandomForestRegressor
    #params = {'n_estimators': 20, 'max_depth': 8}
    ## Standard MLP
    #from sklearn.neural_network import MLPRegressor
    #method = MLPRegressor
    #params = {'hidden_layer_sizes': 250, 'activation': 'logistic',
    #          'solver': 'adam', 'max_iter': 1000, 'verbose': False}
    #-------------------------------------------------#
    for level in ('local', 'local_theta', 'local_beta', 'global'):
        # Instantiate & train the multimodel
        model = MultiModel(method, params, level)
        model.fit(mat_inps_trn, mat_outs_trn)
        # Evaluate the multimodel's accuracy
        cost = 0.
        for inp, out in zip(mat_inps_gen, mat_outs_gen):
            cost += opt.mse(out, model.predict(inp))
        print(level, '\t', f"%.5e" %cost)

def test_MultiModel_init():
    """ Instantiate a MultiModel object (constructor) """
    #--- Locally-implemented models
    from modeling.interpolators import Interpolator
    # Use a simple interpolator
    method = Interpolator
    params = {'order': 2, 'interpolator': 'polynomial'}
    # Instantiate the Multi-Model Networks with local Interpolators
    # with local optimization for both models and memberships
    mmodel = MultiModel(method, params, 'local')
    #--- Scikit-Learn models
    from sklearn.tree import DecisionTreeRegressor
    # Use simple Decision Trees
    method = DecisionTreeRegressor
    params = {'max_depth': 10}
    # Instantiate the Multi-Model with local Decision Trees
    # with local optimization for models and global for memberships
    mmodel = MultiModel(method, params, 'local_beta')

def test_MultiModel_fit():
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
    # Local project generic interpolator
    method = Interpolator
    params = {'order': 2, 'interpolator': 'polynomial'}
    # Instantiate the multimodel
    model = MultiModel(method, params, 'local_beta')
    # Train the multimodel
    model.fit(inps_loc, outs_loc)
    print(model.membership)

def test_MultiModel_predict():
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
    # Local project generic interpolator
    method = Interpolator
    params = {'order': 2, 'interpolator': 'polynomial'}
    # Instantiate & train the multimodel
    model = MultiModel(method, params, 'local_beta')
    model.fit(inps_loc, outs_loc)
    # Evaluate the multimodel's accuracy
    cost = 0.
    for inp, out in zip(inps_loc, outs_loc):
        cost += opt.mse(out, model.predict(inp))
    print(f"%.5e" %cost)



# Launch test/example functions
test_MultiModel()

test_MultiModel_init()

test_MultiModel_fit()

test_MultiModel_predict()

