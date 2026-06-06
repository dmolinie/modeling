import numpy as np
import modeling.optimization._optimizers as opt
from modeling.multimodels._matreg import *


def test_check_oob_index():
    """ Remove all the out-of-bounds indexes """
    # Define the prediction depth and step
    depth, step = 3, 3
    # Extract the not out-of-bounds indexes
    index = np.arange(100)
    index = check_oob_index(index, len(index), depth, step)
    print(len(index))
    # Extract 80% of the indexes for training
    index = np.arange(100)
    rng = np.random.default_rng(seed=0)
    idx_trn = rng.choice(len(index), int(0.8*len(index)), replace=False)
    ind_trn = index[idx_trn]
    ind_trn = check_oob_index(ind_trn, len(index), depth, step)
    print(len(ind_trn))

def test_matreg():
    """ Build the matrices of inputs & outputs """
    # Generate dummy data and wrap them into a set of Clusters
    data = np.linspace(0, 100, 10000).reshape(-1, 10)
    index = np.arange(len(data))
    # Generate a semi-random number generator
    rng = np.random.default_rng(seed=0)
    mask = np.full(len(index), False, dtype=bool)
    # Generate a set of training indexes (80% of the database)
    idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    mask[idx_trn] = True
    # Separate set of indexes into training and testing indexes
    ind_trn = index[mask]
    ind_gen = index[~mask]
    # Remove the out-of-bounds indexes, if any
    ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    ind_gen = check_oob_index(ind_gen, len(index), depth, step)
    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]
    # Build the regression matrix of the `cols` dimensions
    inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))
    print(inputs.shape, outputs.shape)

def test_local_matreg():
    """ Build the regression matrix """
    # Generate dummy data and wrap them into a set of Clusters
    data = np.linspace(0, 100, 10000).reshape(-1, 10)
    index = np.arange(len(data))
    # Generate a semi-random number generator
    rng = np.random.default_rng(seed=0)
    mask = np.full(len(index), False, dtype=bool)
    # Generate a set of training indexes (80% of the database)
    idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    mask[idx_trn] = True
    # Separate set of indexes into training and testing indexes
    ind_trn = index[mask]
    ind_gen = index[~mask]
    # Remove the out-of-bounds indexes, if any
    ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    ind_gen = check_oob_index(ind_gen, len(index), depth, step)
    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]
    # Build the regression matrix of the `cols` dimensions
    inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))
    # Build the regression matrices per cluster (training data)
    mat_inputs, mat_outputs = local_matreg(inputs, outputs, ind_trn_lst)
    print([mat.shape for mat in mat_inputs[0]])
    print([mat.shape for mat in mat_outputs[0]])

def test_build_inps_outs():
    """ Build the local regression matrices from the clusters' data """
    # Generate dummy data and wrap them into a set of Clusters
    data = np.linspace(0, 100, 10000).reshape(-1, 10)
    index = np.arange(len(data))
    # Generate a semi-random number generator
    rng = np.random.default_rng(seed=0)
    mask = np.full(len(index), False, dtype=bool)
    # Generate a set of training indexes (80% of the database)
    idx_trn = rng.choice(len(data), int(0.8*len(data)), replace=False)
    mask[idx_trn] = True
    # Separate set of indexes into training and testing indexes
    ind_trn = index[mask]
    ind_gen = index[~mask]
    # Remove the out-of-bounds indexes, if any
    ind_trn = check_oob_index(ind_trn, len(index), depth:=3, step:=3)
    ind_gen = check_oob_index(ind_gen, len(index), depth, step)
    # Split the training indexes into several subset to emulate
    # the groups obtained once the database is clustered
    ind_trn_lst = [ind_trn[i*200:(i+1)*200] for i in range(4)]
    # Build the regression matrix of the `cols` dimensions
    inputs, outputs = matreg(data, depth, step, (0, 1, 3, 4))
    # Build the regression matrices per cluster (training data)
    mat_inputs, mat_outputs = local_matreg(inputs, outputs, ind_trn_lst)
    # Build the Training & Testing sets
    inp_trn, out_trn, inp_gen, out_gen =\
        build_inps_outs(inputs, outputs, ind_trn, ind_gen)
    print(inp_trn.shape, out_trn.shape, inp_gen.shape, out_gen.shape)



# Launch test/example functions
test_check_oob_index()

test_matreg()

test_local_matreg()

test_build_inps_outs()

