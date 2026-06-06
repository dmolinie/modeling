""" Example of use of the `modeling` package

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: July 2024
Last revised: June 2026

License: GPLv3
"""

import numpy as np

# Split the dataset into training & testing
from sklearn.model_selection import train_test_split

# Tools for optimizing & multimodeling
import modeling.optimization as opt
import modeling.multimodels as multi

# Local models from current implementation or from Scikit-Learn suit
from modeling.interpolators import Interpolator, InterpolatorPast
from modeling.rbf_nets import RBFNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


##############################################################################
##                          Train the Multi-Model                           ##
##############################################################################

# Set the prediction parameters
COLS = [0, 3]                               # Dimensions to model
DEPTH = 3                                   # Nb of previous data to use
STEP = 3                                    # Prediction step

# Set of models to use
METHODS = [
    [Interpolator,
     {'order': 3, 'interpolator': 'polynomial'},
     "Polynomial Interpolators"],
    [InterpolatorPast,
     {'order': 3},
     "Past-Value Interpolators"],
    [MLPRegressor,
     {'hidden_layer_sizes': 250, 'activation': 'logistic',
      'solver': 'adam', 'max_iter': 1000, 'verbose': False},
     "Multi-Layer Perceptrons"],
    [DecisionTreeRegressor,
     {'max_depth': 10},
     "Decision Trees"],
    [RandomForestRegressor,
     {'n_estimators': 20, 'max_depth': 6},
     "Random Forests"],
    [RBFNet,
     {'kernel': 'linear', 'centers': np.linspace(0, 1.5, 10, dtype=float)},
     "RBF Networks"]
]

# Generate a dummy database with 3 Gaussian distributions (regions)
rng = np.random.default_rng()
data = np.vstack(
    (rng.normal(loc=0.5, scale=0.01, size=(200, 5)),
     rng.normal(loc=5.0, scale=0.01, size=(200, 5)),
     rng.normal(loc=9.5, scale=0.01, size=(200, 5))))
# Generate a simple set of indexes
index = np.arange(len(data))

# Generate the training and testing sets of indexes
ind_trn, ind_gen = train_test_split(index, test_size=0.20, random_state=2)

#------------------------   Use direct Time Series   ------------------------#
# Below is an example using direct time-series data, as `np.ndarrays`.
# The out-of-bounds indexes (i.e. those ahead the prediction window) are
# removed from both training & testing indexes, and the list of training
# indexes (thus data) are stacked as distinct arrays to emulate a set of
# clusters to model and link within a multimodel.

# Remove the out-of-bounds indexes, if any
ind_trn = multi.check_oob_index(ind_trn, len(data), DEPTH, STEP)
ind_gen = multi.check_oob_index(ind_gen, len(data), DEPTH, STEP)

# Split the training indexes into several subsets to emulate several
# groups of data (clusters) obtained on the training data only
ind_trn_lst = [ind_trn[i*100:(i+1)*100] for i in range(4)]
#----------------------------------------------------------------------------#

#---------------------   Link to `clustering` Package   ---------------------#
# Below is an example with a true clustering method from the `clustering`
# package; the data & indexes are wrapped into a `Database` class container,
# which is then split using a clustering method (Kohonen's Self-Organizing
# Maps here). Then, the indexes of the training data from the different
# clusters are extracted, indexes that will be used to build the local
# regions matrices later on.

from clustering.formats import Database

# Wrap the data & indexes into a `Database` container
database = Database(data, index)

# Data partitioning
import clustering.cluster as clt

# Split the database into a training and a testing databases
dba_trn = database.select(ind_trn)          # Training dataset
dba_gen = database.select(ind_gen)          # Testing dataset

# Split the database
ksom_params = {
    'nb_clusters': 2, 'cluster': True, 'margins': 0.01, 'tmax': 100,
    'seed': 0, 'verbose': True, 'distance': 'euclidean'}
clusters = clt.cluster(dba_trn, method='ksom', fuse=0., **ksom_params)

# Split the database into training and testing datasets
ind_trn_lst, ind_trn, ind_gen =\
    clt.rebuild_idx(database, clusters, len(database)-DEPTH-STEP)
#----------------------------------------------------------------------------#

#------------------------   Train the Multi-Models   ------------------------#
# Build the whole dataset's regression matrix and its set of objective values
inputs, outputs = multi.matreg(data, DEPTH, STEP, COLS) # or `database.value` if `Database`

# Build the regression matrices for the different dimensions of `COLS`
mat_inputs, mat_outputs = multi.local_matreg(inputs, outputs, ind_trn_lst)

# Build the sets of training & testing sets of inputs & outputs
# from the regression matrices
inp_trn, out_trn, inp_gen, out_gen =\
    multi.build_inps_outs(inputs, outputs, ind_trn, ind_gen)

# Train the local models
models = []
costs = np.empty((len(COLS), len(METHODS), 2), float)
for i, (matinp, matout, intrn, outrn, ingen, ougen) in enumerate(
    zip(mat_inputs, mat_outputs, inp_trn, out_trn, inp_gen, out_gen)):
    mods = []
    for j, method in enumerate(METHODS):
        # Train the Gating Networks
        model = multi.GatingNetworks(method[0], method[1])
        model.fit(matinp, matout)
        # Use the GN for prediction
        costs[i, j, 0] = opt.mse(model.predict(intrn), outrn)
        costs[i, j, 1] = opt.mse(model.predict(ingen), ougen)
        mods.append(model)
    models.append(mods)
print("MSE costs (<dimension>, <method>, <set>):\n", costs)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Display the Results                            ##
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14})

    def set_right_yaxis_label(axis, text, **rtext_params):
        """ Write right y-axis side stacked texts """
        esp = "\n     "
        axis_rgt = axis.twinx()
        axis_rgt.tick_params(right=False, labelright=False)
        axis_rgt.yaxis.set_label_position('right')
        axis_rgt.set_ylabel(esp[1:]+esp.join(text), **rtext_params)


    # Extract the indexes of the training & the testing data
    tstp_trn = index[ind_trn+DEPTH+STEP]     # or `database.index` if `Database`
    tstp_gen = index[ind_gen+DEPTH+STEP]     # or `database.index` if `Database`

    # Instantiate the figure
    fig, axs = plt.subplots(
        len(COLS), len(METHODS),
        figsize=(19.20, 10.80), sharex='col', sharey='row')

    # Plot the predictions on the training data
    for i, (mods, intrn) in enumerate(zip(models, inp_trn)):
        for j, model in enumerate(mods):
            est_trn = model.predict(intrn)
            axs[i, j].plot(tstp_trn, est_trn, '+')

    # Plot the predictions on the testing data
    for i, (mods, ingen) in enumerate(zip(models, inp_gen)):
        for j, model in enumerate(mods):
            est_gen = model.predict(ingen)
            axs[i, j].plot(tstp_gen, est_gen, '+')

    # Axis labels
    for axis in axs[:, 0]:
        axis.set_ylabel("Values [a.u.]", size=18)    
    for axis in axs[-1]:
        axis.set_xlabel("Indexes [a.u.]", size=18)
    for col, axis in zip(COLS, axs[:, -1]):
        set_right_yaxis_label(axis, f"Dim. {col}", rotation=0, va='center')

    # Axis titles
    fig.suptitle("Multi-modeling with Gating Networks "
        + f"(depth={DEPTH} & step={STEP})", size=24)
    for axis, method in zip(axs[0], METHODS):
        axis.set_title(method[2], size=20)

    # Legends
    axs[-1, -1].legend(
        ["Training data", "Testing data"], loc='upper left', fontsize=14)

    # Display the figure
    plt.tight_layout()
    plt.show()

##############################################################################
