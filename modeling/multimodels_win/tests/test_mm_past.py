import numpy as np
import modeling.optimization._optimizers as opt
from modeling.multimodels_win._mm_past import *

class Cluster():
    """ Dummy `Cluster` class """
    def __init__(self, data, tstp):
        self.value = data
        self.index = tstp
    def __getitem__(self, pos):
        return self.value[pos]


def test_nodes2data():
    """ Nodes to full sorted database (data) """
    # Generate dummy data and wrap them into a set of Clusters
    tstp = np.arange(100)
    data = np.arange(1000).reshape(100, 10)
    clusters = [Cluster(data[i*10:(i+1)*10], tstp[i*10:(i+1)*10])
                for i in range(10)]
    # Rebuild the timestamps and data from the set of Clusters
    data2 = nodes2data(clusters, 1)

def test_MultiModelPastWindow():
    """ Windowed variant of the past values-based Multi-Model """
    # Generate dummy data and wrap them into a set of Clusters
    tstp = np.linspace(0, 10, 1000)
    data = np.linspace(0, 100, 10000).reshape(1000, 10)
    clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
                for i in range(10)]
    for method in ('local', 'local_theta', 'local_beta', 'global'):
        # Instantiate the model
        model = MultiModelPastWindow(order:=2, level=method, stop=100)
        # Train the model
        model.fit(clusters, dim:=0, depth:=2, step:=0)
        # Use the model for prediction
        values = clusters[2][50:100, dim]
        estims = model.predict(values)
        print(method, '\t', f"%.3e" %opt.mse(values[depth+step:], estims))

def test_MultiModelPastWindow_fit():
    """ Select the modeling function and train the past-based multi-model """
    # Generate dummy data and wrap them into a set of Clusters
    tstp = np.linspace(0, 10, 1000)
    data = np.linspace(0, 100, 10000).reshape(1000, 10)
    clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
                for i in range(10)]
    # Instantiate the model
    model = MultiModelPastWindow(order:=2, level='local_beta', stop=100)
    # Train the model
    model.fit(clusters, dim:=0, depth:=2, step:=0)
    print(model.membership)

def test_MultiModelPastWindow_predict():
    """ Multi-Model estimate (local estimate + membership function) """
    # Generate dummy data and wrap them into a set of Clusters
    tstp = np.linspace(0, 10, 1000)
    data = np.linspace(0, 100, 10000).reshape(1000, 10)
    clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
                for i in range(10)]
    # Instantiate the model
    model = MultiModelPastWindow(order:=2, level='local_beta', stop=100)
    # Train the model
    model.fit(clusters, dim:=0, depth:=2, step:=0)
    # Use the model for prediction
    values = clusters[2][50:100, dim]
    estims = model.predict(values)
    print(f"%.3e" %opt.mse(values[depth+step:], estims))



# Launch test/example functions
test_nodes2data()

test_MultiModelPastWindow()

test_MultiModelPastWindow_fit()

test_MultiModelPastWindow_predict()

