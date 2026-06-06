""" Multi-Models using past data as variable

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2022
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['nodes2data', 'MultiModelPastWindow']

import numpy as np

import modeling.optimization._optimizers as opt
from . _multimodel import _MultiModel

# Default cost function
cost_func = opt.mse


##############################################################################
##                         Database reconstruction                          ##
##############################################################################

def nodes2data(clusters, dim):
    """ Nodes to full sorted database (data)

    Take a set of clusters and extract their data. First, retrieve the
    timestamps of every cluster and stack them all into a unique array,
    and sort this array in ascending order. Then, retrieve the data of
    every cluster and stack them all into a unique array before sorting
    them in the order of the timestamps previously built. Return only
    the `dim` dimension of the so-sorted data.

    Parameters
    ----------
    clusters : set of Clusters (from the `clustering` package)
        The clusters for which to extract the data.
    dim : int
        The dimension of the data to extract.

    Returns
    -------
    data : np.ndarray
        The clusters' `dim` dimension data, vertically stacked.

    Examples
    --------
    >>> import numpy as np

    >>> class Cluster():
    ...     ''' Dummy `Cluster` class '''
    ...     def __init__(self, data, tstp):
    ...         self.value = data
    ...         self.index = tstp

    # Generate dummy data and wrap them into a set of Clusters
    >>> tstp = np.arange(100)
    >>> data = np.arange(1000).reshape(100, 10)
    >>> clusters = [Cluster(data[i*10:(i+1)*10], tstp[i*10:(i+1)*10])
    ...             for i in range(10)]

    # Rebuild the data from the set of Clusters, sorted by timestamps
    >>> data2 = nodes2data(clusters, 1)
    """
    tstp = np.hstack([node.index for node in clusters])         # All indexes
    idx = tstp.argsort()                                        # Sorted depth
    return np.vstack([k.value for k in clusters])[idx, dim]

##############################################################################



##############################################################################
##                          Past-based Multi-Model                          ##
##############################################################################

class MultiModelPastWindow(_MultiModel):
    r""" Windowed variant of the past values-based Multi-Model

    Take a set of clusters and build the multi-model by modeling every
    local group of data, before connecting them all in a linear fashion.
    By denoting `val` a vector of inputs, the multi-model is defined as
    a linear combination of the estimates outputted by any of the K local
    models, each multiplied by a Gaussian-like membership function (MF):
        ŷ(val, theta, beta) = sum_k^K w_k(val, beta_k) * f_k(val, theta_k)

    where `beta` and `theta` are the sets of the K vectors `beta_k` and
    `theta_k`, each being the parameter vector of the membership function
    and local model, respectively, of the k-th cluster.

    The local models are defined as a polynomial interpolation of the L
    prior values from the current timestamp, each powered to every value
    up to the degree `order` (cf. class's constructor), denoted as D:
        f_k(t, theta_k) =  sum_d^D sum_m^L theta_k[d*D+m] * val[t-m-1]^d

    The membership functions are Gaussian-like functions, defined as:
        rho_k(t, beta_k) = exp( -(t - beta_k[0])^2 / (2*beta_k[1]^2) )

    with `w_k` the weight of model `p`, as defined by:
        w_k(t, beta_k) = rho_k(t, beta_k) / (sum_j rho_j(t, beta_j))

    The modeling uses the `depth` prior values from the current time,
    and, if provided, the prediction is made `step` timestamps ahead.

    Constructor
    -----------
    __init__(order=1, level='local_beta', stop=100)

    Attributes
    ----------
    level : str, getter & setter
        The optimization level.
    regressors : np.ndarray, getter only
        The regressors' parameter vectors.
    membership : np.ndarray, getter only
        The membership functions' parameter vectors.

    Methods
    -------
    fit(clusters, dim, depth=2, step=0)
        Select the modeling function and train the multi-model.
    predict(inputs)
        Multi-Model estimate (local estimate + membership function).

    Examples
    --------
    >>> import numpy as np
    >>> import modeling.optimization as opt

    >>> class Cluster():
    ...     ''' Dummy `Cluster` class '''
    ...     def __init__(self, data, tstp):
    ...         self.value = data
    ...         self.index = tstp
    ...     def __getitem__(self, pos):
    ...         return self.value[pos]

    # Generate dummy data and wrap them into a set of Clusters
    >>> tstp = np.linspace(0, 10, 1000)
    >>> data = np.linspace(0, 100, 10000).reshape(1000, 10)
    >>> clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
    ...             for i in range(10)]

    >>> for method in ('local', 'local_theta', 'local_beta', 'global'):
    ...     # Instantiate the model
    ...     model = MultiModelPastWindow(order:=2, level=method, stop=100)

    ...     # Train the model
    ...     model.fit(clusters, dim:=0, depth:=2, step:=0)

    ...     # Use the model for prediction
    ...     values = clusters[2][50:100, dim]
    ...     estims = model.predict(values)
    ...     print(method, '\t', f"%.3e" %opt.mse(values[depth+step:], estims))
    local           2.702e-03
    local_theta     2.119e-01
    local_beta      2.365e-04
    global          1.174e-04
    """

    #---------------------   Initialization Functions   ---------------------#
    def _beta0(self, clusters, dim):
        """ Initial membership function parameters """
        bcu = np.empty((len(clusters), 2), float)
        for i, clt in enumerate(clusters):
            bcu[i] = clt[:, dim].mean(0), clt[:, dim].std(0)
        return bcu.ravel()

    def _matinfo(self, values, depth=2, beta=None):
        """ Information Matrix """

        # Global IM (sum of the products between local models and memberships)
        if isinstance(beta, (tuple, list, np.ndarray)):
            mod, pos = len(beta)//2, 0
            phi = np.empty((len(values)-depth, self._order*mod*depth), float)

            # All products between the local models and the memberships sum
            apt = np.zeros(len(values)-depth, float)
            den = np.zeros(len(values)-depth, float)
            for bcu in beta.reshape(-1, 2):                        # MF params
                for k in range(depth):                             # Past vals
                    apt = np.exp(-0.5*((values[k:k-depth]-bcu[0])/bcu[1])**2)
                    for j in range(1, self._order+1):              # Pow vals
                        phi[:, pos+j-1] = apt*values[k:k-depth]**j # Loc model
                    den += apt                                     # MF sum
                    pos += self._order

            # Denominator (sum of the memberships)
            den = depth / den
            for i in range(phi.shape[1]):
                phi[:, i] *= den

            return phi

        # Local Information Matrix (data only, no membership)
        phi = np.empty((len(values)-depth, self._order*depth), float)
        pos = 0
        for k in range(depth):
            for j in range(1, self._order+1):
                phi[:, pos+j-1] = values[k:k-depth]**j             # Loc model
            pos += self._order

        return phi
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Functions   -----------------------#
    def _memberships(self, values, beta, depth=2):
        """ Gaussian-like membership function """
        return np.exp(-0.5*((values[:-depth]-beta[0])/beta[1])**2)

    def _estimate(self, values, beta, theta):
        """ Multi-Model estimate (local estimate + membership function) """
        depth, pos = 2 * len(theta) // (len(beta) * self._order), 0
        acc = np.zeros(len(values)-depth, float)
        den = np.zeros(len(values)-depth, float)
        for bcu in beta.reshape(-1, 2):                     # Local MF params
            for k in range(depth):                          # Every past input
                apt = np.exp(-0.5*((values[k:k-depth]-bcu[0])/bcu[1])**2)
                for j in range(1, self._order+1):           # Powered inputs
                    tcu = theta[pos+j-1]                    # Model params
                    acc += apt*tcu*values[k:k-depth]**j     # Local model
                den += apt                                  # MF summation
                pos += self._order
        return depth * acc / den

    def predict(self, inputs):
        """ Multi-Model estimate (local estimate + membership function)

        Parameters
        ----------
        inputs : np.ndarray
            The past data, typically sorted in time order.

        Returns
        -------
        pred : np.ndarray
            The estimated data.

        Examples
        --------
        >>> import numpy as np
        >>> import modeling.optimization as opt

        >>> class Cluster():
        ...     ''' Dummy `Cluster` class '''
        ...     def __init__(self, data, tstp):
        ...         self.value = data
        ...         self.index = tstp
        ...     def __getitem__(self, pos):
        ...         return self.value[pos]

        # Generate dummy data and wrap them into a set of Clusters
        >>> tstp = np.linspace(0, 10, 1000)
        >>> data = np.linspace(0, 100, 10000).reshape(1000, 10)
        >>> clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
        ...             for i in range(10)]

        # Instantiate the model
        >>> model = MultiModelPastWindow(order:=2, level='local_beta', stop=100)

        # Train the model
        >>> model.fit(clusters, dim:=0, depth:=2, step:=0)

        # Use the model for prediction
        >>> values = clusters[2][50:100, dim]
        >>> estims = model.predict(values)
        >>> print(f"%.3e" %opt.mse(values[depth+step:], estims))
        2.365e-04
        """
        return self._estimate(inputs, self._membership, self._regressors)
    #------------------------------------------------------------------------#

    #-------------------   Local Modeling & Membership   --------------------#
    def _model_local(self, clusters, dim, depth=2, step=0):
        """ Local Modeling and Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes

        # Local models (one per cluster)
        tcu = np.empty((len(clusters), self._order*depth), float)
        for i, clt in enumerate(clusters):
            tcu[i] = opt.least_squares(
                self._matinfo(clt[:stop, dim], depth), clt[depth+step:, dim])
        tcu = tcu.ravel()

        # Initial membership parameters
        bcu = self._beta0(clusters, dim).reshape(-1, 2)

        # Local memberships (one per cluster)
        for i, clt in enumerate(clusters):
            bcu[i] = opt.gradient_descent_2nd(
                clt[:stop, dim], clt[depth+step:, dim], bcu[i], depth,
                festim=self._memberships, fcost=cost_func, method='gn')

        self._regressors, self._membership = tcu, bcu.ravel()
    #------------------------------------------------------------------------#

    #----------------   Local Modeling & Global Membership   ----------------#
    def _model_theta(self, clusters, dim, depth=2, step=0):
        """ Local Modeling and Global Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        data = nodes2data(clusters, dim)            # Sorted database

        # Local models (one per cluster)
        tcu = np.empty((len(clusters), self._order*depth), float)
        for i, clt in enumerate(clusters):
            tcu[i] = opt.least_squares(
                self._matinfo(clt[:stop, dim], depth), clt[depth+step:, dim])
        tcu = tcu.ravel()

        # Initial membership parameters
        bcu = self._beta0(clusters, dim)

        # Global memberships (on the whole database)
        tmp = opt.gradient_descent_2nd(
            data[:stop], data[depth+step:], bcu, tcu,
            festim=self._estimate, fcost=cost_func, method='gn')
        if (cost_func(data[depth+step:], self._estimate(data[:stop], tmp, tcu))
            < cost_func(data[depth+step:], self._estimate(data[:stop], bcu, tcu))):
            bcu = tmp

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #----------------   Global Modeling & Local Membership   ----------------#
    def _model_beta(self, clusters, dim, depth=2, step=0):
        """ Global Modeling and Local Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        data = nodes2data(clusters, dim)            # Sorted database

        # Initial membership parameters
        bcu0 = self._beta0(clusters, dim).reshape(-1, 2)

        # Local memberships
        bcu = np.empty((len(clusters), 2), float)
        for i, clt in enumerate(clusters):
            bcu[i] = opt.gradient_descent_2nd(
                clt[:stop, dim], clt[depth+step:, dim], bcu0[i], depth,
                festim=self._memberships, fcost=cost_func, method='gn')
        bcu = bcu.ravel()

        # Global models (on the whole database)
        tcu = opt.least_squares(
            self._matinfo(data[:stop], depth, bcu), data[depth+step:])

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-------------------   Global Modeling & Membership   -------------------#
    def _model_global(self, clusters, dim, depth=2, step=0):
        """ Global Modeling and Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        data = nodes2data(clusters, dim)            # Sorted database

        # Initial membership parameters
        bcu = self._beta0(clusters, dim)

        # Initial local modeling
        tcu = opt.least_squares(
            self._matinfo(data[:stop], depth, bcu), data[depth+step:])
        ccu = cost_func(data[depth+step:], self._estimate(data[:stop], bcu, tcu))

        # Multi-modeling the data
        for _ in range(self._stop):

            # Global membership (on the whole database)
            tmp = opt.gradient_descent_2nd(
                data[:stop], data[depth+step:], bcu, tcu,
                festim=self._estimate, fcost=cost_func, method='gn')
            cst = cost_func(
                data[depth+step:], self._estimate(data[:stop], tmp, tcu))
            if cst < ccu:                           # If smaller error,
                bcu, ccu = tmp, cst                 # update MF's params
            else:                                   # and current error
                break

            # Global modeling (on the whole database)
            tmp = opt.least_squares(
                self._matinfo(data[:stop], depth, bcu), data[depth+step:])
            cst = cost_func(
                data[depth+step:], self._estimate(data[:stop], bcu, tmp))
            if cst < ccu:                           # If smaller error,
                tcu, ccu = tmp, cst                 # update models' params
            else:                                   # and current error
                break

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, clusters, dim, depth=3, step=0):
        """ Select the modeling function and train the past-based multi-model

        Parameters
        ----------
        clusters : Database or Cluster (from the `clustering` package)
            The clusters with the indices and the data to model.
        dim : int
            The sensor to model, as its index in the clusters' data.
        [OPT] depth : int
            Depth for the local interpolated modeling (should be low).
                :Default: 3
        [OPT] step : int
            The prediction step (number of timestamps to predict).
            If set to 0, it is the value at the current time (modeling).
                :Default: 0 (modeling only)

        Returns
        -------
        None : directly set the `regressors` and `membership` attributes.

        Examples
        --------
        >>> import numpy as np

        >>> class Cluster():
        ...     ''' Dummy `Cluster` class '''
        ...     def __init__(self, data, tstp):
        ...         self.value = data
        ...         self.index = tstp
        ...     def __getitem__(self, pos):
        ...         return self.value[pos]

        # Generate dummy data and wrap them into a set of Clusters
        >>> tstp = np.linspace(0, 10, 1000)
        >>> data = np.linspace(0, 100, 10000).reshape(1000, 10)
        >>> clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
        ...             for i in range(10)]

        # Instantiate the model
        >>> model = MultiModelPastWindow(order:=2, level='local_beta', stop=100)

        # Train the model
        >>> model.fit(clusters, dim:=0, depth:=2, step:=0)
        >>> print(model.membership)
        [[ 6.74016395  4.53822384]
         [15.60048183  4.59131721]
         [25.42765112  6.16693475]
         [35.11986351  4.57634269]
         [45.12226531  5.06548946]
         [55.12466679  5.55463638]
         [65.12706834  6.04378307]
         [75.12947066  6.53293008]
         [85.13187127  7.02207631]
         [95.13427341  7.51122272]]
        """
        self._fit(clusters, dim, depth, step)
    #------------------------------------------------------------------------#

##############################################################################
