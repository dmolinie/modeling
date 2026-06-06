""" Multi-Models using time stamps as variable

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2022
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['nodes2times', 'MultiModelTimeWindow']

import numpy as np

import modeling.optimization._optimizers as opt
from . _multimodel import _MultiModel

# Default cost function
cost_func = opt.mse


##############################################################################
##                         Database reconstruction                          ##
##############################################################################

def nodes2times(clusters, dim):
    """ Nodes to full sorted database (tstp + data)

    Take a set of clusters and extract their data. First, retrieve the
    timestamps of every cluster and stack them all into a unique array,
    and sort this array in ascending order. Then, retrieve the data of
    every cluster and stack them all into a unique array before sorting
    them in the order of the timestamps previously built. Return the
    sorted timestamps array and the `dim` dimension of the sorted data.

    Parameters
    ----------
    clusters : set of Clusters (from the `clustering` package)
        The clusters for which to extract the data.
    dim : int
        The dimension of the data to extract.

    Returns
    -------
    tstp : np.ndarray
        The data timestamps.
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

    # Rebuild the timestamps and data from the set of Clusters
    >>> tstp2, data2 = nodes2times(clusters, 1)
    """
    tstp = np.hstack([node.index for node in clusters])         # All indexes
    idx = tstp.argsort()                                        # Sorted depth
    return tstp[idx], np.vstack([k.value for k in clusters])[idx, dim]

##############################################################################



##############################################################################
##                          Time-based Multi-Model                          ##
##############################################################################

class MultiModelTimeWindow(_MultiModel):
    r""" Windowed variant of the time-based Multi-Model

    Take a set of clusters and build the multi-model by modeling every
    local group of data, before connecting them all in a linear fashion.
    By denoting `val` a vector of inputs, the multi-model is defined as
    a linear combination of the estimates outputted by any of the K local
    models, each multiplied by a Gaussian-like membership function (MF):
        ŷ(t, theta, beta) = sum_k^K w_k(t, beta_k) * f_k(t, theta_k)

    where `beta` and `theta` are the sets of the K vectors `beta_k` and
    `theta_k`, each being the parameter vector of the membership function
    and local model, respectively, of the k-th cluster.

    The local models are defined as a polynomial interpolation of the
    weighted current timestamp raised to every power up to `depth`:
        f_k(t, theta_k) = sum_d^D theta_k[d] * t^d

    The membership functions are Gaussian-like functions, defined as:
        rho_k(t, beta_k) = exp( -(t - beta_k[0])^2 / (2*beta_k[1]^2) )

    with `w_k` the weight of model `p`, as defined by:
        w_k(t, beta_k) = rho_k(t, beta_k) / (sum_j rho_j(t, beta_j))

    If provided, the prediction is made `step` timestamps ahead.

    Constructor
    -----------
    __init__(order=1, level='local_beta', stop=100)

    Attributes
    ----------
    level : str, getter & setter
        The otimization level.
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

    >>> for level in ('local', 'local_theta', 'local_beta', 'global'):
    ...     # Instantiate the model
    ...     model = MultiModelTimeWindow(order:=2, level=level, stop=100)

    ...     # Train the model
    ...     model.fit(clusters, dim:=0, depth:=2, step:=0)

    ...     # Use the model for prediction
    ...     stamps = clusters[2].index[50:100]
    ...     values = clusters[2][50:100, dim]
    ...     estims = model.predict(stamps)
    ...     print(level, '\t', f"%.3e" %opt.mse(values, estims))
    local           4.367e-02
    local_theta     1.791e-03
    local_beta      1.830e-02
    global          3.304e-02
    """

    #---------------------   Initialization Functions   ---------------------#
    def _beta0(self, clusters):
        """ Initial membership function parameters """
        bcu = np.empty((len(clusters), 2), float)
        for i, clt in enumerate(clusters):
            bcu[i] = clt.index.mean(), clt.index.std()
        return bcu.ravel()

    def _matinfo(self, tstp, depth=2, beta=None):
        """ Information Matrix """

        # Global IM (sum of the products between local models and memberships)
        if isinstance(beta, (tuple, list, np.ndarray)):
            mod = len(beta) // 2                              # Nb loc models
            phi = np.empty((len(tstp), mod*(depth+1)), float) # Info Matrix

            # All products between the local models and the memberships sum
            pos, apt, den = 0, 0., 0.
            for bcu in beta.reshape(-1, 2):
                apt = np.exp(-0.5*((tstp-bcu[0])/bcu[1])**2)  # Membership
                phi[:, pos] = apt                             # 1st coef is 1
                for k in range(1, depth+1):
                    phi[:, pos+k] = apt * tstp**k             # Local Model
                den += apt                                    # MF summation
                pos += depth+1

            # Denominator (sum of the memberships)
            den = 1/den
            for i in range(phi.shape[1]):
                phi[:, i] *= den

            return phi

        # Local Information Matrix (tstp only, no membership)
        phi = np.ones((len(tstp), depth+1), float)            # Info Matrix
        for k in range(1, depth+1):
            phi[:, k] = tstp**k                               # Local Model

        return phi
    #------------------------------------------------------------------------#

    #-----------------------   Estimation Functions   -----------------------#
    def _memberships(self, tstp, beta, *args):
        """ Gaussian-like membership function """
        # pylint: disable=unused-argument
        return np.exp(-0.5*((tstp-beta[0])/beta[1])**2)

    def _estimate(self, tstp, beta, theta):
        """ Multi-Model estimate (local estimate + membership function) """
        dim, pos = 2 * len(theta) // len(beta), 0
        acc = np.zeros(len(tstp), float)
        den = np.zeros(len(tstp), float)
        for bcu in beta.reshape(-1, 2):             # Local MF params
            tcu = theta[pos:pos+dim]                # Local model params
            est = tcu[0] + sum(p*tstp**(k+1) for k, p in enumerate(tcu[1:]))
            apt = np.exp(-0.5*((tstp-bcu[0])/bcu[1])**2)    # Membership
            acc += np.prod((est, apt), 0)                   # Weighted model
            den += apt                                      # MF summation
            pos += dim
        return acc / den

    def predict(self, tstp):
        """ Multi-Model estimate (local estimate + membership function)

        Parameters
        ----------
        tstp : np.ndarray
            The timestamps to use for regression & prediction.

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
        >>> model = MultiModelTimeWindow(order:=2, level='local_beta', stop=100)

        # Train the model
        >>> model.fit(clusters, dim:=0, depth:=2, step:=0)

        # Use the model for prediction
        >>> stamps = clusters[2].index[50:100]
        >>> values = clusters[2][50:100, dim]
        >>> estims = model.predict(stamps)
        >>> print(f"%.3e" %opt.mse(values, estims))
        1.830e-02
        """
        return self._estimate(tstp, self._membership, self._regressors)
    #------------------------------------------------------------------------#

    #-------------------   Local Modeling & Membership   --------------------#
    def _model_local(self, clusters, dim, depth=2, step=0):
        """ Local Modeling and Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes

        # Local models (one per cluster)
        tcu = np.empty((len(clusters), depth+1), float)
        for i, clt in enumerate(clusters):
            tcu[i] = opt.least_squares(
                self._matinfo(clt.index[:stop], depth), clt[step:, dim])
        tcu = tcu.ravel()

        # Initial membership parameters
        bcu = self._beta0(clusters).reshape(-1, 2)

        # Local memberships (one per cluster)
        for i, clt in enumerate(clusters):
            bcu[i] = opt.gradient_descent_2nd(
                clt.index[:stop], clt[step:, dim], bcu[i], None,
                festim=self._memberships, fcost=cost_func, method='gn')

        self._regressors, self._membership = tcu, bcu.ravel()
    #------------------------------------------------------------------------#

    #----------------   Local Modeling & Global Membership   ----------------#
    def _model_theta(self, clusters, dim, depth=2, step=0):
        """ Local Modeling and Global Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        tstp, data = nodes2times(clusters, dim)     # Sorted database
        tstp, data = tstp[:stop], data[step:]

        # Local models (one per cluster)
        tcu = np.empty((len(clusters), depth+1), float)
        for i, clt in enumerate(clusters):
            tcu[i] = opt.least_squares(
                self._matinfo(clt.index[:stop], depth), clt[step:, dim])
        tcu = tcu.ravel()

        # Initial membership parameters
        bcu = self._beta0(clusters)

        # Global memberships (on the whole database)
        tmp = opt.gradient_descent_2nd(
            tstp, data, bcu, tcu,
            festim=self._estimate, fcost=cost_func, method='gn')
        if (cost_func(data, self._estimate(tstp, tmp, tcu))
            < cost_func(data, self._estimate(tstp, bcu, tcu))):
            bcu = tmp

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #----------------   Global Modeling & Local Membership   ----------------#
    def _model_beta(self, clusters, dim, depth=2, step=0):
        """ Global Modeling and Local Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        tstp, data = nodes2times(clusters, dim)     # Sorted database

        # Initial membership parameters
        bcu0 = self._beta0(clusters).reshape(-1, 2)

        # Local memberships (one per cluster)
        bcu = np.empty((len(clusters), 2), float)
        for i, clt in enumerate(clusters):
            bcu[i] = opt.gradient_descent_2nd(
                clt.index[:stop], clt.value[step:, dim], bcu0[i], None,
                festim=self._memberships, fcost=cost_func, method='gn')
        bcu = bcu.ravel()

        # Global models (on the whole database)
        tcu = opt.least_squares(
            self._matinfo(tstp[:stop], depth, bcu), data[step:])

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-------------------   Global Modeling & Membership   -------------------#
    def _model_global(self, clusters, dim, depth=2, step=0):
        """ Global Modeling and Membership (cf. `fit` method) """

        stop = [-step, None][step == 0]             # All non-zero indexes
        tstp, data = nodes2times(clusters, dim)     # Sorted database
        tstp, data = tstp[:stop], data[step:]

        # Initial membership parameters
        bcu = self._beta0(clusters)

        # Initial local modeling
        tcu = opt.least_squares(self._matinfo(tstp, depth, bcu), data)
        ccu = cost_func(data, self._estimate(tstp, bcu, tcu))

        # Multi-modeling the data
        for _ in range(self._stop):

            # Global membership (on the whole database)
            tmp = opt.gradient_descent_2nd(
                tstp, data, bcu, tcu,
                festim=self._estimate, fcost=cost_func, method='gn')
            cst = cost_func(data, self._estimate(tstp, tmp, tcu))
            if cst < ccu:                           # If smaller error,
                bcu, ccu = tmp, cst                 # update MF's params
            else:                                   # and current error
                break

            # Global modeling (on the whole database)
            tmp = opt.least_squares(self._matinfo(tstp, depth, bcu), data)
            cst = cost_func(data, self._estimate(tstp, bcu, tmp))
            if cst < ccu:                           # If smaller error,
                tcu, ccu = tmp, cst                 # update models' params
            else:                                   # and current error
                break

        self._regressors, self._membership = tcu, bcu
    #------------------------------------------------------------------------#

    #-----------------------   Multi-Model Training   -----------------------#
    def fit(self, clusters, dim, depth=3, step=0):
        """ Select the modeling function and train the time-based multi-model

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
        >>> model = MultiModelTimeWindow(order:=2, level='local_beta', stop=100)

        # Train the model
        >>> model.fit(clusters, dim:=0, depth:=2, step:=0)
        >>> print(model.membership)
        [[0.70368526 0.57304328]
         [1.80703384 1.53604251]
         [2.94383101 3.15689173]
         [3.70365461 2.1572648 ]
         [4.56019249 1.05363563]
         [5.57249381 1.38321081]
         [6.58694887 1.80352463]
         [7.60372473 2.33293011]
         [8.62299126 2.99148212]
         [9.64492053 3.80095046]]
        """
        self._fit(clusters, dim, depth, step)
    #------------------------------------------------------------------------#

##############################################################################
