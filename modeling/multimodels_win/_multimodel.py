""" Base model for building Multi-Models

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2022
Last revised: April 2026

License: GPLv3
"""

__all__ = ['_MultiModel']


##############################################################################
##                         Multimodel Parent Class                          ##
##############################################################################

class _MultiModel():
    """ Base class for the Multimodel using Time or Past data """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, order=1, level='local_beta', stop=100):
        """ Instantiate a MultiModelTimeWindow or MultiModelPastWindow object

        Parameters
        ----------
        [OPT] order : int
            The maximal power to which the inputs are raised to. The
            estimation is a linear combination of the values raised
            to every power up to 'order'.
                :Default: 1
        [OPT] level : str
            The modeling level: local, semi-global or global:
              - 'local':       both theta and beta local;
              - 'local_theta': theta local and beta global;
              - 'local_beta':  theta global and beta local;
              - 'global':      both theta and beta global.
            The last one often gives the best results, but also is the
            slowest; the 'local_beta' variant is often a good trade-off
            between accuracy and execution speediness.
                :Default: 'local_beta'
        [OPT] stop : int
            Maximal number of iterations for iterative methods.
                :Default: 100

        Examples
        --------
        # Order 3 multi-model with local optimizations
        >>> mmodel = _MultiModel(3, 'local', 100)

        # Order 3 multi-model with global optimizations
        >>> mmodel = _MultiModel(3, 'global', 100)
        """
        self._stop = stop                   # Stopping criterion
        self._level = level.lower()         # Level of the training
        self._order = order                 # Max power for input values
        self._regressors = None             # Local models coefficients
        self._membership = None             # Membership funcs. parameters
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def level(self):
        """ Get the optimization level """
        return self._level

    @level.setter
    def level(self, level):
        """ Set the optimization level """
        self._level = level.lower()

    @property
    def regressors(self):
        """ Get the coefficients of the regressors """
        return self._regressors.reshape(
            -1, 2 * len(self._regressors) // len(self._membership))

    @property
    def membership(self):
        """ Get the coefficients of the membership functions """
        return self._membership.reshape(-1, 2)
    #------------------------------------------------------------------------#

    #-------------------   (Virtual) Modeling Functions   -------------------#
    def _model_local(self, clusters, dim, depth, step):
        """ Local Modeling and Membership -- θ(t) & β(t) """
        raise NotImplementedError

    def _model_theta(self, clusters, dim, depth, step):
        """ Local Modeling and Global Membership -- θ(t) & β(t, θ) """
        raise NotImplementedError

    def _model_beta(self, clusters, dim, depth, step):
        """ Global Modeling and Local Membership -- θ(t, β) & β(t) """
        raise NotImplementedError

    def _model_global(self, clusters, dim, depth, step):
        """ Global Modeling and Membership -- θ(t, β) & β(t, θ) """
        raise NotImplementedError
    #------------------------------------------------------------------------#

    #-------------------   MM-Windowed Variant Training   -------------------#
    def _fit(self, clusters, dim, depth=2, step=0):
        """ Train the MM using full windowed datasets
        (cf. `fit` method from `MultiModelPastWindow` class) """

        if self._level == 'local':              # Both theta and beta local
            self._model_local(clusters, dim, depth, step)

        elif self._level == 'local_theta':      # Theta local and beta global
            self._model_theta(clusters, dim, depth, step)

        elif self._level == 'local_beta':       # Theta global and beta local
            self._model_beta(clusters, dim, depth, step)

        elif self._level == 'global':           # Both theta and beta global
            self._model_global(clusters, dim, depth, step)
    #------------------------------------------------------------------------#

##############################################################################
