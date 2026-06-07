Modeling package's modules, functions & classes
+++++++++++++++++++++++++++++++++++++++++++++++

This file aims to summarize all the objects implemented in the ``modeling`` package; to this purpose, it briefly introduces the modules of the package, as well as their respective functions and classes.

.. Contents::


interpolators
=============
Classes to model a time series: generic polynomial interpolator, past-based regression matrix model, past-based multi-model, that is a multi-model generalization (with a membership strategy) of the latter.

..
  _interpol.py.

* ``Interpolator``  
		Polynomial/Sinusoidal Interpolation class.

..
  _interpol_past.py.

* ``InterpolatorPast``  
		Regression Matrix-based past-values Model class.

* ``MultiModelPast``  
		Regression Matrix-based past-values Multi-Model class.


multimodels
===========
Classes to build a multi-model on time series, that is a network that connects several smaller models, typically trained on specific parts of the feature space. Provides both a Gaussian membership Multi-Model and the more regular Gating Networks. Also provides some functions to shape data in the format expected by the multi-models (in particular, their regression matrices).

..
  _matreg.py.

* ``check_oob_index``  
		Remove the out-of-bounds indexes.

* ``matreg``  
		Build the regression matrix.

* ``local_matreg``  
		Build the local regression matrices from the clusters' data.

* ``build_inps_outs``  
		Build the matrices of inputs & outputs.

..
  _gating_nets.py.

* ``GatingNetworks``  
		Gating Networks Multi-Model class.

..
  _multimodels.py.

* ``MultiModel``  
		Membership-based Multi-Model class.


multimodels_win
===============
Simpler, windowed variant of the ``multimodels`` module: here, the data are assumed to be continuous (no data leap) and are interpolated using a sort of sliding window. These models are more restrictive and much more specific than those of ``multimodels``. One model uses the time stamps as variable, and the second uses the past data.

..
  _mm_time.py.

* ``nodes2times``  
		Nodes to full sorted database (tstp + data).

* ``MultiModelTimeWindow``  
		Windowed variant of the time-based Multi-Model.

..
  _mm_past.py.

* ``nodes2data``  
		Nodes to full sorted database (data).

* ``MultiModelPastWindow``  
		Windowed variant of the past values-based Multi-Model class.


optimization
============
Functions and classes to perform first and second order optimizations: (variants of) Gradient Descent, Newton, Gauss-Newton & Levenberg-Marquardt methods, etc. Also provides estimation functions (that can serve as interpolators) and the MSE cost function.

..
  _derivative.py.

* ``matinv``  
		Invert a matrix possibly containing all-zeros rows/cols.

* ``Derivative``  
		Class to compute derivative & gradient.

* ``Hessian``  
		Class to build the Hessian Matrix.

..
  _optim.py.

* ``mse``  
		Quadratic error cost function.

* ``linear``  
		Linear (polynomial) estimation function.

* ``sinus``  
		Sinus estimation function.

* ``gaussian``  
		Gaussian estimation function.

* ``backtracking``  
		Backtracking line search.

* ``least_squares``  
		Ordinary/Recursive Least Squares (OLS/RLS).

* ``gradient_descent_1st``  
		First order Gradient Descent optimization.

* ``gradient_descent_2nd``  
		Second order, Hessian-based nonlinear Gradient Descent optimization.


rbf_nets
========
Class that implements Radial Basis Function (RBF) Networks, simple multi-models that use radial functions as local models. Also provides generic radial kernels for the RBF. Much simpler and faster, but still accurate, than the models provided in ``multimodels``, but more specialized.

..
  _kernels.py.

* ``linear``  
		Linear RBF (y(r) = -r).

* ``thin_plate_spline``  
		Thin plate spline RBF (y(r) = r² * log(r)).

* ``cubic``  
		Cubic RBF (y(r) = -r³).

* ``quintic``  
		Quintic RBF (y(r) = -r⁵).

* ``multiquadric``  
		Multiquadratic RBF (y(r) = -√(1+r²).

* ``inverse_quadratic``  
		Inverse Quadratic RBF (y(r) = 1/(1+r²)).

* ``inverse_multiquadric``  
		Inverse Multiquadratic RBF (y(r) = 1/√(1+r²).

* ``gaussian``  
		Gaussian RBF (y(r) = -exp(-r²)).

* ``get_ker_func``  
		Get the reference to the kernel function.

..
  _rbf_net.py.

* ``RBFNet``  
		Radial Basis Function (RBF) Networks class.

