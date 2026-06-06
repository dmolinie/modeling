Installing the Modeling package
+++++++++++++++++++++++++++++++

.. Contents::


Prerequisites
=============

The ``modeling`` package requires the following software:

1) **Python** 3.8.x or newer.

**Python** https://www.python.org/


Installation
============

Except the optional packages specified below, the ``modeling`` project relies on standard Python libraries only. The project's modules that depend on packages that are not installed on the system will not appear as tab-completion options (but may still be accessed via direct call).


Basic Installation
------------------

To install the ``modeling`` package, when in the installation folder, run the following command:

.. code-block:: sh

   pip install .

This will compile the package's sources and install them into the active environment (e.g. "site-packages").


Additional Packages
-------------------

The package can use Machine Learning models from the ``scikit-learn`` suit in its multi-models; to install this package, run the command:

.. code-block:: sh

   pip install .[sklearn]

This will download and install the ``scikit-learn`` suit in addition to the current ``modeling`` package.
