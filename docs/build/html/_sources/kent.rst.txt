Kent's Algorithm 
============

- The **kent** package implements Kent's locally differentially private mean estimation algorithms,including both univariate and multivariate variants. 
- The **`kent.kent`** module provides the univariate LDP mean estimator based on interval partitioning and private voting. 
- The **`kent.multivariate_kent`** module extends the univariate Kent mechanism to multivariate seeting where all data points are in l_inf ball of radius 1 i.e. inside [-1,1]^d



Submodules
----------

kent.kent module
----------------

.. automodule:: kent.kent
   :members:
   :show-inheritance:
   :undoc-members:

kent.multivariate\_kent module
------------------------------

.. automodule:: kent.multivariate_kent
   :members:
   :show-inheritance:
   :undoc-members:

