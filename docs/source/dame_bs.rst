dame\_bs package
================

The **dame_bs** package provides a set of functions for private mean estimation under user level 
Local Differential Privacy (LDP) using the algorithm called DAME-BS (Distribution-Aware Mean Estimation
under User-level Local Differential Privacy using Binary Search) for univariate and multivariate cases. 
The following functions are implemented - 

- A binary-search based subroutine (`dame_bs.binary_search`) that is responsible for the localization phase and returns the interval that contains estimated private mean.  
- The core function of DAME-BS algorithm which combine localization and estimation phase and returns final estimated mean for univariate data (`dame_bs.dame_bs`).  
- This is an extension of DAME-BS algorithm to multivariate data where data points are in l_inf ball of radius 1. (`dame_bs.multivariate_dame_bs`).  
- Utility functions for calculating theoretical upper bound on expected squared error and plotting errorbar. (`dame_bs.utils`).


dame\_bs.binary\_search 
-----------------------

.. automodule:: dame_bs.binary_search
   :members:
   :show-inheritance:
   :undoc-members:

dame\_bs.dame\_bs 
-----------------

.. automodule:: dame_bs.dame_bs
   :members:
   :show-inheritance:
   :undoc-members:

dame\_bs.multivariate\_dame\_bs
-------------------------------

.. automodule:: dame_bs.multivariate_dame_bs
   :members:
   :show-inheritance:
   :undoc-members:

dame\_bs.utils 
--------------

.. automodule:: dame_bs.utils
   :members:
   :show-inheritance:
   :undoc-members:

