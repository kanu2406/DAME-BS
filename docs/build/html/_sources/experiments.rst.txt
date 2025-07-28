Experiments
===========

This section presents experiments done to compare the DAME-BS algorithm and Kent's algorithm for both univariate and multivariate cases.


Univariate Experiments
----------------------

.. automodule:: experiments.univariate_experiment
   :members:
   :show-inheritance:
   :undoc-members:


Multivariate Experiments 
------------------------

.. automodule:: experiments.multivariate_experiment
   :members:
   :show-inheritance:
   :undoc-members:


Results
-------
Mean Squared Error vs Alpha for the different distributions.

.. image:: ../figures/risk_vs_alpha.png
   :alt: Mean Squared Error vs Alpha for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs n (total number of users) for the different distributions.


.. image:: ../figures/risk_vs_n.png
   :alt: Mean Squared Error vs n (total number of users) for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs delta (tolerated failure probability of Binary Search) for the different distributions.


.. image:: ../figures/risk_vs_delta.png
   :alt: Mean Squared Error vs delta (tolerated failure probability of Binary Search) for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs n (total number of users) for the different distributions for different values of privacy parameter alpha.


.. image:: ../figures/risk_vs_n_diff_alpha.png
   :alt: Mean Squared Error vs n (total number of users) for the different distributions
   :align: center
   :width: 600px
