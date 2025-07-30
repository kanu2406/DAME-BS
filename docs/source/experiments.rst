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
=======

This section presents a comparison between the dame_bs algorithm and Kent's algorithm in both
the univariate and multivariate case. We generated datasets from three distributions—
Normal, Uniform, and Binomial—with a mean of 0.1. All data points and the true mean were then 
scaled to the range [-1, 1]. The comparison was based on the mean squared error between the 
estimated mean and the scaled true mean across 50 trials.

Univariate case
---------------

Mean Squared Error vs privacy parameter alpha for the different distributions.

.. image:: ../figures/mse_vs_alpha_univariate.png
   :alt: Mean Squared Error vs Alpha for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs n (total number of users) for the different distributions.

.. image:: ../figures/mse_vs_n_univariate.png
   :alt: Mean Squared Error vs n (total number of users) for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs m (number of samples per user) for the different distributions.

.. image:: ../figures/mse_vs_m_univariate.png
   :alt: MSE vs m for the different distributions
   :align: center
   :width: 600px

Multivariate case
-----------------

Mean Squared Error vs privacy parameter alpha for the different distributions.

.. image:: ../figures/mse_vs_alpha_multivariate.png
   :alt: Mean Squared Error vs Alpha for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs n (total number of users) for the different distributions.

.. image:: ../figures/mse_vs_n_multivariate.png
   :alt: Mean Squared Error vs n (total number of users) for the different distributions
   :align: center
   :width: 600px

Mean Squared Error vs m (number of samples per user) for the different distributions.

.. image:: ../figures/mse_vs_m_multivariate.png
   :alt: MSE vs m for the different distributions
   :align: center
   :width: 600px

Real World Data 
---------------

We conducted experiments using a mean estimation algorithm to estimate the average price of stock data. 
In our setup, each stock was considered as a separate user, and its price history served as the sample data. 
For each stock, we used 249 data points and compared the performance of Kent's algorithm with DAME-BS. 
The results below show the computation time and mean squared errors, both for scaled prices within the 
range [-1, 1] and for the actual price scale.

.. image:: ../figures/Stocks_Data_Result.png
   :alt: MSE for DAME-BS and Kent
   :align: center
   :width: 600px