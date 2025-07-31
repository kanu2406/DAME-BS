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

Real Data : MIMIC-III 
---------------------


.. automodule:: experiments.real_data_experiments.mimic.preprocess
   :members:
   :show-inheritance:
   :undoc-members:


.. automodule:: experiments.real_data_experiments.mimic.run_experiment_mimic
   :members:
   :show-inheritance:
   :undoc-members:

Real Data : Stock Prices 
------------------------


.. automodule:: experiments.real_data_experiments.stocks_data.preprocess
   :members:
   :show-inheritance:
   :undoc-members:


.. automodule:: experiments.real_data_experiments.stocks_data.run_experiment_stock_prices
   :members:
   :show-inheritance:
   :undoc-members:

Real Data : GLOBEM
------------------

.. automodule:: experiments.real_data_experiments.globem.preprocess
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: experiments.real_data_experiments.globem.run_experiment_sleep
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: experiments.real_data_experiments.globem.run_experiment_steps
   :members:
   :show-inheritance:
   :undoc-members:


Results
=======

This section presents a comparison between the dame_bs algorithm and Kent's algorithm in both
the univariate and multivariate case. We generated datasets from three distributions—
Normal, Uniform, and Binomial—with a mean of 0.1. All data points and the true mean were then 
scaled to the range [-1, 1]. The comparison was based on the mean squared error between the 
estimated mean and the scaled true mean across 200 trials.

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

Mean Squared Error vs d (dimensionality of each sample) for the different distributions.

.. image:: ../figures/mse_vs_d.png
   :alt: MSE vs d for the different distributions
   :align: center
   :width: 600px

Real World Data 
---------------


Stock Prices
------------

We conducted experiments using a mean estimation algorithm to estimate the average price of stock data. 
In our setup, each stock was considered as a separate user, and its price history served as the sample data. 
For each stock, we used 249 data points and compared the performance of Kent's algorithm with DAME-BS. 
The results below show the computation time and mean squared errors, both for scaled prices within the 
range [-1, 1] and for the actual price scale.

.. image:: ../figures/mimic_result.png
   :alt: MSE for DAME-BS and Kent
   :align: center
   :width: 600px





MIMIC-III ('Medical Information Mart for Intensive Care')
---------------------------------------------------------

We conducted this experiment using MIMIC-III Dataset which consists of comprehensive clinical data of critical 
care admissions from 2001-2012 (Dataset : `<https://www.kaggle.com/datasets/asjad99/mimiciii>`_). We used heart rate of patients over time. 
We could only find data for 48 patients with 11 samples per user. We conducted mean estimation using
DAME-BS and Kent's algorithm 500 times and report MSE (for both scaled and unscaled values)
and average time taken by both algorithms. The results were limited in quality due to the small nummber of users 
and low number of samples per user.

.. image:: ../figures/Stocks_Data_Result.png
   :alt: MSE for DAME-BS and Kent
   :align: center
   :width: 600px


GLOBEM Dataset
--------------

We conducted this experiment using GLOBEM Dataset which consists of data from a mobile phone and 
a wearable fitness tracker 24×7, including Location, PhoneUsage, Call, Bluetooth, PhysicalActivity, 
and Sleep behavior. The datasets capture various aspects of participants' life experiences, 
such as general behavior patterns, the weekly routine cycle, the impact of COVID (Year3, 2020), and 
the gradual recovery after COVID (Year4, 2021) (Dataset : `<https://github.com/UW-EXP/GLOBEM/tree/main/data_raw>`_).
We used steps from four segments of the day and total sleep duration per day for each user.
For both cases, we conducted mean estimation using DAME-BS and Kent's algorithm 500 times and report MSE (for both scaled and unscaled values)
and average time taken by both algorithms. The results were limited in quality due to the small nummber of users 
and low number of samples per user.


For steps data, we had a total of 40 user with 88 samples per user. Here are the results for steps data - 


.. image:: ../figures/steps_result.png
   :alt: MSE for DAME-BS and Kent
   :align: center
   :width: 600px


For sleep data, we had a total of 40 user with 15 samples per user. Here are the results for sleep data - 


.. image:: ../figures/sleep_result.png
   :alt: MSE for DAME-BS and Kent
   :align: center
   :width: 600px