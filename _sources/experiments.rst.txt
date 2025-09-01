Experiments
===========

This section presents experiments done to compare the DAME-BS algorithm, Kent's algorithm and Girgis' algorithm for both univariate and multivariate cases.


Univariate Experiments
----------------------

.. automodule:: experiments.synthetic_data_experiments.univariate_experiment
   :members:
   :show-inheritance:
   :undoc-members:


Multivariate Experiments 
------------------------

.. automodule:: experiments.synthetic_data_experiments.multivariate_experiment
   :members:
   :show-inheritance:
   :undoc-members:

Real Data : MIMIC-III 
---------------------


.. automodule:: experiments.real_data_experiments.mimic.preprocess
   :members:
   :show-inheritance:
   :undoc-members:


.. automodule:: experiments.real_data_experiments.mimic.run_mimic_experiment
   :members:
   :show-inheritance:
   :undoc-members:

Real Data : Stock Prices 
------------------------


.. automodule:: experiments.real_data_experiments.stocks_data.preprocess
   :members:
   :show-inheritance:
   :undoc-members:


.. automodule:: experiments.real_data_experiments.stocks_data.run_stocks_experiment
   :members:
   :show-inheritance:
   :undoc-members:

Real Data : GLOBEM
------------------

.. automodule:: experiments.real_data_experiments.globem.preprocess
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: experiments.real_data_experiments.globem.run_sleep_experiment
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: experiments.real_data_experiments.globem.run_steps_experiment
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


.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_alpha_normal.png
         :width: 250px
         :alt: MSE vs alpha
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_alpha_uniform.png
         :width: 250px
         :alt: MSE vs alpha
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_alpha_binomial.png
         :width: 250px
         :alt: MSE vs alpha
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_alpha_standard_t.png
         :width: 250px
         :alt: MSE vs alpha

Mean Squared Error vs n (total number of users) for the different distributions.


.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_n_normal.png
         :width: 250px
         :alt: MSE vs n
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_n_uniform.png
         :width: 250px
         :alt: MSE vs n
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_n_binomial.png
         :width: 250px
         :alt: MSE vs n
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_n_standard_t.png
         :width: 250px
         :alt: MSE vs n

Mean Squared Error vs m (number of samples per user) for the different distributions.


.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_m_normal.png
         :width: 250px
         :alt: MSE vs m
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_m_uniform.png
         :width: 250px
         :alt: MSE vs m
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_m_binomial.png
         :width: 250px
         :alt: MSE vs m
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_m_standard_t.png
         :width: 250px
         :alt: MSE vs m


Multivariate case
-----------------

Mean Squared Error vs privacy parameter alpha for the different distributions.


.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_alpha_normal.png
         :width: 250px
         :alt: MSE vs alpha
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_alpha_uniform.png
         :width: 250px
         :alt: MSE vs alpha
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_alpha_binomial.png
         :width: 250px
         :alt: MSE vs alpha
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_alpha_standard_t.png
         :width: 250px
         :alt: MSE vs alpha

Mean Squared Error vs n (total number of users) for the different distributions.

.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_n_normal.png
         :width: 250px
         :alt: MSE vs n
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_n_uniform.png
         :width: 250px
         :alt: MSE vs n
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_n_binomial.png
         :width: 250px
         :alt: MSE vs n
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_n_standard_t.png
         :width: 250px
         :alt: MSE vs n

Mean Squared Error vs m (number of samples per user) for the different distributions.


.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_m_normal.png
         :width: 250px
         :alt: MSE vs m
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_m_uniform.png
         :width: 250px
         :alt: MSE vs m
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_m_binomial.png
         :width: 250px
         :alt: MSE vs m
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_m_standard_t.png
         :width: 250px
         :alt: MSE vs m


Mean Squared Error vs d (dimensionality of each sample) for the different distributions.

.. list-table::
   :header-rows: 0
   :align: center

   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_d_normal.png
         :width: 250px
         :alt: MSE vs d
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_d_uniform.png
         :width: 250px
         :alt: MSE vs d
   * - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_d_binomial.png
         :width: 250px
         :alt: MSE vs d
     - .. image:: ../../experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_d_standard_t.png
         :width: 250px
         :alt: MSE vs d



Real World Data 
---------------


Stock Prices
------------

We conducted experiments using a mean estimation algorithm to estimate the average price of stock data. 
In our setup, each stock was considered as a separate user, and its price history served as the sample data. 
For each stock, we used 249 data points and compared the performance of Kent's algorithm and Girgis' algorithm with DAME-BS. 
The results below show the computation time and median squared errors, both for scaled prices within the 
range [-1, 1] and for the actual price scale.


.. raw:: html

   <div style="margin: 1em 0; text-align: center;">

.. image:: ../../experiments/real_data_experiments/stocks_data/results/stocks_data.png
   :alt: MSE for DAME-BS, Kent and Girgis
   :align: center
   :width: 500px

.. raw:: html

   <div style="margin: 1em 0; text-align: center;">



MIMIC-III ('Medical Information Mart for Intensive Care')
---------------------------------------------------------

We conducted this experiment using MIMIC-III Dataset which consists of comprehensive clinical data of critical 
care admissions from 2001-2012 (Dataset : `<https://www.kaggle.com/datasets/asjad99/mimiciii>`_). We used heart rate of patients over time. 
We could only find data for 48 patients with 11 samples per user. We conducted mean estimation using
DAME-BS, Kent's and Girgis' algorithm 500 times and report median squared error. (for both scaled and unscaled values)
and average time taken by both algorithms. The results were limited in quality due to the small nummber of users 
and low number of samples per user.

.. raw:: html

   <div style="margin: 1em 0; text-align: center;">

.. image:: ../../experiments/real_data_experiments/mimic/results/mimic_data.png
   :alt: for DAME-BS, Kent and Girgis
   :align: center
   :width: 500px

.. raw:: html

   <div style="margin: 1em 0; text-align: center;">


GLOBEM Dataset
--------------

We conducted this experiment using GLOBEM Dataset which consists of data from a mobile phone and 
a wearable fitness tracker 24×7, including Location, PhoneUsage, Call, Bluetooth, PhysicalActivity, 
and Sleep behavior. The datasets capture various aspects of participants' life experiences, 
such as general behavior patterns, the weekly routine cycle, the impact of COVID (Year3, 2020), and 
the gradual recovery after COVID (Year4, 2021) (Dataset : `<https://github.com/UW-EXP/GLOBEM/tree/main/data_raw>`_).
We used steps from four segments of the day and total sleep duration per day for each user.
For both cases, we conducted mean estimation using DAME-BS, Kent's and Girgis algorithm 500 times and report median squared error (for both scaled and unscaled values)
and average time taken by both algorithms. The results were limited in quality due to the small nummber of users 
and low number of samples per user.


For steps data, we had a total of 40 user with 88 samples per user. Here are the results - 

.. raw:: html

   <div style="margin: 1em 0; text-align: center;">

.. image:: ../../experiments/real_data_experiments/globem/results/globem_steps.png
   :alt: MSE for DAME-BS, Kent and Girgis
   :align: center
   :width: 500px


.. raw:: html

   <div style="margin: 2em 0;">

For sleep data, we had a total of 40 users with 15 samples per user. Here are the results - 


.. raw:: html

   <div style="margin: 2em 0;">

.. image:: ../../experiments/real_data_experiments/globem/results/globem_sleep.png
   :alt: MSE for DAME-BS, Kent and Girgis
   :align: center
   :width: 500px

.. raw:: html

   <div style="margin: 2em 0;">