
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

================  ==============  ===========
    Algorithm      Scaled Error     Time (s)
================  ==============  ===========
    DAME-BS           0.0015        0.00023
    Kent’s            0.0050        0.00066  
    Girgis            0.0246        0.00320  
================  ==============  ===========

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

================  ==============  ===========
    Algorithm      Scaled Error     Time (s)
================  ==============  ===========
    DAME-BS           0.0194        0.00023
    Kent’s            0.0242        0.00066  
    Girgis            0.1028        0.00320  
================  ==============  ===========

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

   <div style="text-align: center;">

================  ==============  ===========
    Algorithm      Scaled Error     Time (s)
================  ==============  ===========
    DAME-BS           0.3842        0.00025
    Kent’s            0.4985        0.00069  
    Girgis            0.0353        0.00417  
================  ==============  ===========

.. raw:: html

   </div>

For sleep data, we had a total of 40 users with 15 samples per user. Here are the results - 


.. raw:: html

   <div style="text-align: center;">
   
================  ==============  ===========
    Algorithm      Scaled Error     Time (s)
================  ==============  ===========
    DAME-BS           0.0327        0.00045
    Kent’s            0.0423        0.00118  
    Girgis            0.1483        0.00844  
================  ==============  ===========

.. raw:: html

   </div>

We note that Girgis’ algorithm generally performs the worst across datasets, but achieves superior
performance on the step-count data. This can be due to the discrete and highly clustered nature of
step-count data. Note that we used step counts across four segments of the day which makes it highly
clustered. Girgis’ algorithm is particularly well-suited to this regime. Its Range_scalar step identifies the
dominant cluster through Hadamard-based frequency estimation, and the subsequent Mean_user step yields
a robust estimate once the correct concentration interval is located. In contrast, Kent’s method, while also
bin-based, does not apply such a transform and instead depends on noisy user votes for localization. Both
Kent’s method and DAME-BS are designed to perform better when the data distribution is smoother
or less clustered, whereas Girgis benefits in this specific discrete setting. Nonetheless, across real world
datasets more broadly, DAME-BS consistently matches or outperforms the alternatives.