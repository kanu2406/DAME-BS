.. dame_bs documentation master file, created by
   sphinx-quickstart on Tue Jun 17 21:27:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dame_bs documentation
================================


Overview
--------

The **dame_bs** package provides a set of functions for private mean estimation under user level 
Local Differential Privacy (LDP) using the algorithm called DAME-BS (Distribution-Aware Mean Estimation
under User-level Local Differential Privacy using Binary Search) for univariate and multivariate cases.
The algorithm is structured as follows:

   - **Localization Phase:** The first half of users are used to identify an interval likely to contain the true mean. This is done privately using a binary search mechanism.
   - **Estimation Phase:** The second half of users project their sample averages onto the localized interval and laplacian noise is added to it. The final estimate is the average of all noisy reports.

Theoretical guarantees for univariate case
------------------------------------------

   - The number of samples per user `m` must satisfy:

   .. math::

      m \geq \frac{1}{2}\left(\frac{3}{2}\right)^4 \ln(12) \approx 6.28

   - Under this assumption, the expected mean squared error of the estimate is bounded by:

   .. math::
      :nowrap:

      \begin{aligned}
      \mathbb{E}\bigl[(\hat{\theta}-\theta)^2\bigr]
      &\le 
      \frac{389}{nm\alpha^2}\Bigl\{\ln\bigl(2nm\alpha^2\bigr)\vee 1\Bigr\}\\
      &\quad
      +16\max\!\Bigl\{\,2n\exp\!\Bigl(-\tfrac{n(2\pi_\alpha-1)^2}{2}\Bigr),\;
      \frac{9}{\alpha\sqrt{8}}\exp\!\Bigl(-\tfrac{1}{4}(2\pi_\alpha-1)\sqrt{n}\Bigr)\Bigr\}
      \end{aligned}


   where `α` is the privacy parameter, `n` is the number of users, and `π_α` is a function of α.

   

.. rubric:: Contents of this package - 

.. toctree::
   :maxdepth: 1
 
   dame_bs
   kent
   girgis
   experiments
   results
   tests

