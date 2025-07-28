import numpy as np

def kent_mean_estimator(X, alpha, K=1.0):

    """
    Implements the Kent's algorithm for mean estimation under user level LDP when 
    the data is in l_inf ball of radius 1.
    
    Parameters
    ----------
        X : array, shape (n, T) 
            User Samples in [-1,1]^d
        alpha : float
            Privacy Parameter
        K : float (>=0)
            Constant (Default = 1.0).
            
    Returns
    --------
        theta_hat : float
            final estimated mean
    """

    # --- Input validation ---
    n, T = X.shape
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer") 
    if not isinstance(K, (int, float)) or K <= 0:
        raise ValueError("K must be a positive number")
    if not isinstance(T, int) or T <= 0:
        raise ValueError("Number of samples must be a positive integer")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if not isinstance(X, (list, tuple,np.ndarray)) or len(X) != n:
        raise ValueError(f"user_samples must be a list of length {n}")
    if not np.all((X >= -1) & (X <= 1)):
        raise ValueError("All entries of X must lie in [-1, 1]")
    for i, sample in enumerate(X):
        if not hasattr(sample, '__len__') or len(sample) != T:
            raise ValueError(f"Each user sample must be an array-like of length {T}")
    
    
    # to prevent overflow, checking if arg is very high
    LOG_MAX = np.log(np.finfo(float).max)  
    arg = n * alpha**2 / (K )
    arg_safe=min(LOG_MAX,arg)
    T_star = min(T, int(np.exp(arg_safe)) + 1)
    
    delta = np.sqrt(2 * np.log(n * T_star * alpha**2) / T_star)
    
    # Partition [−1,1] into intervals of width 2δ
    intervals = partition_interval(-1.0, 1.0, delta)  
    # intervals is a list of (L_j, U_j) for all j
    
   
    n1 = n // 2
    theta_loc = X[:n1, :T_star].mean(axis=1)  
    
    V = np.zeros((n1, len(intervals)), dtype=int)
    for i in range(n1):
        for j, (L, U) in enumerate(intervals):
            # union of neighbor intervals
            neigh = intervals[max(j-1,0):min(j+2,len(intervals))]
            if any(L_i <= theta_loc[i] < U_i for (L_i, U_i) in neigh):
                V[i,j] = 1
    
    p = np.exp(alpha/6)/(1 + np.exp(alpha/6))
    U_rand = np.random.rand(n1, len(intervals))
    Ve = np.where(U_rand <= p, V, 1 - V)
    
    j_star = np.argmax(Ve.sum(axis=0))
    L_star, U_star = intervals[j_star]
    L_tilde, U_tilde = L_star - 6*delta, U_star + 6*delta
    
    in_interval = lambda x: min(max(x, L_tilde), U_tilde)
    theta_refined = []
    for i in range(n1, n):
       
        clipped = in_interval(X[i,:T_star].mean())
        noise = np.random.laplace( scale=(14*delta/alpha))
        theta_i = clipped + noise
        theta_refined.append(theta_i)
    
    # Final estimate: average of the refined n/2 users
    theta_hat = np.mean(theta_refined)
    return theta_hat


def partition_interval(a, b, delta):
    """
    Partitions a closed interval [a, b] into sub-intervals of length 2 * delta.
    This creates a list of non-overlapping intervals whose union covers [a, b]:


      I_1 = [a,      a + 2·δ),

      I_2 = [a+2·δ,  a + 4·δ),

      ... 

      I_n = [a+2·δ·(n-1), b]   (where the final upper bound is clipped to b).

    Parameters
    ----------
        a : float
            Left endpoint of the interval.
        b : float
            Right endpoint of the interval. 
        delta : float
            Desired half length of the sub-interval.

    Returns
    -------
        intervals : list of tuple of floats
            A list `[(L₁,U₁), (L₂,U₂), …, (L_N,U_N)]` where:
            - U_j - L_j = 2·delta for all j < N,
            - The last interval may be shorter so that U_N = b
    """
    N = int(np.ceil((b - a) / (2*delta)))
    intervals = []
    for j in range(N):
        L = a + 2*delta * j
        U = min(a + 2*delta * (j+1), b)
        intervals.append((L, U))
    return intervals

