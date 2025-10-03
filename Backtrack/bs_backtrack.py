import numpy as np
import math
import warnings


def attempting_insertion_using_bs_backtrack(alpha,delta,n,m,user_samples,tau=0.5):
   
    # --- Input validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if user_samples.shape != (n, m):
        raise ValueError(f"user_samples must be a 2D array of shape ({n}, {m})")

    if n % 2 != 0:
        warnings.warn(f"n = {n} is odd; reducing user samples to {n - 1} to make it even.")
        n -= 1
        user_samples = user_samples[:n]
    
    if not isinstance(delta, (int, float)) or delta<=0 or delta>1:
        raise ValueError("delta must be a positive number less than 1")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    
    if not np.all((user_samples >= -1) & (user_samples <= 1)):
        raise ValueError("All entries must lie in [-1, 1]")
        
    # check m is greater than or equal to 7
    if m< 7:
        warnings.warn(f"m = {m} is below the recommended minimum 7; result may be unreliable.")

      
    # Precomputing the probability of truthful response under randomized response
    if alpha == np.inf:
        pi_alpha=1
    else:
        pi_alpha = np.exp(alpha/2) / (1 + np.exp(alpha/2))

    denom = np.log(8 * m / 9*math.log(12)) 
    b1 = (2 * n * np.log(3 / 2)) / denom
    b2 = (2 / (((2 * pi_alpha - 1) ** 2) + 1e-3)) * np.log(2 * n / (delta +1e-3))
    b_max = int(math.ceil(min(b1, b2)))
    b_max = min(b_max, n)
    
    t_max = n // b_max

    # Initializing search interval [L, R]
    L, R = -1.0, 1.0
    p = pi_alpha 




    #################################################################
    #################################################################

    # Threshold tau for the backtracking check
    tau = (1 - pi_alpha)


    # Maximum number of ternary-search rounds
    t_max = max(n // b_max, 1) # ensure at least one round


    # Initialize interval
    L, R = -1.0, 1.0
    L_prev, R_prev = L, R
    LR_list=[]
    LR_list.append([L,R])


    # Precompute each group's start/end indices to behave like Z_t sets
    group_indices = [(t * b_max, min((t + 1) * b_max, n)) for t in range(t_max)]
    # Performing t_max iterations of private ternary search
    for t, (start, end) in enumerate(group_indices):

        group = user_samples[start:end]
        if group.shape[0] == 0:
            # No more users in this group (shouldn't normally happen) -> stop
            break

        backtrack =  False

        # --- Backtracking check for t > 0 ---
        if t > 0:
            
            c_tilde = -1
            x_bars = np.mean(group, axis=1)
            C = (x_bars >= L) & (x_bars <= R)
           
            flips = np.random.rand(len(C)) < pi_alpha
            c_tilde = np.sum(np.where(flips, C, 1 - C))

            c_tilde = c_tilde/b_max



            

            if c_tilde < tau:
                # Not enough evidence that the current interval contains the mass -> backtrack
                if LR_list!=[]:
                    LR_list.pop()
                    [L, R] = LR_list[-1]
                else:
                    L,R=-1,1
                    
                # print("backtrack done.")
                
                backtrack=True
                
        
        if not backtrack:
            

            # Length of each third
            gamma = (R - L) / 3.0
            I1_L, I1_R = L, L + gamma  # left interval
            I3_L, I3_R = R - gamma, R  # right interval

            # Initialize noisy counts for left and right intervals
            V1_tilde = 0
            V3_tilde = 0

            # group = user_samples[start: end]
            x_bars = np.mean(group, axis=1)

            V1 = (I1_L <= x_bars) & (x_bars <= I1_R)
            V3 = (I3_L <= x_bars) & (x_bars <= I3_R)

            # Randomized response
            flips = np.random.rand(len(V1)) < pi_alpha
            V1_tilde = np.sum(np.where(flips, V1, 1 - V1))
            V3_tilde = np.sum(np.where(flips, V3, 1 - V3))


            # Discard the interval (I1 or I3) with smaller noisy count
            if V1_tilde < V3_tilde:
                # Drop the left third: shift L to the start of I2
                L += gamma
            else:
                # Drop the right third: shift R to the end of I2
                R -= gamma

            LR_list.append([L,R])

   
    return [L,R]

    