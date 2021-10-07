LFM
================
Haolin Zhong (UNI: hz2771)
2021/10/6

# LFM ALM + Gradient Descent Implementation

## Import dependencies

``` python
import numpy as np
import pandas as pd
```

## Data preparation

``` python
# user-item rating matrix
R = np.array([[4, 0, 2, 0, 1],
            [0, 2, 3, 0, 0],
            [1, 0, 2, 4, 0],
            [5, 0, 0, 3, 1],
            [0, 0, 1, 5, 1],
            [0, 3, 2, 4, 1]])
```

## Algorithm implementation

``` python
# input args
# R: M*N rating matrix
# K: number of latent factor dimensions
# max_iter: maximum recursion round
# alpha: stepwise
# lamda: regularization coefficient

# output
# P: M*K user feature matrix
# Q: N*K item feature matrix


K = 5
max_iter = 5000
alpha = 0.0002
lamda = 0.004

# core algorithms

def lfm_grad_desc(R, K, max_iter, alpha, lamda):
    M = R.shape[0]
    N = R.shape[1]
    
    # initialize P, Q
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    
    Q = Q.T
    
    # start recursion
    for step in range(max_iter):
        # traverse all users and items, perform gradient descent
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    e_ui = np.dot(P[u, :], Q[:, i]) - R[u][i]
                    # update P_u, Q_i by formula
                    for k in range(K):
                        P[u][k] -= alpha * (2 * e_ui * Q[k][i] + 2 * lamda * P[u][k])
                        Q[k][i] -= alpha * (2 * e_ui * P[u][k] + 2 * lamda * Q[k][i])
        
        # traverse completed
        R_pred = np.dot(P, Q)
    
        # calculate loss function
        cost = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                  cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                  for k in range(K):
                      cost += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
                
        if cost < 0.0001:
            break
        
    return P, Q.T, cost
    
```

## Test

``` python
P, Q, cost = lfm_grad_desc(R, K, max_iter, alpha, lamda)

R_pred = P.dot(Q.T)

print(R_pred)
```

    ## [[3.97815649 2.39367937 1.97620961 3.82952129 1.08981087]
    ##  [3.199646   2.02787983 2.99030665 3.89959724 1.13053142]
    ##  [1.03236561 1.91282151 1.87800914 4.03002843 0.83852504]
    ##  [4.99688918 2.69002215 2.07773261 2.99706775 0.9724181 ]
    ##  [1.58500091 2.36436537 1.1083554  4.93524048 1.01496775]
    ##  [2.99409011 2.95380936 2.00860295 4.02930509 0.92117355]]

``` python
print(cost)
```

    ## 0.5789715281242696
