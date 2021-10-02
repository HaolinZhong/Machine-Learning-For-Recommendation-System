Linear Regression
================
Haolin Zhong (UNI: hz2771)
2021/10/1

# Unary Linear Regression: Core ideas

-   Find
    ![f(x\_i) = wx\_i + b](https://latex.codecogs.com/png.latex?f%28x_i%29%20%3D%20wx_i%20%2B%20b "f(x_i) = wx_i + b"),
    making
    ![f(x\_i) \\simeq y\_i](https://latex.codecogs.com/png.latex?f%28x_i%29%20%5Csimeq%20y_i "f(x_i) \simeq y_i")

-   Least square method:
    ![\\displaystyle (w^\*, b^\*) = arg\\space min \_{(w, b)} \\sum\_{i=1}^{m} (f(x\_i) - y\_i)^2](https://latex.codecogs.com/png.latex?%5Cdisplaystyle%20%28w%5E%2A%2C%20b%5E%2A%29%20%3D%20arg%5Cspace%20min%20_%7B%28w%2C%20b%29%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28f%28x_i%29%20-%20y_i%29%5E2 "\displaystyle (w^*, b^*) = arg\space min _{(w, b)} \sum_{i=1}^{m} (f(x_i) - y_i)^2")

-   Loss function:
    ![E(w, b) = \\displaystyle \\sum\_{i=1}^{m} (f(x\_i) - y\_i)^2](https://latex.codecogs.com/png.latex?E%28w%2C%20b%29%20%3D%20%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28f%28x_i%29%20-%20y_i%29%5E2 "E(w, b) = \displaystyle \sum_{i=1}^{m} (f(x_i) - y_i)^2")

-   When
    ![\\frac {\\partial E(w, b)}{\\partial w} = 0](https://latex.codecogs.com/png.latex?%5Cfrac%20%7B%5Cpartial%20E%28w%2C%20b%29%7D%7B%5Cpartial%20w%7D%20%3D%200 "\frac {\partial E(w, b)}{\partial w} = 0"),
    ![\\frac {\\partial E(w, b)}{\\partial b} = 0](https://latex.codecogs.com/png.latex?%5Cfrac%20%7B%5Cpartial%20E%28w%2C%20b%29%7D%7B%5Cpartial%20b%7D%20%3D%200 "\frac {\partial E(w, b)}{\partial b} = 0"),
    min loss achieved. (the derivatives are monotonically increasing)

    -   ![w=\\frac{\\sum\_{i=1}^{m} y\_{i}\\left(x\_{i}-\\bar{x}\\right)}{\\sum\_{i=1}^{m} x\_{i}^{2}-\\frac{1}{m}\\left(\\sum\_{i=1}^{m} x\_{i}\\right)^{2}}](https://latex.codecogs.com/png.latex?w%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20y_%7Bi%7D%5Cleft%28x_%7Bi%7D-%5Cbar%7Bx%7D%5Cright%29%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20x_%7Bi%7D%5E%7B2%7D-%5Cfrac%7B1%7D%7Bm%7D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20x_%7Bi%7D%5Cright%29%5E%7B2%7D%7D "w=\frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{m} x_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} x_{i}\right)^{2}}")
    -   ![b=\\frac{1}{m} \\displaystyle \\sum\_{i=1}^{m}\\left(y\_{i}-w x\_{i}\\right)](https://latex.codecogs.com/png.latex?b%3D%5Cfrac%7B1%7D%7Bm%7D%20%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28y_%7Bi%7D-w%20x_%7Bi%7D%5Cright%29 "b=\frac{1}{m} \displaystyle \sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)")

# Practice

``` python
import numpy as np
import matplotlib.pyplot as plt
```

## Import data

``` python
points = np.genfromtxt("../Data/data.csv", delimiter=',')

X = points[:, 0]
Y = points[:, 1]

plt.scatter(X,Y)
plt.show()
```

<img src="Linear-Regression_files/figure-gfm/unnamed-chunk-2-1.png" width="672" />

## Define cost function

``` python
def compute_cost(w, b, points):
    X = points[:, 0]
    Y = points[:, 1]
    
    costs = (Y - w * X - b) ** 2
    
    return np.mean(costs)
```

## Define fit function

``` python
def fit(points):
    X = points[:, 0]
    Y = points[:, 1]
    N = len(points)
    
    YX = np.sum(Y * (X - np.mean(X)))
    X2 = np.sum(X**2)
    
    w = YX / (X2 - N*(np.mean(X)**2))
    
    b = np.mean(Y - w*X)
    
    return w, b
```

## Test

``` python
w, b = fit(points)

print("w is " + str(w) + ", " + "b is " + str(b))
```

    ## w is 1.32243102275536, b is 7.9910209822703795

``` python
print("cost is " + str(compute_cost(w, b, points)))
```

    ## cost is 110.25738346621316

## Visualize the fit line

``` python
pred_Y = w * X + b

plt.close()
plt.scatter(X, Y)
plt.plot(X, pred_Y, c = 'r')
plt.show()
```

<img src="Linear-Regression_files/figure-gfm/unnamed-chunk-6-3.png" width="672" />

# Solving Multiple Linear Regression by Gradient Descent

-   ![\\quad h\_{\\theta}(x)=X^{T} \\theta=\\sum\_{k=0}^{n} \\theta\_{k} x\_{k}](https://latex.codecogs.com/png.latex?%5Cquad%20h_%7B%5Ctheta%7D%28x%29%3DX%5E%7BT%7D%20%5Ctheta%3D%5Csum_%7Bk%3D0%7D%5E%7Bn%7D%20%5Ctheta_%7Bk%7D%20x_%7Bk%7D "\quad h_{\theta}(x)=X^{T} \theta=\sum_{k=0}^{n} \theta_{k} x_{k}")
-   ![J(\\theta)=\\frac{1}{m} \\sum\_{i=1}^{m}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right)^{2}](https://latex.codecogs.com/png.latex?J%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D "J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}")
-   ![\\theta\_{n+1}=\\theta\_{n}-\\alpha \\cdot \\frac{\\partial J(\\theta\_{n})}{\\partial \\theta\_n}](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bn%2B1%7D%3D%5Ctheta_%7Bn%7D-%5Calpha%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta_%7Bn%7D%29%7D%7B%5Cpartial%20%5Ctheta_n%7D "\theta_{n+1}=\theta_{n}-\alpha \cdot \frac{\partial J(\theta_{n})}{\partial \theta_n}")
-   ![\\begin{aligned} \\frac{\\partial J(\\theta)}{\\partial \\theta\_{j}} &=\\frac{\\partial}{\\partial \\theta\_{j}} \\frac{1}{m} \\sum\_{i=1}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right)^{2} \\\\ &=2 \\cdot \\frac{1}{m} \\sum\_{i=1}^{m}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right) \\cdot \\frac{\\partial}{\\partial \\theta\_{j}}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right) \\\\ &=\\frac{2}{m} \\sum\_{i=1}^{m}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right) \\cdot \\frac{\\partial}{\\partial \\theta\_{j}}\\left(\\sum\_{k=0}^{n} \\theta\_{k} x\_{k}^{(i)}-y^{(i)}\\right) \\\\ &=\\frac{2}{m} \\sum\_{i=1}^{m}\\left(h\_{\\theta}\\left(x^{(i)}\\right)-y^{(i)}\\right) \\cdot x\_{j}^{(i)} \\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_%7Bj%7D%7D%20%26%3D%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bj%7D%7D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D%20%5C%5C%20%26%3D2%20%5Ccdot%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%20%5Ccdot%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bj%7D%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%20%5C%5C%20%26%3D%5Cfrac%7B2%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%20%5Ccdot%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bj%7D%7D%5Cleft%28%5Csum_%7Bk%3D0%7D%5E%7Bn%7D%20%5Ctheta_%7Bk%7D%20x_%7Bk%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%29%20%5C%5C%20%26%3D%5Cfrac%7B2%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29%20%5Ccdot%20x_%7Bj%7D%5E%7B%28i%29%7D%20%5Cend%7Baligned%7D "\begin{aligned} \frac{\partial J(\theta)}{\partial \theta_{j}} &=\frac{\partial}{\partial \theta_{j}} \frac{1}{m} \sum_{i=1}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2} \\ &=2 \cdot \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\ &=\frac{2}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(\sum_{k=0}^{n} \theta_{k} x_{k}^{(i)}-y^{(i)}\right) \\ &=\frac{2}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{j}^{(i)} \end{aligned}")
