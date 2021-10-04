Logistic Regression
================
Haolin Zhong (UNI: hz2771)
2021/10/3

# Core ideas

-   Find a best curve as the boundary between target categories
-   sigmoid function:
    ![g(z)=\\frac{1}{1+e^{-z}}](https://latex.codecogs.com/png.latex?g%28z%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D "g(z)=\frac{1}{1+e^{-z}}");
    ![z = h\_\\theta(x)](https://latex.codecogs.com/png.latex?z%20%3D%20h_%5Ctheta%28x%29 "z = h_\theta(x)");
    ![h\_\\theta(x)](https://latex.codecogs.com/png.latex?h_%5Ctheta%28x%29 "h_\theta(x)")
    can be a high order fucntion

![](Logistic-Regression_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

-   Loss function:
    -   MSE: cannot guanranteed to be a convex function
    -   loss shouldn’t be defined as
        ![g(Z) - Y](https://latex.codecogs.com/png.latex?g%28Z%29%20-%20Y "g(Z) - Y"),
        because in such definition there is basically no difference
        being right or wrong nearing z = 0
    -   The idea cost function should drastically decrease nearing 0,
        then slowly decrease with abs(z) grows. A log function do meet
        this demand, and it is also able to simplify the exponential
        function
    -   ![\\operatorname{Cost}\\left(h\_{\\theta}(x), y\\right)=\\left\\{\\begin{aligned}-\\log \\left(h\_{\\theta}(x)\\right) & \\text { if } y=1 \\\\-\\log \\left(1-h\_{\\theta}(x)\\right) & \\text { if } y=0 \\end{aligned}\\right.](https://latex.codecogs.com/png.latex?%5Coperatorname%7BCost%7D%5Cleft%28h_%7B%5Ctheta%7D%28x%29%2C%20y%5Cright%29%3D%5Cleft%5C%7B%5Cbegin%7Baligned%7D-%5Clog%20%5Cleft%28h_%7B%5Ctheta%7D%28x%29%5Cright%29%20%26%20%5Ctext%20%7B%20if%20%7D%20y%3D1%20%5C%5C-%5Clog%20%5Cleft%281-h_%7B%5Ctheta%7D%28x%29%5Cright%29%20%26%20%5Ctext%20%7B%20if%20%7D%20y%3D0%20%5Cend%7Baligned%7D%5Cright. "\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0 \end{aligned}\right.")
    -   ![-\\frac{1}{m} \\sum\_{i=1}^{m}\\left\[y^{(i)} \\log h\_{\\theta}\\left(x^{(i)}\\right)+\\left(1-y^{(i)}\\right) \\log \\left(1-h\_{\\theta}\\left(x^{(i)}\\right)\\right)\\right\]](https://latex.codecogs.com/png.latex?-%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%5By%5E%7B%28i%29%7D%20%5Clog%20h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%2B%5Cleft%281-y%5E%7B%28i%29%7D%5Cright%29%20%5Clog%20%5Cleft%281-h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%29%5Cright%5D "-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]")
    -   To get a convex function and avoid overfit, add a regularization
        term:
        ![J(\\theta)=-\\frac{1}{m} \\sum\_{i=1}^{m}\\left\[y^{(i)} \\log h\_{\\theta}\\left(x^{(i)}\\right)+\\left(1-y^{(i)}\\right) \\log \\left(1-h\_{\\theta}\\left(x^{(i)}\\right)\\right)\\right\]+\\frac{\\lambda}{2 m} \\sum\_{j=1}^{n} \\theta\_{j}^{2}](https://latex.codecogs.com/png.latex?J%28%5Ctheta%29%3D-%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%5By%5E%7B%28i%29%7D%20%5Clog%20h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%2B%5Cleft%281-y%5E%7B%28i%29%7D%5Cright%29%20%5Clog%20%5Cleft%281-h_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%29%5Cright%5D%2B%5Cfrac%7B%5Clambda%7D%7B2%20m%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20%5Ctheta_%7Bj%7D%5E%7B2%7D "J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}")
-   Gradient descent:
    -   ![\\theta\_{j}:=\\theta\_{j}-\\alpha \\frac{\\partial}{\\partial \\theta\_{j}} J(\\theta)](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bj%7D%3A%3D%5Ctheta_%7Bj%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bj%7D%7D%20J%28%5Ctheta%29 "\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)")
    -   Omit calculating the deriatives.
