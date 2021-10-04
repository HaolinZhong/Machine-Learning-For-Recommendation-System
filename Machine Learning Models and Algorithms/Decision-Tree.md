Decision Tree
================
Haolin Zhong (UNI: hz2771)
2021/10/4

# Core ideas

-   Decision Tree algorithm selects features based on finding maximum
    Information Gain ![g](https://latex.codecogs.com/png.latex?g "g").

-   ![g(D\|A) = H(D) - H(D\|A)](https://latex.codecogs.com/png.latex?g%28D%7CA%29%20%3D%20H%28D%29%20-%20H%28D%7CA%29 "g(D|A) = H(D) - H(D|A)")

-   entrophy:
    ![H(X)=-\\sum\_{i=1}^{n} p\_{i} \\log p\_{i}](https://latex.codecogs.com/png.latex?H%28X%29%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20p_%7Bi%7D%20%5Clog%20p_%7Bi%7D "H(X)=-\sum_{i=1}^{n} p_{i} \log p_{i}"),
    used to evaluating how random a random variable can be.

-   conditional entrophy:
    ![H(Y \\mid X)=\\displaystyle \\sum\_{i=1}^{n} p\_{i} H\\left(Y \\mid X=x\_{i}\\right)](https://latex.codecogs.com/png.latex?H%28Y%20%5Cmid%20X%29%3D%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20p_%7Bi%7D%20H%5Cleft%28Y%20%5Cmid%20X%3Dx_%7Bi%7D%5Cright%29 "H(Y \mid X)=\displaystyle \sum_{i=1}^{n} p_{i} H\left(Y \mid X=x_{i}\right)"),
    the randomness of ![Y](https://latex.codecogs.com/png.latex?Y "Y")
    given known ![X](https://latex.codecogs.com/png.latex?X "X")

-   By choosing a good feature as classification criteria, the entrophy
    will decrease(lower uncertainty).

-   some algorithms:

    -   ID3: use above algorithm
    -   C4.5: use Information Gain Proportion to select features
        (standardized information gain for features can be classified
        into more categories)
    -   CART: consist of feature selection, generate, prune
