TD-IDF
================
Haolin Zhong (UNI: hz2771)
2021/10/6

# Core ideas

## UGC-based recommendation

-   ![p(u, i) = \\displaystyle \\sum\_b n\_{u, b}n\_{b, i}](https://latex.codecogs.com/png.latex?p%28u%2C%20i%29%20%3D%20%5Cdisplaystyle%20%5Csum_b%20n_%7Bu%2C%20b%7Dn_%7Bb%2C%20i%7D "p(u, i) = \displaystyle \sum_b n_{u, b}n_{b, i}")

-   p: the interest that the user
    ![u](https://latex.codecogs.com/png.latex?u "u") holds for the item
    ![i](https://latex.codecogs.com/png.latex?i "i")

-   ![n\_{u, b}](https://latex.codecogs.com/png.latex?n_%7Bu%2C%20b%7D "n_{u, b}"):
    how many times the user gives tag b

-   ![n\_{b, i}](https://latex.codecogs.com/png.latex?n_%7Bb%2C%20i%7D "n_{b, i}"):
    how many times the item receives tag b

-   problem: may overweight popular tags

## TF-IDF

-   ![TF-IDF = TF \\times IDF](https://latex.codecogs.com/png.latex?TF-IDF%20%3D%20TF%20%5Ctimes%20IDF "TF-IDF = TF \times IDF")

-   The importance of the word should increase with increasing term
    frequency (TF) in its document, and decrease with increasing
    appearance in the overall library (which is described by Inverse
    Document Frequency, IDF) (for standardizing those innate highly
    appeared words) .

-   TF:
    ![TF\_{i,j} = \\frac {n\_{i, j}}{n\_{\*,j}}](https://latex.codecogs.com/png.latex?TF_%7Bi%2Cj%7D%20%3D%20%5Cfrac%20%7Bn_%7Bi%2C%20j%7D%7D%7Bn_%7B%2A%2Cj%7D%7D "TF_{i,j} = \frac {n_{i, j}}{n_{*,j}}")
    i: word i, j: doc j

-   IDF:
    ![IDF\_i = log(\\frac {N + 1}{N\_i + 1})](https://latex.codecogs.com/png.latex?IDF_i%20%3D%20log%28%5Cfrac%20%7BN%20%2B%201%7D%7BN_i%20%2B%201%7D%29 "IDF_i = log(\frac {N + 1}{N_i + 1})")

## UGC with TF-IDF

-   ![\\mathrm{p}(\\mathrm{u}, \\mathrm{i})=\\displaystyle \\sum\_{b} \\frac{n\_{u, b}}{\\log \\left(1+n\_{b}^{(u)}\\right)} \\frac{n\_{b, i}}{\\log \\left(1+n\_{i}^{(u)}\\right)}](https://latex.codecogs.com/png.latex?%5Cmathrm%7Bp%7D%28%5Cmathrm%7Bu%7D%2C%20%5Cmathrm%7Bi%7D%29%3D%5Cdisplaystyle%20%5Csum_%7Bb%7D%20%5Cfrac%7Bn_%7Bu%2C%20b%7D%7D%7B%5Clog%20%5Cleft%281%2Bn_%7Bb%7D%5E%7B%28u%29%7D%5Cright%29%7D%20%5Cfrac%7Bn_%7Bb%2C%20i%7D%7D%7B%5Clog%20%5Cleft%281%2Bn_%7Bi%7D%5E%7B%28u%29%7D%5Cright%29%7D "\mathrm{p}(\mathrm{u}, \mathrm{i})=\displaystyle \sum_{b} \frac{n_{u, b}}{\log \left(1+n_{b}^{(u)}\right)} \frac{n_{b, i}}{\log \left(1+n_{i}^{(u)}\right)}")

-   ![n\_b^{(u)}](https://latex.codecogs.com/png.latex?n_b%5E%7B%28u%29%7D "n_b^{(u)}"):
    tag b has been used by how many users

-   ![n\_i^{(u)}](https://latex.codecogs.com/png.latex?n_i%5E%7B%28u%29%7D "n_i^{(u)}"):
    item has been tagged by how many users

# TF-IDF Practice

## Import denpendencies

``` python
import numpy as np
import pandas as pd
import math
```

## Define and preprocess data

``` python
doc_a = "The cat sat on my bed"
doc_b = "The dog sat on my knees"

bow_a = doc_a.split(" ")
bow_b = doc_b.split(" ")

# Construct wordset
word_set = set(bow_a).union(set(bow_b))
```

## Perform word count

``` python
word_dict_a = dict.fromkeys(word_set, 0)
word_dict_b = dict.fromkeys(word_set, 0)

for word in bow_a:
    word_dict_a[word] += 1

for word in bow_b:
    word_dict_b[word] += 1
    
```

## Calculate TF

``` python
def compute_tf(word_dict, bow):
    tf_dict = {}
    nbow_count = len(bow)
    
    for word, count in word_dict.items():
        tf_dict[word] = count / nbow_count
    
    return tf_dict
```

## Calculate IDF

``` python
def compute_idf(word_dict_list):
    idf_dict = dict.fromkeys(word_dict_list[0], 0)
    # number of documents
    N = len(word_dict_list)
    
    for word_dict in word_dict_list:
        for word, count in word_dict.items():
            if count > 0:
                idf_dict[word] += 1
    
    for word, ni in idf_dict.items():
        idf_dict[word] = math.log10((N + 1) / (ni + 1))
    
    return idf_dict
```

## Calculate TF-IDF

``` python
def compute_tf_idf(tf, idf):
    tf_idf = {}
    for word, tf_val in tf.items():
        tf_idf[word] = tf_val * idf[word]
    return tf_idf
    
```

## Test

``` python
tf_a = compute_tf(word_dict_a, bow_a)
tf_b = compute_tf(word_dict_b, bow_b)

idf = compute_idf([word_dict_a, word_dict_b])

tf_idf_a = compute_tf_idf(tf_a, idf)
tf_idf_b = compute_tf_idf(tf_b, idf)

pd.DataFrame([tf_idf_a, tf_idf_b])
```

    ##    The       bed       cat       dog     knees   my   on  sat
    ## 0  0.0  0.029349  0.029349  0.000000  0.000000  0.0  0.0  0.0
    ## 1  0.0  0.000000  0.000000  0.029349  0.029349  0.0  0.0  0.0
