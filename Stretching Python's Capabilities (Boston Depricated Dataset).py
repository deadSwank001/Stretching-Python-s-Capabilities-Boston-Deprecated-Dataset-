# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:48:50 2023

@author: swank
"""

Playing with Scikit-learn
Defining applications for data science
#http://scikit-learn.org/stable/developers/
#http://scikit-learn.org/stable/faq.html

from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data,boston.target
print("X:%s y:%s" % (X.shape, y.shape))
from sklearn.linear_model import LinearRegression
hypothesis = LinearRegression(normalize=True)
hypothesis.fit(X, y)
print(hypothesis.coef_)
import numpy as np
new_observation = np.array([1, 0, 1, 0, 0.5, 7, 59, 
                            6, 3, 200, 20, 350, 4], 
                           dtype=float).reshape(1, -1)
print(hypothesis.predict(new_observation))
hypothesis.score(X, y)

#Well this is a horrible Dataset to try and run

#######################################################################
ImportError                               Traceback (most recent call last)
Cell In[1], line 1
----> 1 from sklearn.datasets import load_boston
      2 boston = load_boston()
      3 X, y = boston.data,boston.target

File C:\ProgramData\anaconda3\lib\site-packages\sklearn\datasets\__init__.py:156, in __getattr__(name)
    105 if name == "load_boston":
    106     msg = textwrap.dedent(
    107         """
    108         `load_boston` has been removed from scikit-learn since version 1.2.
   (...)
    154         """
    155     )
--> 156     raise ImportError(msg)
    157 try:
    158     return globals()[name]

ImportError: 
`load_boston` has been removed from scikit-learn since version 1.2.

The Boston housing prices dataset has an ethical problem: as
investigated in [1], the authors of this dataset engineered a
non-invertible variable "B" assuming that racial self-segregation had a
positive impact on house prices [2]. Furthermore the goal of the
research that led to the creation of this dataset was to study the
impact of air quality but it did not give adequate demonstration of the
validity of this assumption.

The scikit-learn maintainers therefore strongly discourage the use of
this dataset unless the purpose of the code is to study and educate
about ethical issues in data science and machine learning.

In this special case, you can fetch the dataset from the original
source::

    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

Alternative datasets include the California housing dataset and the
Ames housing dataset. You can load the datasets as follows::

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

for the California housing dataset and::

    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True)

for the Ames housing dataset.

[1] M Carlisle.
"Racist data destruction?"
<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>

[2] Harrison Jr, David, and Daniel L. Rubinfeld.
"Hedonic housing prices and the demand for clean air."
Journal of environmental economics and management 5.1 (1978): 81-102.
<https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>

##################################################################################


# LinearRegression
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
print(scaler.transform(new_observation))
Performing the Hashing Trick
Demonstrating the hashing trick
print(hash('Python'))
print(abs(hash('Python')) % 1000)
from sklearn.feature_extraction.text import *
oh_enconder = CountVectorizer()
oh_enconded = oh_enconder.fit_transform([
'Python for data science','Python for machine learning'])
​
print(oh_enconder.vocabulary_)
string_1 = 'Python for data science'
string_2 = 'Python for machine learning'
​
def hashing_trick(input_string, vector_size=20):
    feature_vector = [0] * vector_size
    for word in input_string.split(' '):
        index = abs(hash(word)) % vector_size
        feature_vector[index] = 1
    return feature_vector
print(hashing_trick(
    input_string='Python for data science', 
    vector_size=20))
print(hashing_trick(
    input_string='Python for machine learning', 
    vector_size=20))
Working with deterministic selection
from scipy.sparse import csc_matrix
print(csc_matrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]))
http://scikit-learn.org/stable/modules/feature_extraction.html http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

import sklearn.feature_extraction.text as txt
htrick = txt.HashingVectorizer(n_features=20, 
                           binary=True, norm=None)
hashed_text = htrick.transform(['Python for data science',
                           'Python for machine learning'])
hashed_text
oh_enconder.transform(['New text has arrived']).todense()
htrick.transform(['New text has arrived']).todense()
Considering Timing and Performance
Benchmarking with timeit
%timeit l = [k for k in range(10**6)]
%timeit -n 20 -r 5 l = [k for k in range(10**6)]
%%timeit 
l = list()
for k in range(10**6):
    l.append(k)
import sklearn.feature_extraction.text as txt
htrick = txt.HashingVectorizer(n_features=20, 
                           binary=True, 
                           norm=None) 
oh_enconder = txt.CountVectorizer()
texts = ['Python for data science', 
         'Python for machine learning']
%timeit oh_enconded = oh_enconder.fit_transform(texts)
%timeit hashing = htrick.transform(texts)
import timeit
cumulative_time = timeit.timeit(
    "hashing = htrick.transform(texts)", 
    "from __main__ import htrick, texts", 
    number=10000)
print(cumulative_time / 10000.0)
Working with the memory profiler
# Installation procedures
import sys
!{sys.executable} -m pip install memory_profiler
# Initialization from IPython (to be repeat at every IPython start)
%load_ext memory_profiler
hashing = htrick.transform(texts)
%memit dense_hashing = hashing.toarray()
%%writefile example_code.py
def comparison_test(text):
    import sklearn.feature_extraction.text as txt
    htrick = txt.HashingVectorizer(n_features=20, 
                                   binary=True, 
                                   norm=None) 
    oh_enconder = txt.CountVectorizer()
    oh_enconded = oh_enconder.fit_transform(text)
    hashing = htrick.transform(text)
    return oh_enconded, hashing
from example_code import comparison_test
text = ['Python for data science',
        'Python for machine learning']
%mprun -f comparison_test comparison_test(text)
Running in Parallel on Multiple Cores
Demonstrating multiprocessing
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data,digits.target
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
%timeit single_core = cross_val_score(SVC(), X, y, \
                                      cv=20, n_jobs=1)
%timeit multi_core = cross_val_score(SVC(), X, y, \
                                     cv=20, n_jobs=-1)