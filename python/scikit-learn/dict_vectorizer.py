#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer
http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation
"""

from sklearn.feature_extraction import DictVectorizer
measurements = [
    {'city': 'Dubai', 'temperature': 31.0, 'country': 'U.A.E.'},
    {'city': 'London', 'country': 'U.K.', 'temperature': 27.0},
    {'city': 'San Fransisco', 'country': 'U.S.', 'temperature': 24.0},
]
vec = DictVectorizer()
# 一回fit
print vec.fit(measurements)

# to sparse matrix and to dense
print vec.fit_transform(measurements).toarray()

# feature name
print vec.get_feature_names()

# indexing result
print vec.vocabulary_

# vocabularyにないデータは空扱い
print vec.transform({'city': 'Cambridge', 'country': 'U.K.', 'temperature': 19.0})

# vocabularyにあればちゃんと１がはいる
print vec.transform({'city': 'London', 'country': 'U.K.', 'temperature': 19.0})
