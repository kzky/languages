#!/usr/bin/env python

import pandas as pd
import numpy as np

# time index
dates = pd.date_range("20130101", periods=6)
for date in dates:
    print date
    pass

# dateframe
data = np.random.randn(6, 4)
data = pd.DataFrame(data, index=dates, columns=list("ABCD"))
print data

print data.loc["20130101":"20130103", ['A', 'B']]     # slicing (first:last fashion)

# different dtypes for each column (=label)
df = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'})

print df.index         # only rows
print df.columns   # only columns
print df.values       # only data (ndarray)

print df.describe()  # summarize data for each column; too useful
print df.T                  # transpose

print df.sort_index(axis=1, ascending=False)   # sort by column (axis=1)
print df.sort(columns='B')                                     # sort by value of column

# group by
df = pd.DataFrame({
    "A": ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    "B": ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    "C": np.random.randn(8),
    "D": np.random.randn(8)})
df.groupby("A").sum()

# pivot-table
df = pd.DataFrame({
    'A': ['one', 'one', 'two', 'three'] * 3,
    'B': ['A', 'B', 'C'] * 4,
    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D': np.random.randn(12),
    'E': np.random.randn(12)})

print df
print pd.pivot_table(df, rows=['A', 'B'], cols=['C'], values='D')  # notaion is old


