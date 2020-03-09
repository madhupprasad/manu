# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:49:17 2020

@author: madhu
"""

import pandas as pd
import numpy as np

def gradientAscent(x,y_result,w,lr,cnt):
  for i in range(cnt):
    grad = 0
    for j in range(len(y_result)):
      sig = 1/(1+np.exp(-np.matmul(w,x[j])))
      print(sig)
      input()
      grad = grad+lr*(y_result[j]-sig)*x[j]
      print(grad)
    w=w+grad
  return w

def sigmoid(x,w):
    return 1/(1+np.exp(-np.dot(w,x)))

df = pd.read_csv('iris.data',header=None)
print(df.head())

unique = df[4].unique()
df = df.replace(unique,[0,1,2])
print(df.head())

y_result = df[4]
df = df.drop(columns=[4])
print(df.head())

n_arr = np.array(df)

ones = np.ones((len(y_result),1))
x_arr = np.append(ones,n_arr,axis=1)
w = np.zeros((1,len(x_arr[0])))[0]
lr = 0.3
print(np.dot(w,x_arr[0]))
w = gradientAscent(x_arr,y_result,w,lr,50)
print(w)
