import pandas as pd
import numpy as np

df = pd.read_csv("iris.data", header=None)

uni = df[4].unique()
print(uni)
df = df.replace(uni,[0,1,2])


result = df[4]
df = df.drop(columns=[4])
print(df)

initial_data = np.array(df)
df = df-df.mean()

n_arr = np.array(df)
n_arr = n_arr.transpose()
cov = np.cov(n_arr,bias=1)

w, v =np.linalg.eig(cov)
w, v = zip(*sorted(zip(w,v),reverse=True))

D_v = np.array(v[0:2]).transpose()

New_X = initial_data.dot(D_v)
#print(New_X)