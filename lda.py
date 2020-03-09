from sklearn import datasets
import pandas as pd
import numpy as np

cancer=datasets.load_breast_cancer()
data=pd.DataFrame(cancer.data,columns=cancer.feature_names)
y=cancer.target
data['target']=y
data.head()

d1=data.loc[data.target==0]
d1=d1.drop(columns=['target'])
d2=data.loc[data.target==1]
d2=d2.drop(columns=['target'])

m1 = d1.mean()
m2 = d2.mean()

d1 = d1-m1
d2 = d2-m2

s1 = sum([np.matmul( i.reshape(len(i),1), i.reshape(1,len(i))) for i in d1.values])
s2 = sum([np.matmul( i.reshape(len(i),1), i.reshape(1,len(i))) for i in d2.values])

sw = s1+s2
sb = np.matmul((m1.values-m2.values).reshape(len(m1),1),(m1.values-m2.values).reshape(1,len(m1)))

A = np.matmul(np.linalg.inv(sw),sb)

eV,eVe = np.linalg.eig(A)
eVe = eVe.T
eV,eVe = zip(*sorted(zip(eV,eVe),reverse=True))

k=1
eVe = eVe[:k]
print(eVe)