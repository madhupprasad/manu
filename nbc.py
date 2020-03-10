from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split

data = arff.loadarff('/content/drive/My Drive/Colab Notebooks/dbworld_bodies_stemmed.arff')
df = pd.DataFrame(data[0])
df = df.astype(int)

result = df['CLASS']
df=df.drop(columns=['CLASS'])

y_train.head()

from sklearn.naive_bayes import BernoulliNB

gnb = BernoulliNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))