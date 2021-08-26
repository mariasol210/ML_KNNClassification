import numpy as np 
import pandas as pd 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_csv('iris.data')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'])

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)





