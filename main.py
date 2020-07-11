import pandas as pd 
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from perceptron import Perceptron

DATASET_IRIS_PATH = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

print("esercizio 1")
print("loading iris dataset from" , DATASET_IRIS_PATH)

df = pd.read_csv(DATASET_IRIS_PATH , header=None , encoding='utf-8')

print("dataset loaded")
print("dataset has " , df.shape[0] , " examples")

assert df.shape[0] > 0 , 'incorrect dataset loaded'

num_examples = 100

y = df.iloc[0:num_examples , 4].values
y = np.where(y == 'Iris-setosa' , -1 , 1)

X = df.iloc[0:num_examples , [0 , 2]].values

plt.scatter(X[:50 , 0] , X[:50 , 1] , color="red" , marker="o" , label="setosa")
plt.scatter(X[50:num_examples , 0] , X[50:num_examples , 1] , color="blue" , marker="x" , label="versicolor")

plt.title("tipo di fiore per caratteristiche stelo/petali")
plt.xlabel("lunghezza stelo [cm]")
plt.ylabel("lunghezza petali [cm]")
plt.legend(loc="upper left")

plt.show()

ppn = Perceptron(X.shape[0])

ppn.fit(X , y)

plt.plot(range(1 , len))