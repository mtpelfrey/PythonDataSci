import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
names = ["Sepal Length", "Sepal Width", "Petal Lenth", "Petal Width", "Class"])

print(df.head())
print(df.describe())
df.hist(bins=20)
#plt.show()

data_array = df.values
np.random.shuffle(data_array)
X_learning = data_array[:80][:,0:4]
Y_learning = data_array[:80][:,4]

svc = SVC()
svc.fit(X_learning,Y_learning)

X = data_array[-20:][:,0:4]
Y = data_array[-20:][:,4]

predictions = svc.predict(X)

print("Predicted results: ")
print(predictions)
print("Acutal Results: ")
print(Y)
print("Accuracy rate: %f" % (accuracy_score(Y, predictions)))

print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))